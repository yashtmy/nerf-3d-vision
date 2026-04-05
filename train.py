"""
NeRF Training Script
=====================
Trains a Neural Radiance Field on the standard NeRF-Synthetic (Blender) dataset.

Usage (local):
    python train.py --scene lego --data_dir ./data/nerf_synthetic

Usage (Colab):
    See nerf_colab.py — it calls train_nerf() directly with correct paths.

Training strategy:
  - Random ray sampling per batch (not per image) for better convergence
  - Two-model hierarchy: coarse NeRF + fine NeRF (hierarchical sampling)
  - MSE photo-metric loss on both coarse and fine predictions
  - PSNR tracked every N iterations
  - Checkpoints saved to ./checkpoints/
"""

import os
import json
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm

from model import NeRFMLP
from utils import get_rays, sample_points, volume_render, hierarchical_sample


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_blender_data(base_dir: str, split: str = "train", half_res: bool = True):
    """
    Load the NeRF-Synthetic (Blender) dataset.

    Directory structure expected:
        <base_dir>/
            transforms_train.json
            transforms_val.json
            transforms_test.json
            train/  r_0.png  r_1.png  ...
            val/    ...
            test/   ...

    Returns:
        images : [N, H, W, 4]  RGBA float32 in [0, 1]
        poses  : [N, 4, 4]     camera-to-world matrices
        focal  : float          focal length in pixels (after half-res)
        near/far: scene bounds (fixed at 2.0 / 6.0 for Blender data)
    """
    meta_path = os.path.join(base_dir, f"transforms_{split}.json")
    with open(meta_path) as f:
        meta = json.load(f)

    images, poses = [], []
    for frame in meta["frames"]:
        img_path = os.path.join(base_dir, frame["file_path"] + ".png")
        img = np.array(Image.open(img_path)) / 255.0          # [H, W, 4]
        images.append(img)
        poses.append(np.array(frame["transform_matrix"]))

    images = np.stack(images).astype(np.float32)               # [N, H, W, 4]
    poses  = np.stack(poses).astype(np.float32)                # [N, 4, 4]

    H, W = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    if half_res:
        H, W, focal = H // 2, W // 2, focal / 2
        images_small = np.zeros((images.shape[0], H, W, 4), dtype=np.float32)
        for i, img in enumerate(images):
            pil = Image.fromarray((img * 255).astype(np.uint8))
            images_small[i] = np.array(pil.resize((W, H), Image.LANCZOS)) / 255.0
        images = images_small

    return images, poses, focal, 2.0, 6.0


def composite_white(images: np.ndarray) -> np.ndarray:
    """Blend RGBA over a white background → RGB."""
    alpha = images[..., 3:4]
    return images[..., :3] * alpha + (1.0 - alpha)


# ---------------------------------------------------------------------------
# Batch rendering
# ---------------------------------------------------------------------------

def render_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    coarse_model: NeRFMLP,
    fine_model: NeRFMLP,
    near: float,
    far: float,
    n_coarse: int = 64,
    n_fine: int = 128,
    white_bkgd: bool = True,
    perturb: bool = True,
) -> dict:
    """
    Full coarse-to-fine NeRF forward pass for a batch of rays.

    Returns a dict with keys:
        'rgb_coarse', 'depth_coarse'  — coarse predictions
        'rgb_fine',   'depth_fine'    — fine predictions  (primary output)
    """
    # ---- Coarse pass -------------------------------------------------------
    pts_c, t_c = sample_points(rays_o, rays_d, near, far, n_coarse, perturb)

    # Flatten for batched MLP forward
    N = rays_o.shape[0]
    dirs_c = rays_d[:, None, :].expand_as(pts_c).reshape(-1, 3)

    rgb_c, sigma_c = coarse_model(pts_c.reshape(-1, 3), dirs_c)
    rgb_c   = rgb_c.view(N, n_coarse, 3)
    sigma_c = sigma_c.view(N, n_coarse, 1)

    colour_c, depth_c, weights_c = volume_render(rgb_c, sigma_c, t_c, rays_d, white_bkgd)

    # ---- Fine pass (hierarchical sampling) ---------------------------------
    t_fine = hierarchical_sample(t_c.detach(), weights_c.detach(), n_fine)
    t_f, _ = torch.sort(torch.cat([t_c, t_fine], dim=-1), dim=-1)   # [N, S_c+S_f]

    pts_f = rays_o[:, None, :] + t_f[:, :, None] * rays_d[:, None, :]
    dirs_f = rays_d[:, None, :].expand_as(pts_f).reshape(-1, 3)

    rgb_f, sigma_f = fine_model(pts_f.reshape(-1, 3), dirs_f)
    rgb_f   = rgb_f.view(N, -1, 3)
    sigma_f = sigma_f.view(N, -1, 1)

    colour_f, depth_f, _ = volume_render(rgb_f, sigma_f, t_f, rays_d, white_bkgd)

    return {
        "rgb_coarse":   colour_c,
        "depth_coarse": depth_c,
        "rgb_fine":     colour_f,
        "depth_fine":   depth_f,
    }


def psnr(mse: float) -> float:
    return -10.0 * np.log10(mse + 1e-10)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_nerf(
    data_dir: str,
    scene: str = "lego",
    n_iters: int = 100_000,
    batch_size: int = 1024,
    lr: float = 5e-4,
    n_coarse: int = 64,
    n_fine: int = 128,
    log_every: int = 100,
    save_every: int = 5000,
    render_every: int = 2500,
    half_res: bool = True,
    white_bkgd: bool = True,
    device: str = "cuda",
    checkpoint_dir: str = "./checkpoints",
):
    """
    End-to-end NeRF training.

    Args:
        data_dir     : path to nerf_synthetic dataset root
        scene        : scene name, e.g. 'lego', 'hotdog', 'chair'
        n_iters      : number of gradient steps
        batch_size   : rays sampled per step
        lr           : Adam learning rate
        n_coarse     : coarse samples per ray
        n_fine       : fine samples per ray  (hierarchical)
        log_every    : print loss / PSNR every N iters
        save_every   : save checkpoint every N iters
        render_every : save a rendered validation image every N iters
        half_res     : halve image resolution  (400→200) for speed
        white_bkgd   : composite RGBA over white background
        device       : 'cuda' or 'cpu'
        checkpoint_dir: where to save .pt checkpoints
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # -- Load data -----------------------------------------------------------
    scene_dir = os.path.join(data_dir, scene)
    print(f"Loading scene '{scene}' from {scene_dir} ...")
    images, poses, focal, near, far = load_blender_data(scene_dir, "train", half_res)
    images = composite_white(images)                 # [N, H, W, 3]
    images_t = torch.from_numpy(images).to(device)
    poses_t  = torch.from_numpy(poses).to(device)
    H, W = images.shape[1:3]
    N_train = images.shape[0]

    val_imgs, val_poses, _, _, _ = load_blender_data(scene_dir, "val", half_res)
    val_imgs = composite_white(val_imgs)

    print(f"Loaded {N_train} training images  [{H}×{W}], focal={focal:.1f}")

    # Precompute all rays
    all_rays_o, all_rays_d, all_pixels = [], [], []
    for i in range(N_train):
        ro, rd = get_rays(H, W, focal, poses_t[i])
        all_rays_o.append(ro.reshape(-1, 3))
        all_rays_d.append(rd.reshape(-1, 3))
        all_pixels.append(images_t[i].reshape(-1, 3))

    all_rays_o = torch.cat(all_rays_o, dim=0)   # [N*H*W, 3]
    all_rays_d = torch.cat(all_rays_d, dim=0)
    all_pixels = torch.cat(all_pixels, dim=0)
    n_total = all_rays_o.shape[0]
    print(f"Total rays: {n_total:,}")

    # -- Models & optimizer --------------------------------------------------
    coarse_model = NeRFMLP(pos_freq=10, dir_freq=4).to(device)
    fine_model   = NeRFMLP(pos_freq=10, dir_freq=4).to(device)

    params = list(coarse_model.parameters()) + list(fine_model.parameters())
    optimizer = optim.Adam(params, lr=lr)

    # Cosine-decay learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=lr / 100)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("./renders", exist_ok=True)

    # -- Training ------------------------------------------------------------
    print("\n--- Starting training ---")
    t0 = time.time()
    perm = torch.randperm(n_total, device=device)
    cursor = 0

    for step in range(1, n_iters + 1):
        coarse_model.train()
        fine_model.train()

        # Reshuffle when we exhaust all rays
        if cursor + batch_size > n_total:
            perm = torch.randperm(n_total, device=device)
            cursor = 0

        idx = perm[cursor: cursor + batch_size]
        cursor += batch_size

        rays_o = all_rays_o[idx]
        rays_d = all_rays_d[idx]
        target = all_pixels[idx]

        out = render_rays(
            rays_o, rays_d, coarse_model, fine_model,
            near, far, n_coarse, n_fine, white_bkgd,
        )

        loss_c = ((out["rgb_coarse"] - target) ** 2).mean()
        loss_f = ((out["rgb_fine"]   - target) ** 2).mean()
        loss   = loss_c + loss_f

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ---- Logging -------------------------------------------------------
        if step % log_every == 0:
            elapsed = time.time() - t0
            train_psnr = psnr(loss_f.item())
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[{step:6d}/{n_iters}]  loss={loss_f.item():.4f}  "
                f"PSNR={train_psnr:.2f} dB  lr={lr_now:.2e}  "
                f"time={elapsed:.0f}s"
            )

        # ---- Render validation image ---------------------------------------
        if step % render_every == 0:
            _render_val(
                coarse_model, fine_model,
                val_imgs[0], torch.from_numpy(val_poses[0]).to(device),
                H, W, focal, near, far, n_coarse, n_fine, white_bkgd, device,
                save_path=f"./renders/val_{step:06d}.png",
            )

        # ---- Save checkpoint -----------------------------------------------
        if step % save_every == 0:
            ckpt = {
                "step": step,
                "coarse": coarse_model.state_dict(),
                "fine":   fine_model.state_dict(),
                "optim":  optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(checkpoint_dir, f"nerf_{step:06d}.pt"))
            print(f"  Checkpoint saved at step {step}")

    print("\nTraining complete!")
    return coarse_model, fine_model


@torch.no_grad()
def _render_val(
    coarse_model, fine_model,
    gt_img, pose, H, W, focal, near, far,
    n_coarse, n_fine, white_bkgd, device,
    chunk=1024, save_path="./renders/val.png",
):
    """Render a single validation image and save it (with ground-truth side-by-side)."""
    coarse_model.eval()
    fine_model.eval()

    rays_o, rays_d = get_rays(H, W, focal, pose)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    rgbs = []
    for i in range(0, rays_o.shape[0], chunk):
        out = render_rays(
            rays_o[i: i + chunk], rays_d[i: i + chunk],
            coarse_model, fine_model,
            near, far, n_coarse, n_fine, white_bkgd, perturb=False,
        )
        rgbs.append(out["rgb_fine"].cpu())

    pred = torch.cat(rgbs, dim=0).reshape(H, W, 3).numpy()
    gt   = gt_img

    # Side-by-side comparison
    side_by_side = np.concatenate([gt, pred], axis=1)
    img = (side_by_side * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)

    val_mse  = ((pred - gt) ** 2).mean()
    val_psnr = psnr(val_mse)
    print(f"  ↳ Val PSNR = {val_psnr:.2f} dB  →  {save_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeRF on NeRF-Synthetic dataset")
    parser.add_argument("--data_dir",    default="./data/nerf_synthetic")
    parser.add_argument("--scene",       default="lego")
    parser.add_argument("--n_iters",     type=int, default=100_000)
    parser.add_argument("--batch_size",  type=int, default=1024)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--n_coarse",    type=int, default=64)
    parser.add_argument("--n_fine",      type=int, default=128)
    parser.add_argument("--half_res",    action="store_true", default=True)
    parser.add_argument("--no_half_res", dest="half_res", action="store_false")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--ckpt_dir",    default="./checkpoints")
    args = parser.parse_args()

    train_nerf(
        data_dir=args.data_dir,
        scene=args.scene,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        n_coarse=args.n_coarse,
        n_fine=args.n_fine,
        half_res=args.half_res,
        device=args.device,
        checkpoint_dir=args.ckpt_dir,
    )
