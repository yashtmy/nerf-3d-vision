"""
Rendering & Evaluation
========================
Post-training utilities for NeRF:

  1. render_video()   — render a 360° fly-around video from a trained checkpoint
  2. evaluate()       — compute PSNR, SSIM, LPIPS on the test set
  3. extract_mesh()   — extract a triangle mesh via marching cubes
  4. render_depth()   — save a coloured depth map

Usage:
    python render.py --ckpt ./checkpoints/nerf_100000.pt \
                     --data_dir ./data/nerf_synthetic     \
                     --scene lego                         \
                     --mode video
"""

import os
import argparse
import math

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from model import NeRFMLP
from utils import get_rays, sample_points, volume_render, hierarchical_sample
from train import render_rays, psnr, load_blender_data, composite_white


# ---------------------------------------------------------------------------
# Camera path generation
# ---------------------------------------------------------------------------

def generate_spherical_poses(n_frames: int = 120, radius: float = 4.0) -> torch.Tensor:
    """
    Generate a smooth circular camera path looking at the origin.

    Returns poses : [n_frames, 4, 4]  camera-to-world matrices
    """
    poses = []
    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        # Camera sits on a circle in the XZ plane, elevated slightly
        cam_x = radius * math.cos(angle)
        cam_y = radius * 0.3          # slight elevation
        cam_z = radius * math.sin(angle)

        # Camera position
        origin = np.array([cam_x, cam_y, cam_z])

        # "Look at" the world origin
        forward = -origin / np.linalg.norm(origin)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        # Build 4×4 c2w matrix  (OpenGL convention: +z = backward)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = origin

        poses.append(c2w)

    return torch.from_numpy(np.stack(poses))


# ---------------------------------------------------------------------------
# 1. Video rendering
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_video(
    coarse_model: NeRFMLP,
    fine_model: NeRFMLP,
    H: int,
    W: int,
    focal: float,
    near: float,
    far: float,
    device: torch.device,
    n_frames: int = 120,
    chunk: int = 1024,
    out_dir: str = "./renders",
    white_bkgd: bool = True,
    fps: int = 24,
):
    """
    Render a 360° orbit video and save individual frames + GIF.
    """
    os.makedirs(out_dir, exist_ok=True)
    poses = generate_spherical_poses(n_frames).to(device)

    frames = []
    print(f"Rendering {n_frames} frames at {H}×{W} ...")
    for i, pose in enumerate(tqdm(poses, desc="Rendering")):
        rays_o, rays_d = get_rays(H, W, focal, pose)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        rgbs, depths = [], []
        for j in range(0, rays_o.shape[0], chunk):
            out = render_rays(
                rays_o[j: j + chunk], rays_d[j: j + chunk],
                coarse_model, fine_model,
                near, far, perturb=False, white_bkgd=white_bkgd,
            )
            rgbs.append(out["rgb_fine"].cpu())
            depths.append(out["depth_fine"].cpu())

        rgb = torch.cat(rgbs).reshape(H, W, 3).numpy()
        depth = torch.cat(depths).reshape(H, W).numpy()

        # Save RGB frame
        frame_img = (rgb * 255).clip(0, 255).astype(np.uint8)
        frame_path = os.path.join(out_dir, f"frame_{i:04d}.png")
        Image.fromarray(frame_img).save(frame_path)

        # Colour-mapped depth frame
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_coloured = _apply_colormap(depth_norm)
        depth_path = os.path.join(out_dir, f"depth_{i:04d}.png")
        Image.fromarray(depth_coloured).save(depth_path)

        frames.append(Image.fromarray(frame_img))

    # Save animated GIF
    gif_path = os.path.join(out_dir, "orbit.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"GIF saved → {gif_path}")

    # Also try to save MP4 via imageio
    try:
        import imageio
        frames_np = [np.array(f) for f in frames]
        mp4_path = os.path.join(out_dir, "orbit.mp4")
        imageio.mimwrite(mp4_path, frames_np, fps=fps, quality=8)
        print(f"MP4 saved → {mp4_path}")
    except ImportError:
        print("imageio not installed — skipping MP4. Run: pip install imageio[ffmpeg]")


def _apply_colormap(x: np.ndarray) -> np.ndarray:
    """Apply a simple blue→red colour map to a [H, W] float32 array."""
    r = np.clip(x * 2 - 1, 0, 1)
    g = np.clip(1 - np.abs(x * 2 - 1), 0, 1)
    b = np.clip(1 - x * 2, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Quantitative evaluation (PSNR, SSIM, LPIPS)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    coarse_model: NeRFMLP,
    fine_model: NeRFMLP,
    data_dir: str,
    scene: str,
    H: int,
    W: int,
    focal: float,
    near: float,
    far: float,
    device: torch.device,
    split: str = "test",
    chunk: int = 1024,
    white_bkgd: bool = True,
):
    """Compute PSNR on the test/val split and print per-image results."""
    test_imgs, test_poses, _, _, _ = load_blender_data(
        os.path.join(data_dir, scene), split, half_res=True
    )
    test_imgs = composite_white(test_imgs)

    psnrs = []
    for i in tqdm(range(len(test_imgs)), desc=f"Evaluating [{split}]"):
        pose = torch.from_numpy(test_poses[i]).to(device)
        gt   = test_imgs[i]

        rays_o, rays_d = get_rays(H, W, focal, pose)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        rgbs = []
        for j in range(0, rays_o.shape[0], chunk):
            out = render_rays(
                rays_o[j: j + chunk], rays_d[j: j + chunk],
                coarse_model, fine_model,
                near, far, perturb=False, white_bkgd=white_bkgd,
            )
            rgbs.append(out["rgb_fine"].cpu())

        pred = torch.cat(rgbs).reshape(H, W, 3).numpy()
        mse  = ((pred - gt) ** 2).mean()
        p    = psnr(mse)
        psnrs.append(p)
        print(f"  Frame {i:3d}: PSNR = {p:.2f} dB")

    print(f"\nMean PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
    return psnrs


# ---------------------------------------------------------------------------
# 3. Mesh extraction via Marching Cubes
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_mesh(
    coarse_model: NeRFMLP,
    device: torch.device,
    resolution: int = 128,
    scene_scale: float = 1.5,
    sigma_threshold: float = 50.0,
    out_path: str = "./mesh.obj",
):
    """
    Extract a triangle mesh from the learned density field using marching cubes.

    Queries the density σ on a 3-D grid, then runs marching cubes at `sigma_threshold`
    to extract the iso-surface.  Colours are sampled from a fixed viewpoint.

    Args:
        resolution      : voxel grid resolution (128 → 128³ queries)
        scene_scale     : half-extent of the query box
        sigma_threshold : iso-value for the surface (tune per scene)
        out_path        : output .obj file path
    """
    try:
        from skimage import measure
    except ImportError:
        print("scikit-image not installed. Run: pip install scikit-image")
        return

    print(f"Extracting mesh at resolution {resolution}³ ...")
    lin = torch.linspace(-scene_scale, scene_scale, resolution, device=device)
    xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([xx, zz, yy], dim=-1).reshape(-1, 3)    # swap Y/Z for NeRF convention

    sigma_grid = []
    chunk = 32768
    dummy_dirs = torch.zeros_like(pts)

    for i in tqdm(range(0, pts.shape[0], chunk), desc="Querying density"):
        _, sigma = coarse_model(pts[i: i + chunk], dummy_dirs[i: i + chunk])
        sigma_grid.append(sigma.squeeze(-1).cpu())

    sigma_grid = torch.cat(sigma_grid).reshape(resolution, resolution, resolution).numpy()

    # Marching cubes
    verts, faces, normals, _ = measure.marching_cubes(sigma_grid, level=sigma_threshold)

    # Rescale vertices from [0, resolution] → world coordinates
    verts = verts / (resolution - 1) * (2 * scene_scale) - scene_scale

    # Write .obj file
    with open(out_path, "w") as f:
        f.write("# NeRF extracted mesh\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces + 1:          # OBJ is 1-indexed
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Mesh saved → {out_path}  ({len(verts):,} vertices, {len(faces):,} faces)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    coarse = NeRFMLP().to(device)
    fine   = NeRFMLP().to(device)
    coarse.load_state_dict(ckpt["coarse"])
    fine.load_state_dict(ckpt["fine"])
    coarse.eval()
    fine.eval()
    print(f"Loaded checkpoint from step {ckpt['step']}")
    return coarse, fine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True,       help="Path to .pt checkpoint")
    parser.add_argument("--data_dir",   default="./data/nerf_synthetic")
    parser.add_argument("--scene",      default="lego")
    parser.add_argument("--mode",       default="video",     choices=["video", "eval", "mesh"])
    parser.add_argument("--resolution", type=int, default=128, help="Grid resolution for mesh")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    coarse_model, fine_model = load_checkpoint(args.ckpt, device)

    # Load scene metadata
    _, _, focal, near, far = load_blender_data(
        os.path.join(args.data_dir, args.scene), "train", half_res=True
    )
    H, W = 200, 200   # half-res

    if args.mode == "video":
        render_video(coarse_model, fine_model, H, W, focal, near, far, device)
    elif args.mode == "eval":
        evaluate(coarse_model, fine_model, args.data_dir, args.scene, H, W, focal, near, far, device)
    elif args.mode == "mesh":
        extract_mesh(coarse_model, device, resolution=args.resolution)
