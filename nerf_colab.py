# %% — Cell 1: Install dependencies
# ─────────────────────────────────

# !pip install -q torch torchvision tqdm Pillow numpy scikit-image imageio imageio-ffmpeg

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for p in ["tqdm", "scikit-image", "imageio", "imageio-ffmpeg"]:
    install(p)

print("✅ Dependencies installed")


# %% — Cell 2: Clone the project repo (or set paths if running locally)
# ─────────────────────────────────────────────────────────────────────
import os, sys

# If running from a cloned GitHub repo, project files are already here.
# Otherwise, copy model.py / utils.py / train.py / render.py into /content.

PROJECT_DIR = "/content"    # Colab default working directory
os.chdir(PROJECT_DIR)

# Verify project files are present
for fname in ["model.py", "utils.py", "train.py", "render.py"]:
    if not os.path.exists(fname):
        print(f"⚠️  {fname} not found — upload it or clone the repo first.")
    else:
        print(f"✅ {fname}")


# %% — Cell 3: Download NeRF-Synthetic (Blender) dataset
# ──────────────────────────────────────────────────────
import urllib.request, zipfile, os

DATA_DIR = "/content/data/nerf_synthetic"
SCENE    = "lego"         
os.makedirs("/content/data", exist_ok=True)

if not os.path.exists(os.path.join(DATA_DIR, SCENE)):
    print("Downloading NeRF-Synthetic dataset (~770 MB) ...")
    url = "https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NeRF.zip"

    # This is the original NeRF dataset mirror; fallback below:
    nerf_url = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip"

    # Download with progress
    def reporthook(count, block_size, total_size):
        pct = count * block_size * 100 // total_size
        print(f"\r  {pct}%", end="")

    zip_path = "/content/data/nerf_synthetic.zip"
    urllib.request.urlretrieve(nerf_url, zip_path, reporthook)
    print("\nExtracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/content/data/")
    print(f"✅ Dataset extracted → {DATA_DIR}")
else:
    print(f"✅ Dataset already present at {DATA_DIR}")


# %% — Cell 4: Verify dataset & visualise sample images
# ──────────────────────────────────────────────────────
import json, numpy as np
from PIL import Image
import matplotlib.pyplot as plt

meta_path = os.path.join(DATA_DIR, SCENE, "transforms_train.json")
with open(meta_path) as f:
    meta = json.load(f)

print(f"Scene: {SCENE}")
print(f"Training frames: {len(meta['frames'])}")
print(f"camera_angle_x: {meta['camera_angle_x']:.4f} rad")

# Show 4 sample images
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, ax in enumerate(axes):
    img_path = os.path.join(DATA_DIR, SCENE, meta["frames"][i*10]["file_path"] + ".png")
    img = np.array(Image.open(img_path))
    ax.imshow(img)
    ax.set_title(f"Frame {i*10}")
    ax.axis("off")
plt.suptitle(f"NeRF-Synthetic: {SCENE}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("/content/sample_images.png", dpi=120, bbox_inches="tight")
plt.show()
print("✅ Sample images visualised")


# %% — Cell 5: Visualise camera poses
# ─────────────────────────────────────
from mpl_toolkits.mplot3d import Axes3D

poses = np.array([f["transform_matrix"] for f in meta["frames"]])

fig = plt.figure(figsize=(8, 8))
ax  = fig.add_subplot(111, projection="3d")

for p in poses[::5]:
    cam_center = p[:3, 3]
    cam_dir    = p[:3, 2]   # z-axis of camera in world space
    ax.scatter(*cam_center, color="royalblue", s=20)
    ax.quiver(*cam_center, *(-cam_dir * 0.2), color="tomato", length=1, normalize=True)

ax.scatter(0, 0, 0, color="gold", s=200, marker="*", label="Scene origin")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("Camera Poses (blue=position, red=view direction)")
ax.legend()
plt.tight_layout()
plt.savefig("/content/camera_poses.png", dpi=120, bbox_inches="tight")
plt.show()
print("✅ Camera poses visualised")


# %% — Cell 6: Configure training hyperparameters
# ─────────────────────────────────────────────────

CONFIG = dict(
    data_dir     = DATA_DIR,
    scene        = SCENE,
    n_iters      = 50_000,    # 50k steps (~2h on T4)  →  set to 10k for quick test
    batch_size   = 1024,      # rays per gradient step
    lr           = 5e-4,
    n_coarse     = 64,        # coarse samples per ray
    n_fine       = 128,       # fine (hierarchical) samples
    half_res     = True,      # 400→200px (faster training)
    white_bkgd   = True,
    log_every    = 200,
    save_every   = 10_000,
    render_every = 5_000,
    device       = "cuda",
    checkpoint_dir = "/content/checkpoints",
)

print("Training configuration:")
for k, v in CONFIG.items():
    print(f"  {k:20s}: {v}")


# %% — Cell 7: Run training
# ──────────────────────────
# ⏱  ~2 hours for 50k iters on a T4 GPU (15–20 min for 10k quick test)

from train import train_nerf

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs("/content/renders", exist_ok=True)

coarse_model, fine_model = train_nerf(**CONFIG)

print("\n Training complete")


# %% — Cell 8: Plot training curves
# ───────────────────────────────────
print("Training complete Check /content/renders/ for validation images.")
print("Use TensorBoard for live loss curves — see README for setup.")


# %% — Cell 9: Render novel views (360° orbit video)
# ───────────────────────────────────────────────────
import torch
from train import load_blender_data, composite_white
from render import render_video

_, _, focal, near, far = load_blender_data(
    os.path.join(DATA_DIR, SCENE), "train", half_res=True
)
H, W = 200, 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

render_video(
    coarse_model, fine_model,
    H, W, focal, near, far, device,
    n_frames=120,
    out_dir="/content/renders",
    fps=24,
)

# Show the GIF inline
from IPython.display import Image as IPImage
IPImage(filename="/content/renders/orbit.gif")


# %% — Cell 10: Quantitative evaluation on test set
# ───────────────────────────────────────────────────
from render import evaluate

coarse_model.eval(); fine_model.eval()

psnrs = evaluate(
    coarse_model, fine_model,
    DATA_DIR, SCENE,
    H, W, focal, near, far,
    device, split="test",
)

print(f"\n📊 Mean Test PSNR: {np.mean(psnrs):.2f} dB")
print("  (NeRF paper reports ~32–33 dB on the lego scene)")


# %% — Cell 11: Extract mesh via marching cubes
# ──────────────────────────────────────────────
from render import extract_mesh

extract_mesh(
    coarse_model, device,
    resolution=128,
    scene_scale=1.5,
    sigma_threshold=50.0,
    out_path="/content/mesh.obj",
)

# Download the mesh
from google.colab import files
files.download("/content/mesh.obj")
print("✅ Mesh downloaded — open in Blender / MeshLab to inspect.")


# %% — Cell 12: Visualise depth maps
# ────────────────────────────────────
import glob
depth_frames = sorted(glob.glob("/content/renders/depth_*.png"))

fig, axes = plt.subplots(1, min(4, len(depth_frames)), figsize=(16, 4))
for ax, path in zip(axes, depth_frames[:4]):
    ax.imshow(np.array(Image.open(path)))
    ax.set_title(os.path.basename(path))
    ax.axis("off")
plt.suptitle("Predicted Depth Maps", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/content/depth_maps.png", dpi=120, bbox_inches="tight")
plt.show()


# %% — Cell 13: Download results
# ────────────────────────────────
from google.colab import files

for f in ["/content/renders/orbit.gif",
          "/content/sample_images.png",
          "/content/camera_poses.png",
          "/content/depth_maps.png"]:
    if os.path.exists(f):
        files.download(f)
        print(f"Downloaded: {f}")

