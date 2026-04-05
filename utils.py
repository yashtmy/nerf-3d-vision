"""
Ray Generation & Volume Rendering
===================================
Implements the core 3D vision geometry primitives needed by NeRF:

  1. get_rays()           — generate camera rays from intrinsics + extrinsics
  2. sample_points()      — stratified + hierarchical sampling along rays
  3. volume_render()      — numerical integration of the rendering equation
  4. hierarchical_sample()— importance-sample using coarse density predictions
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Ray Generation
# ---------------------------------------------------------------------------

def get_rays(
    H: int,
    W: int,
    focal: float,
    c2w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a ray (origin + unit direction) for every pixel in an H×W image.

    Uses the pinhole camera model:
        pixel (u, v)  →  normalized image-plane coordinate
        camera-space direction d_c = K⁻¹ [u, v, 1]ᵀ
        world-space direction  d_w = R · d_c

    Args:
        H, W   : image height and width (pixels)
        focal  : focal length in pixels  (fx = fy assumed)
        c2w    : [4, 4] camera-to-world transformation matrix

    Returns:
        rays_o : [H, W, 3]  ray origins (camera centre in world space)
        rays_d : [H, W, 3]  unit ray directions in world space
    """
    device = c2w.device

    # Pixel grid — offset by 0.5 so (0,0) is the centre of the top-left pixel
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )

    # Normalised, image-plane coordinates (z = −1 for OpenGL convention)
    dirs = torch.stack(
        [
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,   # flip y: image ↓  =  camera ↑
            -torch.ones_like(i),
        ],
        dim=-1,
    )  # [H, W, 3]

    # Rotate directions into world space  (no translation for directions)
    rays_d = (dirs[..., None, :] * c2w[:3, :3]).sum(dim=-1)   # [H, W, 3]
    rays_d = F.normalize(rays_d, dim=-1)

    # All rays share the same origin = camera centre
    rays_o = c2w[:3, 3].expand(rays_d.shape)                  # [H, W, 3]

    return rays_o, rays_d


# ---------------------------------------------------------------------------
# 2. Stratified Sampling along Rays
# ---------------------------------------------------------------------------

def sample_points(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample 3-D points along rays using *stratified* sampling.

    The [near, far] interval is divided into n_samples equal bins.
    One random point is drawn from each bin during training (perturb=True),
    giving better coverage than a fixed grid.

    Args:
        rays_o    : [..., 3]  ray origins
        rays_d    : [..., 3]  unit directions
        near/far  : float      scene bounds
        n_samples : int        samples per ray
        perturb   : bool       add uniform noise within each bin

    Returns:
        pts : [..., n_samples, 3]  world-space sample positions
        t   : [..., n_samples]     depth values along each ray
    """
    # Evenly-spaced depth values in [near, far]
    t = torch.linspace(near, far, n_samples, device=rays_o.device)
    t = t.expand(*rays_o.shape[:-1], n_samples)                # [..., S]

    if perturb:
        # Random offset within each bin (stratified jitter)
        dt = (far - near) / n_samples
        t = t + torch.rand_like(t) * dt

    # r(t) = o + t·d
    pts = rays_o[..., None, :] + t[..., None] * rays_d[..., None, :]
    return pts, t


# ---------------------------------------------------------------------------
# 3. Volume Rendering  (core NeRF equation)
# ---------------------------------------------------------------------------

def volume_render(
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    t: torch.Tensor,
    rays_d: torch.Tensor,
    white_bkgd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Numerically integrate the volume-rendering equation:

        C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt

    where   T(t) = exp(−∫₀ᵗ σ(r(s)) ds)   is the transmittance.

    Discrete approximation (quadrature):
        δᵢ    = tᵢ₊₁ − tᵢ          (segment length)
        αᵢ    = 1 − exp(−σᵢ δᵢ)    (opacity of segment i)
        Tᵢ    = ∏_{j<i} (1 − αⱼ)   (accumulated transmittance)
        Ĉ(r) = Σ Tᵢ αᵢ cᵢ

    Args:
        rgb    : [..., S, 3]   predicted colours at each sample
        sigma  : [..., S, 1]   predicted densities at each sample
        t      : [..., S]      depth values
        rays_d : [..., 3]      ray directions (for computing δ in world units)
        white_bkgd: bool       composite over a white background

    Returns:
        colour  : [..., 3]     rendered pixel colour
        depth   : [..., 1]     expected depth (weighted sum of t)
        weights : [..., S]     per-sample weights Tᵢαᵢ  (used for hier. sampling)
    """
    sigma = sigma.squeeze(-1)   # [..., S]

    # Segment lengths δᵢ = tᵢ₊₁ − tᵢ
    deltas = t[..., 1:] - t[..., :-1]                          # [..., S-1]
    # Last segment extends to "infinity"
    last_delta = torch.full(
        (*t.shape[:-1], 1), 1e10, device=t.device, dtype=t.dtype
    )
    deltas = torch.cat([deltas, last_delta], dim=-1)            # [..., S]

    # Scale by ray direction length so δ is in world-space units
    deltas = deltas * rays_d.norm(dim=-1, keepdim=True)

    # Alpha (opacity) at each sample
    alpha = 1.0 - torch.exp(-sigma * deltas)                   # [..., S]

    # Transmittance Tᵢ = ∏_{j<i}(1−αⱼ)  — exclusive product
    transmittance = torch.cumprod(
        torch.cat(
            [torch.ones((*alpha.shape[:-1], 1), device=alpha.device), 1.0 - alpha + 1e-10],
            dim=-1,
        ),
        dim=-1,
    )[..., :-1]

    weights = transmittance * alpha                              # [..., S]

    # Composite colour
    colour = (weights[..., None] * rgb).sum(dim=-2)            # [..., 3]

    # Expected depth
    depth = (weights * t).sum(dim=-1, keepdim=True)            # [..., 1]

    # Optionally composite over white background
    if white_bkgd:
        acc = weights.sum(dim=-1, keepdim=True)
        colour = colour + (1.0 - acc)

    return colour, depth, weights


# ---------------------------------------------------------------------------
# 4. Hierarchical (Importance) Sampling
# ---------------------------------------------------------------------------

def hierarchical_sample(
    t_coarse: torch.Tensor,
    weights: torch.Tensor,
    n_fine: int,
) -> torch.Tensor:
    """
    Sample additional points where the coarse network predicted high density.

    Converts coarse weights into a discrete PDF, then uses inverse-CDF
    sampling to place more samples in dense regions of the scene.

    Args:
        t_coarse : [..., S_c]   coarse depth values
        weights  : [..., S_c]   coarse sample weights (Tᵢαᵢ)
        n_fine   : int           number of additional samples

    Returns:
        t_fine   : [..., n_fine]  new depth values (unsorted)
    """
    # Mid-points of coarse segments
    t_mid = 0.5 * (t_coarse[..., 1:] + t_coarse[..., :-1])    # [..., S_c-1]
    w_mid = weights[..., 1:-1] + 1e-5                           # avoid zeros

    # Normalise to form a PDF
    pdf = w_mid / w_mid.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    # Draw uniform samples and invert the CDF
    u = torch.rand(*cdf.shape[:-1], n_fine, device=cdf.device)
    u = u.contiguous()

    # Binary search: find CDF intervals for each u
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds - 1).clamp(min=0)
    above = inds.clamp(max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)

    cdf_g = torch.gather(cdf, -1, inds_g.view(*inds_g.shape[:-2], -1)).view(*inds_g.shape)
    t_g   = torch.gather(t_mid, -1, inds_g.view(*inds_g.shape[:-2], -1)).view(*inds_g.shape)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    t_fine = t_g[..., 0] + (u - cdf_g[..., 0]) / denom * (t_g[..., 1] - t_g[..., 0])
    return t_fine
