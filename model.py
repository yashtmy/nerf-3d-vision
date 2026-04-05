"""
NeRF Model Architecture
========================
Implements the Neural Radiance Field (NeRF) MLP as described in:
"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
Mildenhall et al., ECCV 2020 — https://arxiv.org/abs/2003.08934

Architecture:
  - Positional encoding for (x, y, z) and (θ, φ) directions
  - 8-layer MLP with skip connection at layer 4
  - Outputs: RGB colour + volume density (sigma)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Maps input coordinates to a higher-dimensional space using sinusoidal
    functions, allowing the MLP to represent high-frequency scene details.

    γ(p) = (p, sin(2⁰πp), cos(2⁰πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp))
    """

    def __init__(self, num_frequencies: int, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        # Precompute frequency bands: [1, 2, 4, ..., 2^(L-1)]
        freqs = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freqs", freqs)

    @property
    def output_dim(self) -> int:
        dim = 2 * self.num_frequencies  # sin + cos per frequency
        if self.include_input:
            dim += 1  # raw input passthrough
        return dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., D] — raw coordinates
        Returns:
            encoded: [..., D * output_dim]
        """
        parts = [x] if self.include_input else []
        for freq in self.freqs:
            parts.append(torch.sin(freq * x))
            parts.append(torch.cos(freq * x))
        return torch.cat(parts, dim=-1)


class NeRFMLP(nn.Module):
    """
    Core NeRF network: maps (position, direction) → (RGB, density).

    Two-stage design:
      1. Density network  : 8 fully-connected layers on encoded position
      2. Colour network   : 1 extra layer that also receives view direction
    """

    def __init__(
        self,
        pos_freq: int = 10,       # L for position encoding
        dir_freq: int = 4,        # L for direction encoding
        hidden_dim: int = 256,    # neurons per hidden layer
        skip_layer: int = 4,      # layer index where skip connection is added
    ):
        super().__init__()
        self.skip_layer = skip_layer

        # Positional encoders
        self.pos_enc = PositionalEncoding(pos_freq)
        self.dir_enc = PositionalEncoding(dir_freq)

        pos_dim = 3 * self.pos_enc.output_dim   # encoded x,y,z
        dir_dim = 3 * self.dir_enc.output_dim   # encoded θ,φ,ψ

        # ------------------------------------------------------------------
        # Density (σ) backbone — 8 layers with skip at layer `skip_layer`
        # ------------------------------------------------------------------
        self.density_layers = nn.ModuleList()
        in_dim = pos_dim
        for i in range(8):
            if i == skip_layer:
                in_dim += pos_dim   # concatenate original encoding again
            self.density_layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        # Raw density output (no activation — applied later during rendering)
        self.sigma_head = nn.Linear(hidden_dim, 1)

        # Feature vector passed to colour head
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)

        # ------------------------------------------------------------------
        # Colour (RGB) head — takes feature + view direction
        # ------------------------------------------------------------------
        self.colour_layer = nn.Linear(hidden_dim + dir_dim, hidden_dim // 2)
        self.rgb_head = nn.Linear(hidden_dim // 2, 3)

    def forward(
        self, positions: torch.Tensor, directions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions  : [N, 3]  — 3-D sample points along rays
            directions : [N, 3]  — unit viewing directions (same for all
                                    samples on the same ray)
        Returns:
            rgb   : [N, 3]   — predicted colour  ∈ [0, 1]
            sigma : [N, 1]   — volume density    ≥ 0
        """
        # Encode inputs
        pos_enc = self.pos_enc(positions)          # [N, 3*pos_out]
        dir_enc = self.dir_enc(directions)          # [N, 3*dir_out]

        # --- Density backbone ---
        x = pos_enc
        for i, layer in enumerate(self.density_layers):
            if i == self.skip_layer:
                x = torch.cat([x, pos_enc], dim=-1)
            x = F.relu(layer(x))

        sigma = F.relu(self.sigma_head(x))          # [N, 1]  density ≥ 0
        feat  = self.feature_head(x)                # [N, H]

        # --- Colour head (view-dependent) ---
        h = F.relu(self.colour_layer(torch.cat([feat, dir_enc], dim=-1)))
        rgb = torch.sigmoid(self.rgb_head(h))       # [N, 3]  ∈ [0, 1]

        return rgb, sigma
