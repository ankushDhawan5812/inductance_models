"""
Differentiable physics manifold for inductance-position relationship.

Loads a precomputed overlap-area manifold (from conv_fit) and provides
differentiable bilinear interpolation lookup:  (x, y) → expected_inductance.

Used as a physics consistency loss during training: if the model predicts
position (x, y), the manifold says what inductance should be observed there.
Penalizing mismatch between predicted and observed inductance constrains the
model to produce physically plausible positions.

The manifold values are in arbitrary overlap-area units. A learnable affine
transform (scale + offset) maps them to match the actual inductance scale.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsManifold(nn.Module):
    """
    Differentiable lookup table: (x, y) → expected inductance.

    Uses bilinear interpolation on a precomputed 2D grid. Gradients flow
    back through the (x, y) inputs, so this can be used in a loss that
    penalizes positions whose expected inductance doesn't match observation.

    The manifold is stored as a fixed buffer (not learned). Two learnable
    parameters (scale, offset) map manifold values to actual inductance:
        predicted_inductance = scale * manifold(x, y) + offset

    Args:
        manifold_path: path to .npz file with keys 'xs', 'ys', 'overlap'
    """

    def __init__(self, manifold_path: str):
        super().__init__()

        data = np.load(manifold_path)
        xs = data['xs'].astype(np.float32)
        ys = data['ys'].astype(np.float32)
        overlap = data['overlap'].astype(np.float32)

        # Grid metadata for coordinate → grid index conversion
        self.register_buffer('x_min', torch.tensor(xs[0]))
        self.register_buffer('x_max', torch.tensor(xs[-1]))
        self.register_buffer('y_min', torch.tensor(ys[0]))
        self.register_buffer('y_max', torch.tensor(ys[-1]))

        # Store as (1, 1, H, W) for grid_sample
        # grid_sample expects (N, C, H, W) input
        # overlap shape is (ny, nx) where ny=rows(y), nx=cols(x)
        grid = torch.from_numpy(overlap).unsqueeze(0).unsqueeze(0)
        self.register_buffer('grid', grid)

        # Learnable affine transform: manifold_value → inductance
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Look up expected inductance at positions (x, y).

        Args:
            xy: (batch, 2) — positions in mm (x, y)

        Returns:
            (batch,) — predicted inductance values
        """
        # Normalize coordinates to [-1, 1] for grid_sample
        # grid_sample uses (x, y) where x indexes width (cols) and y indexes height (rows)
        x = xy[:, 0]
        y = xy[:, 1]

        x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_norm = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0

        # Clamp to grid bounds
        x_norm = x_norm.clamp(-1.0, 1.0)
        y_norm = y_norm.clamp(-1.0, 1.0)

        # grid_sample expects grid of shape (N, H_out, W_out, 2)
        # We want one sample per batch element → H_out=1, W_out=1
        sample_grid = torch.stack([x_norm, y_norm], dim=-1)  # (batch, 2)
        sample_grid = sample_grid.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, 2)

        # Expand manifold grid for batch: (batch, 1, H, W)
        batch_size = xy.size(0)
        grid_expanded = self.grid.expand(batch_size, -1, -1, -1)

        # Bilinear interpolation
        sampled = F.grid_sample(
            grid_expanded, sample_grid,
            mode='bilinear', padding_mode='border', align_corners=True,
        )
        # sampled shape: (batch, 1, 1, 1) → squeeze to (batch,)
        manifold_val = sampled.squeeze(-1).squeeze(-1).squeeze(-1)

        # Apply learnable affine transform
        return self.scale * manifold_val + self.offset
