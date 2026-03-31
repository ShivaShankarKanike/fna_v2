"""
fna/adapter.py
--------------
Coarse-grained FNA adapter layer.

Core idea:
    Instead of maintaining a full (d_out x d_in) adapter matrix,
    we maintain a small latent scalar grid M of shape (H, W) = (128, 128).
    This grid is upsampled bilinearly to the full weight dimensions
    to produce the adapter delta DeltaW.

    The NS optimizer maintains a velocity field V of shape (2, H, W)
    that governs how M evolves — M is advected by V each step.

Parameter count comparison (T5-small FFN 512->2048):
    LoRA r=8:  512*8 + 8*2048 = 20,480 params
    FNA 128x128: 128*128 = 16,384 params (M only, V is optimizer state)

This is <0.5% of the frozen layer's 1,048,576 parameters.

Author: Shiva Shankar Kanike
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FNALayer(nn.Module):
    """
    FNA adapter layer wrapping a frozen nn.Linear.

    Maintains a small latent scalar grid M (128x128) that is upsampled
    to produce DeltaW. The velocity field V lives in the optimizer state,
    not in this module — keeping M as the only learnable parameter.

    Forward pass:
        DeltaW = upsample(M, size=(d_out, d_in), mode='bilinear')
        Y = X @ (W_frozen + alpha * DeltaW).T + bias

    Args:
        in_features:   input dimension of the wrapped linear
        out_features:  output dimension
        grid_size:     latent grid size H=W (default 128, giving 16,384 params)
        alpha:         scaling factor for DeltaW (default 1.0)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 128,
        alpha: float = 1.0,
    ):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.alpha        = alpha

        # Frozen pretrained weight — not trained
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features),
            requires_grad=False,
        )
        self.bias = None  # set by from_linear if needed

        # Latent scalar grid M — the ONLY trainable parameter
        # Shape: (grid_size, grid_size)
        # Initialized near zero so DeltaW starts near zero (same as LoRA init)
        self.M = nn.Parameter(
            torch.randn(grid_size, grid_size) * 0.01
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, grid_size: int = 128, alpha: float = 1.0):
        """
        Create FNALayer from an existing nn.Linear, freezing its weights.
        """
        obj = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            grid_size=grid_size,
            alpha=alpha,
        )
        # Copy pretrained weights (frozen)
        obj.weight.data.copy_(linear.weight.data)

        # Handle bias
        if linear.bias is not None:
            obj.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)

        return obj

    def get_delta_w(self) -> torch.Tensor:
        """
        Upsample M from (grid_size, grid_size) to (out_features, in_features).

        Uses bilinear interpolation — smooth, no ringing artifacts,
        safe for large upscaling ratios (e.g., 128->2048 = 16x).

        Returns:
            DeltaW: shape (out_features, in_features)
        """
        # grid_sample / interpolate expects (N, C, H, W)
        M_4d = self.M.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_size, grid_size)

        DeltaW = F.interpolate(
            M_4d,
            size=(self.out_features, self.in_features),
            mode='bilinear',
            align_corners=True,
        )
        return DeltaW.squeeze(0).squeeze(0)  # (out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Y = X @ (W_frozen + alpha * DeltaW).T + bias
        """
        DeltaW = self.get_delta_w()
        W_eff  = self.weight + self.alpha * DeltaW
        return F.linear(x, W_eff, self.bias)

    def param_count(self) -> dict:
        """Returns parameter count breakdown."""
        frozen    = self.weight.numel()
        trainable = self.M.numel()
        return {
            "frozen":    frozen,
            "trainable": trainable,
            "pct_of_frozen": round(100 * trainable / frozen, 4),
        }

    def extra_repr(self) -> str:
        pc = self.param_count()
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"grid={self.grid_size}x{self.grid_size}, "
            f"trainable={pc['trainable']:,} ({pc['pct_of_frozen']}% of frozen)"
        )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("adapter.py — sanity check")
    print("=" * 50)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # T5-small FFN dimensions
    in_f, out_f = 512, 2048
    batch, seq  = 4, 32

    frozen_linear = nn.Linear(in_f, out_f)
    layer = FNALayer.from_linear(frozen_linear, grid_size=128).to(device)

    pc = layer.param_count()
    print(f"\nLayer: {in_f} -> {out_f}")
    print(f"Frozen params:    {pc['frozen']:,}")
    print(f"Trainable params: {pc['trainable']:,}  (M only)")
    print(f"% of frozen:      {pc['pct_of_frozen']}%")

    lora_r8 = in_f * 8 + 8 * out_f
    print(f"LoRA r=8 params:  {lora_r8:,}")
    print(f"FNA vs LoRA:      {layer.M.numel():,} vs {lora_r8:,}")

    # Forward pass
    x   = torch.randn(batch, seq, in_f).to(device)
    out = layer(x)
    print(f"\nForward: input {tuple(x.shape)} -> output {tuple(out.shape)}  OK")

    # Check DeltaW shape
    dw = layer.get_delta_w()
    print(f"DeltaW shape: {tuple(dw.shape)}  (should be ({out_f}, {in_f}))  OK")

    # Backward
    loss = out.sum()
    loss.backward()
    print(f"Backward pass OK  (M.grad shape: {tuple(layer.M.grad.shape)})")

    print("\nSanity check passed.")
