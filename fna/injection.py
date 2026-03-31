"""
fna/injection.py
----------------
Utilities to inject FNA adapters into HuggingFace transformer models.

Recursively traverses any HuggingFace model and replaces target linear
layers with FNALayer wrappers. All base weights are frozen. Only the
latent grid M in each FNALayer is trainable.

Target modules for T5-small FFN layers:
    - DenseReluDense.wi   (512 -> 2048)
    - DenseReluDense.wo   (2048 -> 512)

For LLaMA-2/3 attention layers (future):
    - q_proj, v_proj, k_proj, o_proj

Author: Shiva Shankar Kanike
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .adapter import FNALayer


# Default target module names for T5-small FFN layers
T5_FFN_TARGETS = [
    "DenseReluDense.wi",
    "DenseReluDense.wo",
]

# Future: LLaMA attention targets
LLAMA_ATTN_TARGETS = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
]


def inject_fna_adapters(
    model: nn.Module,
    target_modules: List[str] = None,
    grid_size: int = 128,
    alpha: float = 1.0,
    verbose: bool = True,
) -> List[FNALayer]:
    """
    Recursively replace target nn.Linear modules with FNALayer adapters.

    Freezes all base model parameters. Only the latent grid M in each
    FNALayer is left trainable.

    Args:
        model:          any HuggingFace or PyTorch model
        target_modules: list of module name suffixes to replace
                        (e.g. ["DenseReluDense.wi", "DenseReluDense.wo"])
                        Defaults to T5 FFN targets.
        grid_size:      latent grid size for each FNALayer (default 128)
        alpha:          DeltaW scaling factor (default 1.0)
        verbose:        print injection summary

    Returns:
        fna_layers: list of all injected FNALayer instances
                    (useful for monitoring / optimizer setup)
    """
    if target_modules is None:
        target_modules = T5_FFN_TARGETS

    # Step 1 — freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2 — inject FNA layers at target locations
    fna_layers = []
    injected_names = []

    for full_name, module in model.named_modules():
        for target in target_modules:
            if full_name.endswith(target):
                if not isinstance(module, nn.Linear):
                    continue

                # Get parent module and the attribute name of this layer
                parts  = full_name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr = parts[-1]

                # Create FNALayer from this linear
                fna_layer = FNALayer.from_linear(module, grid_size=grid_size, alpha=alpha)

                # Replace in parent
                setattr(parent, attr, fna_layer)

                fna_layers.append(fna_layer)
                injected_names.append(full_name)
                break  # matched this module, move on

    if verbose:
        print(f"\nFNA Injection Summary")
        print(f"{'='*45}")
        print(f"Target modules:   {target_modules}")
        print(f"Layers replaced:  {len(fna_layers)}")
        print(f"Grid size:        {grid_size}x{grid_size} = {grid_size*grid_size:,} params each")
        print(f"Total FNA params: {len(fna_layers) * grid_size * grid_size:,}")
        print()
        for name in injected_names:
            print(f"  Injected: {name}")

        # Parameter summary
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameter summary:")
        print(f"  Total:     {total:,}")
        print(f"  Trainable: {trainable:,}  ({100*trainable/total:.4f}%)")
        print(f"  Frozen:    {total - trainable:,}")
        print(f"{'='*45}\n")

    return fna_layers


def get_fna_optimizer_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Return only the trainable FNA parameters (M grids) from the model.
    Pass this to FNAOptimizer instead of model.parameters().

    Usage:
        fna_params = get_fna_optimizer_params(model)
        optimizer  = FNAOptimizer(fna_params, dt=0.05, nu=0.1)
    """
    return [p for p in model.parameters() if p.requires_grad]


def print_model_summary(model: nn.Module):
    """Print a summary of FNA layers in the model."""
    print("\nFNA Layer Summary:")
    print(f"{'Layer':<50} {'Grid':<12} {'Params':<10} {'% Frozen'}")
    print("-" * 85)
    for name, module in model.named_modules():
        if isinstance(module, FNALayer):
            pc = module.param_count()
            print(
                f"  {name:<48} "
                f"{module.grid_size}x{module.grid_size:<6} "
                f"{pc['trainable']:<10,} "
                f"{pc['pct_of_frozen']}%"
            )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 50)
    print("injection.py — sanity check (no HF download)")
    print("=" * 50)

    # Build a minimal T5-like FFN block for testing without downloading T5
    class FakeDenseReluDense(nn.Module):
        def __init__(self):
            super().__init__()
            self.wi = nn.Linear(512, 2048, bias=False)
            self.wo = nn.Linear(2048, 512, bias=False)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            return self.wo(self.dropout(torch.relu(self.wi(x))))

    class FakeT5Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.DenseReluDense = FakeDenseReluDense()

        def forward(self, x):
            return self.DenseReluDense(x)

    class FakeT5(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.ModuleList([FakeT5Block() for _ in range(3)])

        def forward(self, x):
            for b in self.block:
                x = b(x)
            return x

    model = FakeT5()

    # Count params before injection
    before = sum(p.numel() for p in model.parameters())
    print(f"Params before injection: {before:,}")

    # Inject FNA adapters
    fna_layers = inject_fna_adapters(model, grid_size=128, verbose=True)

    # Verify forward pass still works
    x   = torch.randn(2, 16, 512)
    out = model(x)
    print(f"Forward pass: input {tuple(x.shape)} -> output {tuple(out.shape)}  OK")

    # Verify only M params are trainable
    trainable = get_fna_optimizer_params(model)
    print(f"Trainable parameter tensors: {len(trainable)}")
    print(f"Expected: {len(fna_layers)} (one M per FNALayer)")

    print_model_summary(model)
    print("Injection sanity check passed.")
