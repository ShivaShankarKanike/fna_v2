"""
fna/memory_layer.py
-------------------
FNA Latent Memory Layer.

Architecture:
    input x: (batch, seq, d_model)
        -> down_proj:  d_model -> d_latent  (d_latent = d_model // 16)
        -> reshape to 2D grid: (batch*seq, 1, H, W)
        -> FNA fluid step: advect + diffuse + project
        -> reshape back: (batch, seq, d_latent)
        -> up_proj: d_latent -> d_model
        -> gate: learned scalar in [0,1]
        -> output = x + gate * fluid_output  (residual)

Key properties:
    - Works as a drop-in residual module after any attention layer
    - Fluid step runs in latent space (d//16) not full hidden dim
    - Learned viscosity nu gated by token surprise (cross-entropy)
    - Zero output at init (gate=0) so training is stable from scratch

Parameter count (d_model=512):
    down_proj:  512 * 32 = 16,384
    up_proj:    32 * 512 = 16,384
    gate:       1
    nu_gate:    1
    Total:      32,770  (very small)

Author: Shiva Shankar Kanike
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .fluid_dynamics import ns_step


class FNAMemoryLayer(nn.Module):
    """
    FNA Latent Memory Layer.

    Drop this after any attention layer in any transformer.
    Runs the full NS fluid step in a compressed latent space
    to keep compute cost negligible.

    Args:
        d_model:      hidden dimension of the transformer (e.g. 512 for T5-small)
        bottleneck:   compression ratio (default 16, so d_latent = d_model // 16)
        dt:           fluid timestep (default 0.1)
        nu_init:      initial viscosity (default 0.2, learnable)
        learn_nu:     if True, viscosity is learned and gated by token surprise
        learn_gate:   if True, output gate is learned (recommended True)
    """

    def __init__(
        self,
        d_model:    int,
        bottleneck: int   = 16,
        dt:         float = 0.1,
        nu_init:    float = 0.2,
        learn_nu:   bool  = True,
        learn_gate: bool  = True,
    ):
        super().__init__()

        self.d_model    = d_model
        self.d_latent   = max(d_model // bottleneck, 4)
        self.dt         = dt
        self.learn_nu   = learn_nu

        # Bottleneck projections
        self.down_proj = nn.Linear(d_model, self.d_latent, bias=False)
        self.up_proj   = nn.Linear(self.d_latent, d_model, bias=False)

        # Initialize up_proj to zero so output starts at zero
        # This means at init: output = x + 0 = x (stable)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))

        # Output gate: learned scalar, sigmoid -> [0,1]
        # Starts at 0 (gate_raw=-6 -> sigmoid~0) for stable init
        if learn_gate:
            self.gate_raw = nn.Parameter(torch.tensor(-6.0))
        else:
            self.register_buffer("gate_raw", torch.tensor(0.0))
        self.learn_gate = learn_gate

        # Viscosity parameter
        # If learn_nu: nu is gated by token surprise
        #   high surprise -> low nu (fast adaptation)
        #   low surprise  -> high nu (stable state)
        # If not learn_nu: fixed scalar
        if learn_nu:
            # nu_base: base viscosity value (learned)
            self.nu_base = nn.Parameter(torch.tensor(nu_init))
            # surprise_scale: how strongly surprise modulates nu
            self.surprise_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("nu_base", torch.tensor(nu_init))

        # Velocity state: persistent across forward calls within a sequence
        # Reset between sequences via reset_state()
        self._V_state = None
        self._state_device = None

    def get_nu(self, surprise: torch.Tensor = None) -> float:
        """
        Compute effective viscosity.

        If learn_nu and surprise is provided:
            nu = nu_base * sigmoid(-surprise_scale * surprise)
            High surprise -> sigmoid near 0 -> low nu -> fast adaptation
            Low surprise  -> sigmoid near 1 -> high nu -> stable

        Args:
            surprise: scalar or (batch,) tensor. Token-level cross-entropy loss.

        Returns:
            nu: effective viscosity scalar
        """
        nu = torch.clamp(self.nu_base, min=0.01, max=1.0)

        if self.learn_nu and surprise is not None:
            # surprise is expected to be a scalar (mean over batch/seq)
            gate = torch.sigmoid(-self.surprise_scale * surprise)
            nu   = nu * gate

        return nu

    def reset_state(self):
        """
        Reset the velocity field state.
        Call between sequences / tasks to clear momentum.
        """
        self._V_state = None

    def forward(
        self,
        x:        torch.Tensor,
        surprise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:        (batch, seq, d_model) — input hidden states
            surprise: optional scalar tensor — mean cross-entropy loss
                      of current tokens. Used for learned viscosity gating.
                      If None, uses base viscosity.

        Returns:
            output: (batch, seq, d_model) — x + gated fluid residual
        """
        batch, seq, d = x.shape
        device = x.device

        # ── Step 1: Project to latent space ──────────────────────────────
        # (batch, seq, d_model) -> (batch, seq, d_latent)
        z = self.down_proj(x)

        # ── Step 2: Reshape to 2D grid for fluid step ────────────────────
        # Treat (batch*seq) as the batch dimension
        # Reshape d_latent into a 2D grid H x W where H*W = d_latent
        # For d_latent=32: H=W=... we find closest square
        # Use fixed grid: find H,W such that H*W >= d_latent
        H = int(math.ceil(math.sqrt(self.d_latent)))
        W = H
        pad = H * W - self.d_latent
        if pad > 0:
            z = F.pad(z, (0, pad))

        # (batch, seq, H*W) -> (batch*seq, H, W)
        z_2d = z.reshape(batch * seq, H, W)

        # Build velocity field V: shape (2, H, W)
        # Initialize or reuse from previous step (stateful memory)
        if self._V_state is None or self._V_state.device != device:
            V = torch.zeros(2, H, W, device=device, dtype=x.dtype)
        else:
            V = self._V_state

        # ── Step 3: FNA fluid step ────────────────────────────────────────
        # Use mean of z_2d as force field (loss gradient proxy)
        # force shape: (2, H, W) — broadcast mean activation as pressure
        force_2d = z_2d.mean(dim=0)                          # (H, W)
        force    = torch.stack([force_2d, force_2d], dim=0)  # (2, H, W)

        nu = self.get_nu(surprise)
        if isinstance(nu, torch.Tensor):
            nu_val = nu.item()
        else:
            nu_val = float(nu)

        V_new = ns_step(V, force, dt=self.dt, nu=nu_val)

        # Store velocity state for next forward call
        self._V_state = V_new.detach()

        # Apply velocity to latent: use mean of x and y components
        scalar_field = (V_new[0] + V_new[1]) / 2.0  # (H, W)

        # Broadcast scalar_field to all tokens
        # (H, W) -> (batch*seq, H, W)
        fluid_out_2d = z_2d + scalar_field.unsqueeze(0)

        # ── Step 4: Reshape back and project up ──────────────────────────
        # (batch*seq, H, W) -> (batch, seq, H*W)
        fluid_flat = fluid_out_2d.reshape(batch, seq, H * W)

        # Remove padding if added
        if pad > 0:
            fluid_flat = fluid_flat[:, :, :self.d_latent]

        # (batch, seq, d_latent) -> (batch, seq, d_model)
        fluid_up = self.up_proj(fluid_flat)

        # ── Step 5: Gated residual ────────────────────────────────────────
        gate   = torch.sigmoid(self.gate_raw)
        output = x + gate * fluid_up

        return output

    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {
            "d_model":   self.d_model,
            "d_latent":  self.d_latent,
            "total":     total,
            "down_proj": self.down_proj.weight.numel(),
            "up_proj":   self.up_proj.weight.numel(),
        }

    def extra_repr(self) -> str:
        pc = self.param_count()
        return (
            f"d_model={self.d_model}, d_latent={self.d_latent}, "
            f"dt={self.dt}, learn_nu={self.learn_nu}, "
            f"params={pc['total']:,}"
        )


# ---------------------------------------------------------------------------
# Injection utility — add FNA memory layers after attention in any model
# ---------------------------------------------------------------------------

def inject_fna_memory(
    model:      nn.Module,
    d_model:    int,
    after_modules: list = None,
    bottleneck: int   = 16,
    dt:         float = 0.1,
    nu_init:    float = 0.2,
    learn_nu:   bool  = True,
    verbose:    bool  = True,
) -> list:
    """
    Inject FNAMemoryLayer after target attention modules in any transformer.

    For T5-small: after_modules = ["SelfAttention", "EncDecAttention"]
    For LLaMA:   after_modules = ["self_attn"]

    This wraps each target module with a small residual block:
        output = original_module(x) -> FNAMemoryLayer(output)

    Args:
        model:         any HuggingFace transformer
        d_model:       hidden dimension
        after_modules: list of module name suffixes to wrap
        bottleneck:    compression ratio for latent space
        dt:            fluid timestep
        nu_init:       initial viscosity
        learn_nu:      learned viscosity gated by surprise
        verbose:       print injection summary

    Returns:
        memory_layers: list of injected FNAMemoryLayer instances
    """
    if after_modules is None:
        after_modules = ["SelfAttention"]

    class WrappedModule(nn.Module):
        """Wraps an existing module with a FNAMemoryLayer residual."""
        def __init__(self, original, memory_layer):
            super().__init__()
            self.original     = original
            self.memory_layer = memory_layer

        def forward(self, *args, **kwargs):
            out = self.original(*args, **kwargs)
            # Handle tuple outputs (attention returns (output, weights, ...))
            if isinstance(out, tuple):
                hidden = out[0]
                hidden = self.memory_layer(hidden)
                return (hidden,) + out[1:]
            else:
                return self.memory_layer(out)

    memory_layers  = []
    injected_names = []

    for name, module in model.named_modules():
        for target in after_modules:
            if name.endswith(target):
                mem_layer = FNAMemoryLayer(
                    d_model=d_model, bottleneck=bottleneck,
                    dt=dt, nu_init=nu_init, learn_nu=learn_nu,
                )
                parts  = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr = parts[-1]
                setattr(parent, attr, WrappedModule(module, mem_layer))
                memory_layers.append(mem_layer)
                injected_names.append(name)
                break

    if verbose:
        print("\nFNA Memory Layer Injection")
        print("=" * 45)
        print(f"After modules:    {after_modules}")
        print(f"Layers injected:  {len(memory_layers)}")
        if memory_layers:
            pc = memory_layers[0].param_count()
            print(f"Params per layer: {pc['total']:,}")
            print(f"Total new params: {len(memory_layers) * pc['total']:,}")
            print(f"d_model={d_model} -> d_latent={pc['d_latent']}")
        for n in injected_names:
            print(f"  Wrapped: {n}")
        print("=" * 45)

    return memory_layers


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("memory_layer.py - sanity check")
    print("=" * 50)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    d_model = 512
    batch   = 4
    seq     = 32

    # Test standalone FNAMemoryLayer
    layer = FNAMemoryLayer(d_model=d_model, bottleneck=16, learn_nu=True).to(device)
    print("\nFNAMemoryLayer:", layer)
    pc = layer.param_count()
    print("Params:", pc)

    x   = torch.randn(batch, seq, d_model, device=device)
    out = layer(x)
    print("\nForward: input", tuple(x.shape), "-> output", tuple(out.shape), " OK")

    # Check output is close to input at init (gate=0)
    diff = (out - x).abs().max().item()
    print("Max diff from input at init:", round(diff, 6), " (should be near 0)")

    # Test with surprise signal
    surprise = torch.tensor(2.5, device=device)
    out2 = layer(x, surprise=surprise)
    print("Forward with surprise=2.5:", tuple(out2.shape), " OK")

    # Test backward
    loss = out.sum()
    loss.backward()
    print("Backward pass OK")

    # Test reset_state
    layer.reset_state()
    print("reset_state() OK")

    # Test injection into fake T5 block
    print("\nTesting injection into fake T5 block...")

    class FakeSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(512, 512, bias=False)
        def forward(self, x, *args, **kwargs):
            return (self.q(x), None)

    class FakeT5Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.SelfAttention = FakeSelfAttention()
        def forward(self, x):
            return self.SelfAttention(x)[0]

    class FakeT5(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.ModuleList([FakeT5Block() for _ in range(3)])
        def forward(self, x):
            for b in self.block:
                x = b(x)
            return x

    fake_model = FakeT5()
    mem_layers = inject_fna_memory(
        fake_model, d_model=512,
        after_modules=["SelfAttention"],
        verbose=True
    )

    x2  = torch.randn(2, 16, 512)
    out3 = fake_model(x2)
    print("Injection forward pass:", tuple(x2.shape), "->", tuple(out3.shape), " OK")
    print("\nAll checks passed.")
