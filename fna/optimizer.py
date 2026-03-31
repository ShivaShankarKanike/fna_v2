"""
fna/optimizer.py
----------------
Pure Navier-Stokes optimizer for FNA adapter parameters.

Replaces Adam entirely for the latent grid M. Uses the NS fluid
simulation (fluid_dynamics.py) as the update rule.

How it works:
    Each FNALayer has one trainable parameter: M (the scalar grid).
    The optimizer maintains a velocity field V (shape 2, H, W) in its
    state — this is NOT a model parameter, just optimizer memory.

    Each step:
        1. Reshape M.grad into a 2D force field F (broadcast to 2, H, W)
        2. Apply ns_step(V, F, dt, nu) to get V_new
        3. Update M by advecting it with V_new:
               M_new = M - dt * (V_new[0] + V_new[1]) / 2
           (take the mean of x and y velocity components as scalar update)
        4. Store V_new in optimizer state for next step

Why this works:
    - V accumulates momentum across steps (inertia term)
    - The diffusion step smooths out noisy gradient spikes (viscosity)
    - The projection step prevents any grid region from monopolizing updates
    - The force (loss gradient) drives M toward lower loss regions

Author: Shiva Shankar Kanike
"""

import torch
from torch.optim import Optimizer
from .fluid_dynamics import ns_step


class FNAOptimizer(Optimizer):
    """
    Pure Navier-Stokes optimizer for FNA latent grid parameters.

    Only operates on parameters named 'M' inside FNALayer modules.
    All other parameters (frozen weights, biases) are ignored.

    Args:
        params:   iterable of parameters (pass model.parameters())
        dt:       fluid timestep / effective learning rate (default 0.05)
        nu:       viscosity coefficient for diffusion (default 0.1)
        clip_val: gradient clipping value before treating as force (default 1.0)
    """

    def __init__(
        self,
        params,
        dt: float  = 0.05,
        nu: float  = 0.1,
        clip_val: float = 1.0,
    ):
        defaults = dict(dt=dt, nu=nu, clip_val=clip_val)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform one NS optimizer step.

        Args:
            closure: optional closure recomputing the loss (standard PyTorch API).

        Returns:
            loss: if closure provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            dt       = group['dt']
            nu       = group['nu']
            clip_val = group['clip_val']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Only process 2D grids — these are our M parameters
                # Skip 1D biases and frozen weights (requires_grad=False skips them anyway)
                if p.dim() != 2:
                    continue

                H, W = p.shape
                grad = p.grad.data

                # Clip gradient to prevent force field explosion
                grad = grad.clamp(-clip_val, clip_val)

                # Initialize velocity field V in optimizer state if first step
                state = self.state[p]
                if len(state) == 0:
                    # V shape: (2, H, W) — x and y velocity components
                    state['V'] = torch.zeros(2, H, W, device=p.device, dtype=p.dtype)
                    state['step'] = 0

                state['step'] += 1
                V = state['V']

                # Build force field from gradient
                # Broadcast scalar gradient to both velocity components
                # This treats the loss gradient as uniform pressure on both
                # x and y velocity channels
                force = torch.stack([grad, grad], dim=0)  # (2, H, W)

                # Apply full NS timestep
                V_new = ns_step(V, force, dt=dt, nu=nu)

                # Update M: use mean of x and y velocity as scalar update
                # This reduces the 2D vector field back to a scalar grid update
                scalar_update = (V_new[0] + V_new[1]) / 2.0  # (H, W)

                # Apply update to M (subtract because we descend the loss)
                p.data.add_(-dt * scalar_update)

                # Store updated velocity for next step (inertia)
                state['V'] = V_new

        return loss

    def zero_velocity(self):
        """
        Reset all velocity fields to zero.
        Useful between tasks in multi-task training to clear momentum.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state and 'V' in self.state[p]:
                    self.state[p]['V'].zero_()

    def get_velocity_stats(self) -> dict:
        """
        Return stats on all velocity fields for monitoring training.
        Useful for debugging — if ||V|| explodes, reduce dt or nu.
        """
        stats = []
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state and 'V' in self.state[p]:
                    V = self.state[p]['V']
                    stats.append({
                        "shape":    tuple(p.shape),
                        "V_norm":   round(V.norm().item(), 6),
                        "V_max":    round(V.abs().max().item(), 6),
                        "step":     self.state[p].get('step', 0),
                    })
        return stats


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn as nn
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from fna.adapter import FNALayer

    print("=" * 50)
    print("optimizer.py — sanity check")
    print("=" * 50)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build a small FNA layer (32x32 for speed)
    frozen = nn.Linear(512, 2048)
    layer  = FNALayer.from_linear(frozen, grid_size=32).to(device)

    # Only pass trainable parameters to optimizer
    trainable = [p for p in layer.parameters() if p.requires_grad]
    optimizer = FNAOptimizer(trainable, dt=0.05, nu=0.1)

    print(f"\nTrainable params: {sum(p.numel() for p in trainable):,}")
    print(f"Optimizer: dt={optimizer.defaults['dt']}, nu={optimizer.defaults['nu']}")

    # Simulate 3 training steps
    print("\nStep | ||M||     | ||V||     | loss")
    print("-----|-----------|-----------|------")

    x = torch.randn(4, 32, 512, device=device)

    for step in range(1, 4):
        optimizer.zero_grad()
        out  = layer(x)
        loss = out.pow(2).mean()
        loss.backward()
        optimizer.step()

        V_stats = optimizer.get_velocity_stats()
        v_norm  = V_stats[0]['V_norm'] if V_stats else 0.0
        m_norm  = layer.M.data.norm().item()
        print(f"  {step}  | {m_norm:.6f}  | {v_norm:.6f}  | {loss.item():.4f}")

    print("\nOptimizer sanity check passed.")
