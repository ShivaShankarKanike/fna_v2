"""
fna/fluid_dynamics.py
---------------------
Full Navier-Stokes fluid dynamics solver for the FNA adapter.

This module implements one complete NS timestep on a 2D latent grid.
All operations are differentiable PyTorch — no custom CUDA kernels needed.

The three steps in order:
    1. Advection   — fluid carries its own velocity forward in time
    2. Diffusion   — viscosity spreads momentum to neighboring cells
    3. Projection  — FFT Poisson solver enforces incompressibility

Coordinate convention:
    V shape: (2, H, W)
        V[0] = x-component (horizontal velocity)
        V[1] = y-component (vertical velocity)

    M shape: (H, W)
        Scalar field — the adapter weight delta values.
        Advected by V to shift weight mass toward high-gradient regions.

Author: Shiva Shankar Kanike
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1 — Advection
# ---------------------------------------------------------------------------

def advect(field: torch.Tensor, V: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """
    Semi-Lagrangian advection via grid_sample backtrace.

    For each cell (i, j), trace backward along the velocity field to find
    where the fluid came from at the previous timestep. Sample the field
    at that origin point using bilinear interpolation.

    This is the standard "stable fluids" approach (Stam 1999) — unconditionally
    stable regardless of dt or velocity magnitude.

    Args:
        field: tensor to advect. Shape (H, W) for scalar M, or (2, H, W) for V.
        V:     velocity field, shape (2, H, W). V[0]=x, V[1]=y.
        dt:    timestep size (controls how far back we trace).

    Returns:
        advected: same shape as field.
    """
    H, W = V.shape[1], V.shape[2]
    device = V.device

    # Build normalized grid coordinates in [-1, 1] for grid_sample
    # grid_sample expects (N, H, W, 2) where the last dim is (x, y)
    gy = torch.linspace(-1, 1, H, device=device)
    gx = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # (H, W) each

    # Convert velocity from pixel units to normalized [-1, 1] units
    # V[0] = x-velocity (W direction), V[1] = y-velocity (H direction)
    vx_norm = V[0] * (2.0 / W)
    vy_norm = V[1] * (2.0 / H)

    # Backtrace: sample from where this cell's fluid came from
    sample_x = grid_x - dt * vx_norm
    sample_y = grid_y - dt * vy_norm

    # grid_sample expects (x, y) ordering in the last dimension
    sample_grid = torch.stack([sample_x, sample_y], dim=-1)  # (H, W, 2)
    sample_grid = sample_grid.unsqueeze(0)                    # (1, H, W, 2)

    # Clamp to [-1, 1] for zero-flux boundary conditions
    sample_grid = sample_grid.clamp(-1, 1)

    if field.dim() == 2:
        # Scalar field (H, W) -> need (N, C, H, W) for grid_sample
        f = field.unsqueeze(0).unsqueeze(0)       # (1, 1, H, W)
        out = F.grid_sample(f, sample_grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
        return out.squeeze(0).squeeze(0)           # (H, W)
    else:
        # Vector field (2, H, W) -> (1, 2, H, W)
        f = field.unsqueeze(0)                     # (1, 2, H, W)
        out = F.grid_sample(f, sample_grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
        return out.squeeze(0)                      # (2, H, W)


# ---------------------------------------------------------------------------
# Step 2 — Diffusion (normalized Laplacian)
# ---------------------------------------------------------------------------

# Normalized 5-point stencil Laplacian kernel.
# Center = 1.0, neighbors = -0.25 each, sum = 0 for uniform fields.
# This is the NORMALIZED form: L(f)[i,j] = f[i,j] - avg(neighbors).
# Applied as: f_new = f + nu * L(f)  (viscosity adds smoothing)
_LAPLACIAN_KERNEL = torch.tensor([
    [ 0.00, -0.25,  0.00],
    [-0.25,  1.00, -0.25],
    [ 0.00, -0.25,  0.00],
], dtype=torch.float32)


def diffuse(V: torch.Tensor, nu: float = 0.1) -> torch.Tensor:
    """
    Apply one step of viscous diffusion to the velocity field.

    Computes: V_new = V + nu * Laplacian(V)

    The Laplacian measures how different each cell is from its neighbors.
    Adding it back smooths out sharp spikes — exactly like viscosity in
    a real fluid preventing turbulent, non-physical velocity spikes.

    Args:
        V:   velocity field, shape (2, H, W).
        nu:  viscosity coefficient. Larger = more smoothing.
             Typical range: 0.05 to 0.2.

    Returns:
        V_diffused: shape (2, H, W).
    """
    device = V.device
    kernel = _LAPLACIAN_KERNEL.to(device)

    # Apply depthwise conv: one kernel per channel (x and y velocity separately)
    # (2, H, W) -> (1, 2, H, W) for conv2d
    V_4d = V.unsqueeze(0)                              # (1, 2, H, W)
    k = kernel.view(1, 1, 3, 3).expand(2, 1, 3, 3)    # (2, 1, 3, 3) depthwise
    lap = F.conv2d(V_4d, k, padding=1, groups=2)       # (1, 2, H, W)
    lap = lap.squeeze(0)                               # (2, H, W)

    return V + nu * lap


# ---------------------------------------------------------------------------
# Step 3 — Pressure Projection (FFT Poisson solver)
# ---------------------------------------------------------------------------

def project(V: torch.Tensor) -> torch.Tensor:
    """
    Enforce incompressibility via FFT Poisson solver.

    In a real fluid, mass is conserved: fluid doesn't pile up or disappear.
    This is the divergence-free condition: div(V) = 0.

    In weight space, this acts as a regularizer preventing any single
    region of the weight grid from accumulating unbounded update mass.

    Algorithm (spectral method, periodic BC):
        1. Compute divergence of V in spatial domain
        2. FFT to frequency domain
        3. Divide by eigenvalues of the Laplacian (Poisson solve)
        4. IFFT back to get pressure field P
        5. Subtract gradient of P from V

    Periodic boundary conditions mean the grid tiles seamlessly —
    valid and fast for a compact 128x128 latent grid.

    Args:
        V: velocity field, shape (2, H, W).

    Returns:
        V_projected: divergence-free velocity field, shape (2, H, W).
    """
    H, W = V.shape[1], V.shape[2]
    device = V.device

    vx = V[0]  # (H, W)
    vy = V[1]  # (H, W)

    # --- Step 1: Compute divergence ---
    # div(V)[i,j] = (vx[i,j+1] - vx[i,j-1]) / 2 + (vy[i+1,j] - vy[i-1,j]) / 2
    # Using central differences with periodic wrap (roll handles the wraparound)
    div_vx = (torch.roll(vx, -1, dims=1) - torch.roll(vx, 1, dims=1)) / 2.0
    div_vy = (torch.roll(vy, -1, dims=0) - torch.roll(vy, 1, dims=0)) / 2.0
    divergence = div_vx + div_vy  # (H, W)

    # --- Step 2: FFT of divergence ---
    div_fft = torch.fft.rfft2(divergence)  # (H, W//2+1) complex

    # --- Step 3: Poisson solve in frequency domain ---
    # Eigenvalues of the discrete Laplacian under periodic BC:
    # lambda[ky, kx] = 2*cos(2pi*kx/W) + 2*cos(2pi*ky/H) - 4
    # We solve: lambda * P_hat = div_hat  =>  P_hat = div_hat / lambda
    ky = torch.arange(H, device=device, dtype=torch.float32)
    kx = torch.arange(W // 2 + 1, device=device, dtype=torch.float32)
    KY, KX = torch.meshgrid(ky, kx, indexing='ij')

    eigenvalues = (2 * torch.cos(2 * torch.pi * KX / W) +
                   2 * torch.cos(2 * torch.pi * KY / H) - 4.0)  # (H, W//2+1)

    # Avoid division by zero at DC component (k=0,0) — set pressure mean to 0
    eigenvalues[0, 0] = 1.0

    pressure_fft = div_fft / eigenvalues.to(div_fft.dtype)
    pressure_fft[0, 0] = 0.0  # zero mean pressure

    # --- Step 4: IFFT to get pressure field ---
    pressure = torch.fft.irfft2(pressure_fft, s=(H, W))  # (H, W)

    # --- Step 5: Subtract pressure gradient from velocity ---
    grad_px = (torch.roll(pressure, -1, dims=1) - torch.roll(pressure, 1, dims=1)) / 2.0
    grad_py = (torch.roll(pressure, -1, dims=0) - torch.roll(pressure, 1, dims=0)) / 2.0

    vx_new = vx - grad_px
    vy_new = vy - grad_py

    return torch.stack([vx_new, vy_new], dim=0)  # (2, H, W)


# ---------------------------------------------------------------------------
# Full NS step — combines all three
# ---------------------------------------------------------------------------

def ns_step(
    V: torch.Tensor,
    force: torch.Tensor,
    dt: float = 0.1,
    nu: float = 0.1,
) -> torch.Tensor:
    """
    Apply one complete Navier-Stokes timestep to the velocity field V.

    Pipeline:
        V  ->  advect(V, V)       [inertia: fluid carries itself]
           ->  diffuse(V, nu)     [viscosity: smooth out spikes]
           ->  V -= dt * force    [pressure: loss gradient pushes flow]
           ->  project(V)         [incompressibility: no mass buildup]

    Args:
        V:     velocity field, shape (2, H, W).
        force: external force field, shape (2, H, W).
               In FNA, this is the loss gradient reshaped and broadcast
               to the velocity field dimensions.
        dt:    timestep / learning rate for force application.
        nu:    viscosity coefficient for diffusion step.

    Returns:
        V_new: updated velocity field, shape (2, H, W).
    """
    # 1. Advection — fluid carries its own momentum
    V = advect(V, V, dt=dt)

    # 2. Diffusion — viscosity smooths the velocity field
    V = diffuse(V, nu=nu)

    # 3. Force application — loss gradient as external pressure
    #    Negative because we want to flow toward lower loss (downhill)
    V = V - dt * force

    # 4. Projection — enforce divergence-free constraint
    V = project(V)

    return V


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("fluid_dynamics.py — sanity check")
    print("=" * 50)

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    H, W = 128, 128

    # Random velocity field and force
    V = torch.randn(2, H, W, device=device) * 0.1
    force = torch.randn(2, H, W, device=device) * 0.01

    print(f"\nV shape:      {tuple(V.shape)}")
    print(f"Force shape:  {tuple(force.shape)}")

    # Test each step individually
    V_adv = advect(V, V, dt=0.1)
    print(f"\nAdvection output shape:  {tuple(V_adv.shape)}  OK")

    V_diff = diffuse(V, nu=0.1)
    print(f"Diffusion output shape:  {tuple(V_diff.shape)}  OK")

    V_proj = project(V)
    print(f"Projection output shape: {tuple(V_proj.shape)}  OK")

    # Check divergence is near zero after projection
    vx = V_proj[0]
    vy = V_proj[1]
    div = ((torch.roll(vx, -1, dims=1) - torch.roll(vx, 1, dims=1)) / 2.0 +
           (torch.roll(vy, -1, dims=0) - torch.roll(vy, 1, dims=0)) / 2.0)
    print(f"\nMax divergence after projection: {div.abs().max().item():.2e}  (should be near 0)")

    # Full NS step
    V_new = ns_step(V, force, dt=0.1, nu=0.1)
    print(f"Full NS step output shape: {tuple(V_new.shape)}  OK")

    norm_before = V.norm().item()
    norm_after  = V_new.norm().item()
    print(f"\n||V|| before: {norm_before:.4f}")
    print(f"||V|| after:  {norm_after:.4f}")

    print("\nAll checks passed.")
