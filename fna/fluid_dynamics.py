import torch
import torch.nn.functional as F


_LAPLACIAN_KERNEL = torch.tensor([
    [ 0.00, -0.25,  0.00],
    [-0.25,  1.00, -0.25],
    [ 0.00, -0.25,  0.00],
], dtype=torch.float32)


def advect(field, V, dt=0.1):
    H, W = V.shape[1], V.shape[2]
    device = V.device
    gy = torch.linspace(-1, 1, H, device=device)
    gx = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
    vx_norm = V[0] * (2.0 / W)
    vy_norm = V[1] * (2.0 / H)
    sample_x = grid_x - dt * vx_norm
    sample_y = grid_y - dt * vy_norm
    sample_grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
    sample_grid = sample_grid.clamp(-1, 1)
    if field.dim() == 2:
        f = field.unsqueeze(0).unsqueeze(0)
        out = F.grid_sample(f, sample_grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
        return out.squeeze(0).squeeze(0)
    else:
        f = field.unsqueeze(0)
        out = F.grid_sample(f, sample_grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
        return out.squeeze(0)


def diffuse(V, nu=0.1):
    device = V.device
    kernel = _LAPLACIAN_KERNEL.to(device)
    V_4d = V.unsqueeze(0)
    k = kernel.view(1, 1, 3, 3).expand(2, 1, 3, 3)
    lap = F.conv2d(V_4d, k, padding=1, groups=2).squeeze(0)
    return V + nu * lap


def project(V):
    H, W = V.shape[1], V.shape[2]
    device = V.device
    vx = V[0]
    vy = V[1]

    # Forward difference divergence
    div = (vx - torch.roll(vx, 1, dims=1)) + (vy - torch.roll(vy, 1, dims=0))

    # FFT Poisson solve
    div_fft = torch.fft.fft2(div)

    ky = torch.arange(H, device=device, dtype=torch.float32)
    kx = torch.arange(W, device=device, dtype=torch.float32)
    KY, KX = torch.meshgrid(ky, kx, indexing='ij')

    # Eigenvalues of forward-difference Laplacian
    lam_x = 2 - 2*torch.cos(2*torch.pi*KX/W)
    lam_y = 2 - 2*torch.cos(2*torch.pi*KY/H)
    eigenvalues = lam_x + lam_y
    eigenvalues[0, 0] = 1.0

    p_fft = div_fft / eigenvalues.to(div_fft.dtype)
    p_fft[0, 0] = 0.0
    p = torch.fft.ifft2(p_fft).real

    # Adjoint gradient (negative forward difference)
    grad_px = p - torch.roll(p, -1, dims=1)
    grad_py = p - torch.roll(p, -1, dims=0)

    vx_new = vx - grad_px
    vy_new = vy - grad_py

    return torch.stack([vx_new, vy_new], dim=0)


def ns_step(V, force, dt=0.1, nu=0.1):
    V = advect(V, V, dt=dt)
    V = diffuse(V, nu=nu)
    V = V - dt * force
    V = project(V)
    return V


if __name__ == "__main__":
    print("=" * 50)
    print("fluid_dynamics.py - sanity check")
    print("=" * 50)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    H, W = 128, 128
    V = torch.randn(2, H, W, device=device) * 0.1
    force = torch.randn(2, H, W, device=device) * 0.01
    print("V shape:     ", tuple(V.shape))
    print("Force shape: ", tuple(force.shape))
    V_adv = advect(V, V, dt=0.1)
    print("Advection output shape: ", tuple(V_adv.shape), " OK")
    V_diff = diffuse(V, nu=0.1)
    print("Diffusion output shape: ", tuple(V_diff.shape), " OK")
    V_proj = project(V)
    print("Projection output shape:", tuple(V_proj.shape), " OK")
    vx = V_proj[0]
    vy = V_proj[1]
    div = (vx - torch.roll(vx,1,dims=1)) + (vy - torch.roll(vy,1,dims=0))
    print("Max divergence after projection:", div.abs().max().item(), " (should be < 1e-5)")
    V_new = ns_step(V, force, dt=0.1, nu=0.1)
    print("Full NS step output shape:", tuple(V_new.shape), " OK")
    print("||V|| before:", V.norm().item())
    print("||V|| after: ", V_new.norm().item())
    print("All checks passed.")
