
import torch
from torch.optim import Optimizer
import torch.nn.functional as F

_LAP_K = torch.tensor([[0.,-0.25,0.],[-0.25,1.,-0.25],[0.,-0.25,0.]], dtype=torch.float32)

def laplacian(M):
    device = M.device
    k = _LAP_K.to(device).view(1,1,3,3)
    return F.conv2d(M.unsqueeze(0).unsqueeze(0), k, padding=1).squeeze()

class FNAOptimizer(Optimizer):
    """
    Hybrid: Adam moves M, viscosity correction smooths the update.
    This is stable and lets us isolate the viscosity contribution clearly.
    """
    def __init__(self, params, lr=3e-4, nu=0.05, betas=(0.9,0.999), eps=1e-8):
        defaults = dict(lr=lr, nu=nu, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr   = group["lr"]
            nu   = group["nu"]
            b1, b2 = group["betas"]
            eps  = group["eps"]

            for p in group["params"]:
                if p.grad is None or p.dim() != 2:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"]    = torch.zeros_like(p)
                    state["v"]    = torch.zeros_like(p)

                state["step"] += 1
                m, v = state["m"], state["v"]
                t    = state["step"]

                # Adam moments
                m.mul_(b1).add_(grad, alpha=1-b1)
                v.mul_(b2).addcmul_(grad, grad, value=1-b2)

                m_hat = m / (1 - b1**t)
                v_hat = v / (1 - b2**t)

                # Adam update
                adam_update = m_hat / (v_hat.sqrt() + eps)

                # Viscosity correction: smooth the update
                visc = nu * laplacian(adam_update)
                smoothed_update = adam_update + visc

                p.data.add_(smoothed_update, alpha=-lr)

        return loss

    def zero_velocity(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state:
                    self.state[p].clear()
