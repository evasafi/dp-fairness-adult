import torch
from torch.optim import Optimizer


class AdamBC(Optimizer):
    """Adam with the DP-noise bias removed from the second-moment estimate.

    Update rule from Tang, Shpilevskiy and Lecuyer (AAAI 2024), written
    from the paper and checked against the authors' adam_corr.py.
    Pass to Opacus make_private; Opacus puts the noisy averaged gradient
    in p.grad. After make_private, set noise_var to
    (noise_multiplier * max_grad_norm / expected_batch_size) ** 2.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), gamma=1e-8):
        defaults = dict(lr=lr, betas=betas, gamma=gamma)
        super().__init__(params, defaults)
        self.noise_var = 0.0; self.n_above = 0; self.n_total = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                g = p.grad
                state["m"].mul_(beta1).add_(g, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m_hat = state["m"] / (1 - beta1 ** t)
                v_hat = state["v"] / (1 - beta2 ** t)
                diff = v_hat - self.noise_var
                self.n_above += int((diff > group["gamma"]).sum())
                self.n_total += diff.numel()
                v_corr = torch.clamp(diff, min=group["gamma"])
                p.addcdiv_(m_hat, v_corr.sqrt(), value=-group["lr"])
        return loss
