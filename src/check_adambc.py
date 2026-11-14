import torch
from adambc import AdamBC

torch.manual_seed(0)
w0 = torch.randn(5)
grads = [torch.randn(5) for _ in range(20)]


def run(opt_cls, noise_var=None, **kw):
    w = w0.clone().requires_grad_(True)
    opt = opt_cls([w], **kw)
    if noise_var is not None:
        opt.noise_var = noise_var
    for g in grads:
        w.grad = g.clone()
        opt.step()
    return w.detach()


a = run(torch.optim.Adam, lr=0.01, eps=0.0)
b = run(AdamBC, noise_var=0.0, lr=0.01, gamma=1e-30)
d0 = (a - b).abs().max().item()
print("max abs diff vs torch Adam (noise_var=0):", d0)
assert d0 < 1e-6

c = run(AdamBC, noise_var=0.5, lr=0.01, gamma=1e-8)
d1 = (a - c).abs().max().item()
print("max abs diff with noise_var=0.5:", d1)
assert d1 > 1e-3
print("ok")
