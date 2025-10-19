import torch 

@torch.no_grad()
def euler_solver(f, x0, t0: float, t1: float, n_steps: int):
    """
    Euler integrator for torch tensors.
    f(t, x) -> dx/dt  (supports broadcasting over batch)
    x0: (B, C, H, W) or any tensor
    Returns: ts (list of floats), xs (list of tensors length n_steps+1)
    """
    device = x0.device
    h = (t1 - t0) / n_steps
    x = x0.clone()
    ts = []
    xs = []
    t = t0
    for _ in range(n_steps + 1):
        ts.append(t)
        xs.append(x.clone())
        dx = f(x,t)
        x = x + h * dx
        t = t + h
    return ts, xs


@torch.no_grad()
def rk2_solver(f, x0, t0: float, t1: float, n_steps: int):
    """
    Euler integrator for torch tensors.
    f(t, x) -> dx/dt  (supports broadcasting over batch)
    x0: (B, C, H, W) or any tensor
    Returns: ts (list of floats), xs (list of tensors length n_steps+1)
    """
    h = (t1 - t0) / n_steps
    x = x0.clone()
    ts = []
    xs = []
    t = t0
    for _ in range(n_steps + 1):
        ts.append(t)
        xs.append(x.clone())
        k1 = f(x,t)
        x_pred = x + h * k1
        k2 = f(x_pred, t + h)
        x = x + (h/2) * (k1 + k2)
        t = t + h
    return xs,ts


def vf_learned(model):
    def f(x,t):
        t_tensor = torch.full((x.shape[0], 1, 1, 1), fill_value=t, device=x.device)
        return model(x, t_tensor)
    return f