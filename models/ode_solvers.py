import torch

@torch.no_grad()
def euler_solver(f, x0, t0: float, t1: float, n_steps: int):
    """Simple Euler integrator."""
    assert n_steps >= 1, "n_steps must be >= 1"
    h = (t1 - t0) / n_steps
    x = x0.clone()
    xs, ts = [], []
    for k in range(n_steps + 1):
        t = t0 + k * h
        ts.append(t)
        xs.append(x.clone())
        if k < n_steps:
            dx = f(x, t)
            x = x + h * dx
    return xs, ts


@torch.no_grad()
def rk2_solver(f, x0, t0: float, t1: float, n_steps: int):
    """Heun's / RK2 integrator."""
    assert n_steps >= 1, "n_steps must be >= 1"
    h = (t1 - t0) / n_steps
    x = x0.clone()
    xs, ts = [], []
    for k in range(n_steps + 1):
        t = t0 + k * h
        ts.append(t)
        xs.append(x.clone())
        if k < n_steps:
            k1 = f(x, t)
            x_pred = x + h * k1
            k2 = f(x_pred, t + h)
            x = x + 0.5 * h * (k1 + k2)
    return xs, ts


def vf_learned(model):
    """Wrapper to handle scalar t values."""
    model.eval()
    def f(x, t: float):
        t_tensor = torch.full(
            (x.shape[0], 1, 1, 1),
            fill_value=t,
            device=x.device,
            dtype=x.dtype,
        )
        return model(x, t_tensor)
    return f


def create_samples(
    n_images: int,
    image_shape,         # (C, H, W)
    ode_solver,
    model,
    n_steps: int,
    return_all: bool = False,
    device=None,
    seed: int | None = None,
    clamp_mode: str | None = "clamp",   # "clamp", "tanh", or None
    clamp_range: tuple[float, float] = (-1.0, 1.0),
):
    """
    Samples n_images using the ODE solver.
    Returns:
      - if return_all=False: Tensor (B, C, H, W) at final time
      - if return_all=True:  list[Tensor] trajectory over time
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    f = vf_learned(model)

    g = None
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(seed)

    x0 = torch.randn((n_images, *image_shape), device=device, generator=g)
    xs,_ = ode_solver(f, x0, 0.0, 1.0, n_steps)

    # Apply clamping/squashing to each step
    if clamp_mode == "clamp":
        xs = [x.clamp_(*clamp_range) for x in xs]
    elif clamp_mode == "tanh":
        xs = [torch.tanh(x) for x in xs]
    elif clamp_mode is not None:
        raise ValueError(f"Unknown clamp_mode: {clamp_mode}")

    return xs if return_all else xs[-1]
