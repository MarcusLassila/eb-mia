import torch

def ode_midpoint(f, x_0, t_0, step_size, n_steps):
    '''
    Integrate an ODE with the second-order midpoint method.
    Args:
        f (callable): Vector field evaluated as `f(x, t)`.
        x_0 (torch.Tensor): Initial state.
        t_0 (torch.Tensor): Initial time values.
        step_size (float): Integration step size.
        n_steps (int): Number of integration steps.
    Returns:
        torch.Tensor: Final state after integration.
    '''
    h = step_size
    x = x_0
    t = t_0
    for _ in range(n_steps):
        k_1 = f(x, t)
        k_2 = f(x + 0.5 * k_1 * h, t + 0.5 * h)
        x += k_2 * h
        t += h
    return x
