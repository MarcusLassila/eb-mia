import torch

def ode_midpoint(f, x_0, t_0, step_size, n_steps):
    ''' 2nd-order Runge-Kutta method (midpoint scheme) '''
    h = step_size
    x = x_0
    t = t_0
    for _ in range(n_steps):
        k_1 = f(x, t)
        k_2 = f(x + 0.5 * k_1 * h, t + 0.5 * h)
        x += k_2 * h
        t += h
    return x
