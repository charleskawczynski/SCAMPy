import numpy as np

class SecantMethod:
    def __init__(self):
        return

def SecantMethodFunc(f, x0, x1, xatol=1e-3, maxiters=10):
    y0 = f(x0)
    y1 = f(x1)
    for i in range(0, maxiters):
        Δx = x1 - x0
        Δy = y1 - y0
        x0, y0 = x1, y1
        x1 -= y1 * Δx / Δy
        y1 = f(x1)
        if abs(x0-x1) < xatol:
            return x1, True
    return x1, False

def find_zero(f, x0, x1, method=SecantMethod(), xatol=1e-3, maxiters=10):
    if isinstance(method, SecantMethod):
        return SecantMethodFunc(f, x0, x1, xatol, maxiters)
