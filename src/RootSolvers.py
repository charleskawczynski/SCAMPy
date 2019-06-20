import numpy as np
import copy
from Grid import Grid, Zmin, Zmax, Center, Node

class SecantMethod:
    def __init__(self):
        return

def roots_equation(f):

def find_zero(f, x0, x1, method=SecantMethod(), xatol=1e-3, maxiters=10)
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

def eos(t_to_prog, prog_to_t, p0, qt, prog):
    qv = qt
    ql = 0.0
    pv_1 = pv_c(p0, qt, qt)
    pd_1 = p0 - pv_1
    T_1 = prog_to_t(prog, pd_1, pv_1, qt)
    pv_star_1 = pv_star(T_1)
    qv_star_1 = qv_star_c(p0,qt,pv_star_1)
    ql_2=0.0
    # If not saturated
    if(qt <= qv_star_1):
        T = T_1
        ql = 0.0
    else:
        ql_1 = qt - qv_star_1
        prog_1 = t_to_prog(p0, T_1, qt, ql_1, 0.0)
        f_1 = prog - prog_1
        T_2 = T_1 + ql_1 * latent_heat(T_1) /((1.0 - qt)*cpd + qv_star_1 * cpv)

        delta_T  = np.fabs(T_2 - T_1)
        T_2 = find_zero(roots_equation, T_1, T_2, SecantMethod())

        while delta_T > 1.0e-3 or ql_2 < 0.0:
            pv_star_2 = pv_star(T_2)
            qv_star_2 = qv_star_c(p0,qt,pv_star_2)
            pv_2 = pv_c(p0, qt, qv_star_2)
            pd_2 = p0 - pv_2
            ql_2 = qt - qv_star_2
            prog_2 =  t_to_prog(p0,T_2,qt, ql_2, 0.0   )
            f_2 = prog - prog_2
            T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1)
            T_1 = T_2
            T_2 = T_n
            f_1 = f_2
            delta_T  = np.fabs(T_2 - T_1)

        T  = T_2
        ql = ql_2

    return T, ql

