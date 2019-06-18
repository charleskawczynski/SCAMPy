import numpy as np
import copy
from Grid import Grid, Zmin, Zmax, Center, Node

def construct_tridiag_diffusion(nzg, gw, dzi, dt, rho_ae_K_m, rho, ae, a, b, c):
    nz = nzg - 2* gw
    for k in range(gw,nzg-gw):
        X = rho[k] * ae[k]/dt
        Y = rho_ae_K_m[k] * dzi * dzi
        Z = rho_ae_K_m[k-1] * dzi * dzi
        if k == gw:
            Z = 0.0
        elif k == nzg-gw-1:
            Y = 0.0
        a[k-gw] = - Z/X
        b[k-gw] = 1.0 + Y/X + Z/X
        c[k-gw] = -Y/X
    return

def construct_tridiag_diffusion_new(grid, Δzi, Δt, tmp, q, i_env, a, b, c):
    for k in grid.over_elems_real(Center()):
        ρ_0_dual = tmp['ρ_0', Dual(k)]
        a_env_dual = q['a', Dual(k), i_env]
        K_m_dual = tmp['K_m', Dual(k)]
        coeff = ρ_0_dual * a_env_dual * K_m_dual
        X = tmp['ρ_0', k] * q['a', k, i_env]/Δt
        Y = coeff[1] * Δzi * Δzi
        Z = coeff[0] * Δzi * Δzi
        a[k] = - Z/X
        b[k] = 1.0 + Y/X + Z/X
        c[k] = -Y/X
    return

def tridiag_solve(nz, x, a, b, c):
    scratch = copy.deepcopy(x)
    scratch[0] = c[0]/b[0]
    x[0] = x[0]/b[0]

    for i in range(1,nz):
        m = 1.0/(b[i] - a[i] * scratch[i-1])
        scratch[i] = c[i] * m
        x[i] = (x[i] - a[i] * x[i-1])*m


    for i in range(nz-2,-1,-1):
        x[i] = x[i] - scratch[i] * x[i+1]

    return
