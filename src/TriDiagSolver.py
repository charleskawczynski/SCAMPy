import numpy as np
from numba import jit, f8
import copy
from Grid import Grid, Zmin, Zmax, Center, Node
from StateVec import StateVec, Cut, Dual

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

def construct_tridiag_diffusion_new(grid, Δzi, Δt, tmp, q, a, b, c):
    i_env = q.i_env
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

## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
@jit(f8[:] (f8[:],f8[:],f8[:],f8[:] ))
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]
    xc = bc
    xc[-1] = dc[-1]/bc[-1]
    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]
    return xc
