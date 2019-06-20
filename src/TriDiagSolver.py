import numpy as np
from numba import jit, f8
import copy
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
from StateVec import StateVec

def construct_tridiag_diffusion(nzg, gw, dzi, dt, rho_ae_K, rho, ae, a, b, c):
    nz = nzg - 2* gw
    for k in range(gw,nzg-gw):
        X = rho[k] * ae[k]/dt
        Y = rho_ae_K[k] * dzi * dzi
        Z = rho_ae_K[k-1] * dzi * dzi
        if k == gw:
            Z = 0.0
        elif k == nzg-gw-1:
            Y = 0.0
        a[k-gw] = - Z/X
        b[k-gw] = 1.0 + Y/X + Z/X
        c[k-gw] = -Y/X
    return

def construct_tridiag_diffusion_new_new(grid, dt, rho_ae_K, rho, ae, a, b, c):
    k1 = grid.first_interior(Zmin())
    k2 = grid.first_interior(Zmax())
    dzi = grid.dzi
    for k in grid.over_elems_real(Center()):
        ρaK_dual = rho_ae_K.Dual(k)
        X = rho[k] * ae[k]/dt
        Z = ρaK_dual[0] * dzi * dzi
        Y = ρaK_dual[1] * dzi * dzi
        if k == k1:
            Z = 0.0
        elif k == k2:
            Y = 0.0
        a[k] = - Z/X
        b[k] = 1.0 + Y/X + Z/X
        c[k] = -Y/X
    # print('a_working = ', a[:])
    # print('b_working = ', b[:])
    # print('c_working = ', c[:])
    return

def construct_tridiag_diffusion_new(grid, Δt, tmp, q, a, b, c, K_name):
    i_env = q.i_env
    k1 = grid.first_interior(Zmin())
    k2 = grid.first_interior(Zmax())
    Δzi2 = grid.dzi**2.0
    for k in grid.over_elems_real(Center()):
        ρ_0_dual = tmp['ρ_0', Dual(k)]
        a_env_dual = q['a', Dual(k), i_env]
        K_dual = tmp[K_name, Dual(k), i_env]
        ρaK_dual = ρ_0_dual * a_env_dual * K_dual
        denom = tmp['ρ_0', k] * q['a', k, i_env]
        Z = ρaK_dual[0] * Δzi2 * Δt
        Y = ρaK_dual[1] * Δzi2 * Δt
        # if k == k1:
        #     Z = 0.0
        # elif k == k2:
        #     Y = 0.0
        a[k] = - Z/denom
        b[k] = 1.0 + Y/denom + Z/denom
        c[k] = -Y/denom
    # print('a_new = ', a[:])
    # print('b_new = ', b[:])
    # print('c_new = ', c[:])
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

def tridiag_solve_wrapper(grid, x, f, a, b, c):
    xtemp = Half(grid)
    β = Half(grid)
    γ = Half(grid)
    slice_real = grid.slice_real(Center())
    tridiag_solve_new(x[slice_real],
                      f[slice_real],
                      a[slice_real],
                      b[slice_real],
                      c[slice_real],
                      grid.nz,
                      xtemp[slice_real],
                      γ[slice_real],
                      β[slice_real])
    return

def tridiag_solve_new(x, f, a, b, c, n, xtemp, γ, β):
  # Define coefficients:
  β[0] = b[0]
  γ[0] = c[0]/β[0]
  for i in range(1, n-1):
    β[i] = b[i]-a[i-1]*γ[i-1]
    γ[i] = c[i]/β[i]
  β[n-1] = b[n-1]-a[n-2]*γ[n-2]

  # Forward substitution:
  xtemp[0] = f[0]/β[0]
  for i in range(1, n):
    m = f[i] - a[i-1]*xtemp[i-1]
    xtemp[i] = m/β[i]

  # Backward substitution:
  x[n-1] = xtemp[n-1]
  for i in range(n-2,-1,-1):
    x[i] = xtemp[i]-γ[i]*x[i+1]
