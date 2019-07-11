import numpy as np
import copy
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
from StateVec import StateVec

def construct_tridiag_diffusion_O2(grid, q, tmp, TS, UpdVar, EnvVar, tri_diag, tke_diss_coeff):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    dzi = grid.dzi
    dzi2 = grid.dzi**2.0
    dti = TS.dti
    k_1 = grid.first_interior(Zmin())
    k_2 = grid.first_interior(Zmax())

    a_env = q['a', i_env]
    w_env = q['w', i_env]
    ρ_0_half = tmp['ρ_0_half']
    for k in grid.over_elems_real(Center()):
        ρ_0_cut = ρ_0_half.Cut(k)
        ae_cut = a_env.Cut(k)
        w_cut = w_env.DualCut(k)
        ρa_K_cut = a_env.DualCut(k) * tmp['K_h'].DualCut(k) * ρ_0_half.DualCut(k)

        D_env = sum([ρ_0_cut[1] *
                     UpdVar.Area.values[i][k] *
                     UpdVar.W.values[i].Mid(k) *
                     tmp['entr_sc', i][k] for i in i_uds])

        l_mix = np.fmax(tmp['l_mix'][k], 1.0)
        tke_env = np.fmax(EnvVar.tke.values[k], 0.0)

        tri_diag.a[k] = (- ρa_K_cut[0] * dzi2 )
        tri_diag.b[k] = (ρ_0_cut[1] * ae_cut[1] * dti
                 - ρ_0_cut[1] * ae_cut[1] * w_cut[1] * dzi
                 + ρa_K_cut[1] * dzi2 + ρa_K_cut[0] * dzi2
                 + D_env
                 + ρ_0_cut[1] * ae_cut[1] * tke_diss_coeff * np.sqrt(tke_env)/l_mix)
        tri_diag.c[k] = (ρ_0_cut[2] * ae_cut[2] * w_cut[2] * dzi - ρa_K_cut[1] * dzi2)

    tri_diag.a[k_1] = 0.0
    tri_diag.b[k_1] = 1.0
    tri_diag.c[k_1] = 0.0

    tri_diag.b[k_2] += tri_diag.c[k_2]
    tri_diag.c[k_2] = 0.0
    return

def construct_tridiag_diffusion_new_new(grid, dt, tri_diag, rho, ae):
    k1 = grid.first_interior(Zmin())
    k2 = grid.first_interior(Zmax())
    dzi = grid.dzi
    for k in grid.over_elems_real(Center()):
        ρaK_dual = tri_diag.ρaK.Dual(k)
        X = rho[k] * ae[k]/dt
        Z = ρaK_dual[0] * dzi * dzi
        Y = ρaK_dual[1] * dzi * dzi
        if k == k1:
            Z = 0.0
        elif k == k2:
            Y = 0.0
        tri_diag.a[k] = - Z/X
        tri_diag.b[k] = 1.0 + Y/X + Z/X
        tri_diag.c[k] = -Y/X
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

def tridiag_solve_wrapper_new(grid, x, tri_diag):
    # xtemp = Half(grid)
    # β = Half(grid)
    # γ = Half(grid)
    slice_real = grid.slice_real(Center())
    tridiag_solve(grid.nz,
                  tri_diag.f[slice_real],
                  tri_diag.a[slice_real],
                  tri_diag.b[slice_real],
                  tri_diag.c[slice_real])
    x[:] = tri_diag.f[:]
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
