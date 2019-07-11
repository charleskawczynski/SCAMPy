import numpy as np
import copy
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
from StateVec import StateVec
from funcs_tridiagsolver import solve_tridiag, solve_tridiag_stored, init_β_γ, solve_tridiag_old

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

def construct_tridiag_diffusion_O1(grid, dt, tri_diag, rho, ae):
    k_1 = grid.first_interior(Zmin())
    k_2 = grid.first_interior(Zmax())
    dzi = grid.dzi
    for k in grid.over_elems_real(Center()):
        ρaK_dual = tri_diag.ρaK.Dual(k)
        X = rho[k] * ae[k]/dt
        Z = ρaK_dual[0] * dzi * dzi
        Y = ρaK_dual[1] * dzi * dzi
        if k == k_1:
            Z = 0.0
        elif k == k_2:
            Y = 0.0
        tri_diag.a[k] = - Z/X
        tri_diag.b[k] = 1.0 + Y/X + Z/X
        tri_diag.c[k] = -Y/X
    return

def solve_tridiag_wrapper(grid, x, tri_diag):
    slice_real = grid.slice_real(Center())
    solve_tridiag_old(grid.nz,
                      tri_diag.f[slice_real],
                      tri_diag.a[slice_real],
                      tri_diag.b[slice_real],
                      tri_diag.c[slice_real])
    x[:] = tri_diag.f[:]

    # solve_tridiag(x[slice_real],
    #               tri_diag.f[slice_real],
    #               tri_diag.a[slice_real],
    #               tri_diag.b[slice_real],
    #               tri_diag.c[slice_real],
    #               grid.nz,
    #               tri_diag.xtemp[slice_real],
    #               tri_diag.γ[slice_real],
    #               tri_diag.β[slice_real])
    return

