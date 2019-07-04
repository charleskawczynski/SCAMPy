import numpy as np
import copy
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
from StateVec import StateVec

def construct_tridiag_diffusion_O2(grid, Covar, dt, tri_diag, rho, ae, rho_ae_K_m, whalf, tmp):
    dzi = grid.dzi
    dzi2 = grid.dzi**2.0
    dti = TS.dti
    k_1 = grid.first_interior(Zmin())
    Covar_surf = Covar.values[k_1]
    for k in grid.over_elems_real(Center()):
        D_env = 0.0
        ρ_0_k = tmp['ρ_0_half'][k]
        ρ_0_kp = tmp['ρ_0_half'][k+1]
        ae_k = ae[k]
        ae_kp = ae[k+1]
        w_k = whalf[k]
        w_kp = whalf[k+1]
        rho_ae_K_k = rho_ae_K_m[k]
        rho_ae_K_km = rho_ae_K_m[k-1]
        for i in i_uds:
            wu_half = UpdVar.W.values[i].Mid(k)
            D_env += ρ_0_k * UpdVar.Area.values[i][k] * wu_half * self.entr_sc[i][k]

        l_mix = np.fmax(self.mixing_length[k], 1.0)
        tke_env = np.fmax(EnvVar.tke.values[k], 0.0)

        tri_diag.a[k] = (- rho_ae_K_km * dzi2 )
        tri_diag.b[k] = (ρ_0_k * ae_k * dti
                 - ρ_0_k * ae_k * whalf[k] * dzi
                 + rho_ae_K_k * dzi2 + rho_ae_K_km * dzi2
                 + D_env
                 + ρ_0_k * ae_k * self.tke_diss_coeff * np.sqrt(tke_env)/l_mix)
        tri_diag.c[k] = (ρ_0_kp * ae_kp * w_kp * dzi - rho_ae_K_k * dzi2)

        tri_diag.f[k] = (ρ_0_k * ae_old[k] * Covar.values[k] * dti
                 + Covar.press[k]
                 + Covar.buoy[k]
                 + Covar.shear[k]
                 + Covar.entr_gain[k]
                 + Covar.rain_src[k])

        tri_diag.a[k_1] = 0.0
        tri_diag.b[k_1] = 1.0
        tri_diag.c[k_1] = 0.0
        tri_diag.f[k_1] = Covar_surf

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
