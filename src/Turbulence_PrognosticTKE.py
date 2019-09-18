import numpy as np
from parameters import *
import sys
from Operators import advect, grad, Laplacian, grad_pos, grad_neg
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid
from Field import Field, Full, Half, Dirichlet, Neumann

from MoistThermodynamics import  *
from Surface import SurfaceBase
from Cases import  CasesBase
from TimeStepping import TimeStepping
from NetCDFIO import NetCDFIO_Stats
from EDMF_Updrafts import *
from funcs_EDMF import *
from funcs_turbulence import *

def pre_compute_vars(grid, q, q_tendencies, tmp, tmp_O2, UpdVar, Case, TS, tri_diag, params):
    gm, en, ud, sd, al = q.idx.allcombinations()

    diagnose_environment(grid, q)
    saturation_adjustment_sd(grid, q, tmp)

    update_dt(grid, TS, q)

    for k in grid.over_elems_real(Center()):
        ts = ActiveThermoState(q, tmp, gm, k)
        tmp['θ_ρ'][k] = virtual_pottemp(ts)
    params.zi = compute_inversion_height(grid, q, tmp, params.Ri_bulk_crit)
    params.wstar = compute_convective_velocity(Case.Sur.bflux, params.zi)

    compute_entrainment_detrainment(grid, UpdVar, Case, tmp, q, params)
    compute_cloud_phys(grid, q, tmp)
    compute_buoyancy(grid, q, tmp, params)

    filter_scalars(grid, q, tmp, params)

    compute_cv_gm(grid, q, 'w', 'w', 'tke', 0.5, Half.Identity)
    compute_mf_gm(grid, q, TS, tmp)
    compute_mixing_length(grid, q, tmp, Case.Sur.obukhov_length, params)
    compute_eddy_diffusivities_tke(grid, q, tmp, Case, params)

    compute_tke_buoy(grid, q, tmp, tmp_O2, 'tke')
    compute_cv_entr(grid, q, tmp, tmp_O2, 'w', 'w', 'tke', 0.5, Half.Identity)
    compute_cv_shear(grid, q, tmp, tmp_O2, 'w', 'w', 'tke')
    compute_cv_interdomain_src(grid, q, tmp, tmp_O2, 'w', 'w', 'tke', 0.5, Half.Identity)
    compute_tke_pressure(grid, q, tmp, tmp_O2, 'tke', params)
    compute_cv_env(grid, q, tmp, tmp_O2, 'w', 'w', 'tke', 0.5, Half.Identity)

    cleanup_covariance(grid, q)

def update(grid, q_new, q, q_tendencies, tmp, tmp_O2, UpdVar, Case, TS, tri_diag, params):

    assign_new_to_values(grid, q_new, q, tmp)

    compute_tendencies_en_O2(grid, q_tendencies, tmp_O2, 'tke')
    compute_tendencies_gm_scalars(grid, q_tendencies, q, tmp, Case, TS)
    compute_tendencies_ud(grid, q_tendencies, q, tmp, TS, params)

    compute_new_ud_a(grid, q_new, q, q_tendencies, tmp, UpdVar, TS, params)
    apply_bcs(grid, q_new, tmp, UpdVar, Case, params)

    compute_new_ud_w(grid, q_new, q, q_tendencies, tmp, TS, params)
    compute_new_ud_scalars(grid, q_new, q, q_tendencies, tmp, UpdVar, TS, params)

    apply_bcs(grid, q_new, tmp, UpdVar, Case, params)

    compute_new_en_O2(grid, q_new, q, q_tendencies, tmp, tmp_O2, TS, 'tke', tri_diag, params.tke_diss_coeff)
    compute_new_gm_scalars(grid, q_new, q, q_tendencies, TS, tmp, tri_diag)

    assign_values_to_new(grid, q, q_new, tmp)
    apply_bcs(grid, q, tmp, UpdVar, Case, params)

    return
