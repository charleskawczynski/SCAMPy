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

def compute_entrainment_detrainment(grid, UpdVar, Case, tmp, q, params):
    quadrature_order = 3
    gm, en, ud, sd, al = q.idx.allcombinations()
    compute_cloud_base_top_cover(grid, q, tmp, UpdVar)
    k_1 = grid.first_interior(Zmin())
    dzi = grid.dzi
    n_updrafts = len(ud)
    input_st = type('', (), {})()
    input_st.wstar = params.wstar
    input_st.b_mean = 0
    input_st.dz = grid.dz
    for i in ud:
        input_st.zi = UpdVar[i].cloud_base
        for k in grid.over_elems_real(Center()):
            input_st.quadrature_order = quadrature_order
            input_st.z                = grid.z_half[k]
            input_st.ml               = tmp['l_mix'][k]
            input_st.b                = tmp['buoy', i][k]
            input_st.w                = q['w', i][k]
            input_st.af               = q['a', i][k]
            input_st.tke              = q['tke', en][k]
            input_st.qt_env           = q['q_tot', en][k]
            input_st.q_liq_env        = tmp['q_liq', en][k]
            input_st.θ_liq_env        = q['θ_liq', en][k]
            input_st.b_env            = tmp['buoy', en][k]
            input_st.w_env            = q['w', en].Mid(k)
            input_st.θ_liq_up         = q['θ_liq', i][k]
            input_st.qt_up            = q['q_tot', i][k]
            input_st.p0               = tmp['p_0'][k]
            input_st.alpha0           = tmp['α_0'][k]
            input_st.tke              = q['tke', en][k]
            input_st.tke_ed_coeff     = params.tke_ed_coeff
            input_st.L                = 20000.0 # need to define the scale of the GCM grid resolution
            input_st.n_up             = n_updrafts
            w_cut = q['w', i].Cut(k)
            w_env_cut = q['w', en].Cut(k)
            a_cut = q['a', i].Cut(k)
            a_env_cut = (1.0-q['a', i].Cut(k))
            aw_cut = a_cut * w_cut + a_env_cut * w_env_cut
            input_st.dwdz = grad(aw_cut, grid)
            ret = params.entr_detr_fp(input_st)
            tmp['ε_model', i][k] = ret.entr_sc * params.entrainment_factor
            tmp['δ_model', i][k] = ret.detr_sc * params.detrainment_factor
        tmp['ε_model', i][k_1] = 2.0 * dzi
        tmp['δ_model', i][k_1] = 0.0
    return

class EDMF_PrognosticTKE:
    def __init__(self):
        return

    def initialize_vars(self, grid, q, q_tendencies, tmp, tmp_O2, UpdVar, Case, TS, tri_diag, params):
        gm, en, ud, sd, al = q.idx.allcombinations()

        for k in grid.over_elems_real(Center()):
            ts = ActiveThermoState(q, tmp, gm, k)
            tmp['θ_ρ'][k] = virtual_pottemp(ts)
        params.zi = compute_inversion_height(grid, q, tmp, params.Ri_bulk_crit)

        zs = params.zi
        params.wstar = compute_convective_velocity(Case.Sur.bflux, zs)
        ws = params.wstar
        ws3 = ws**3.0
        us3 = Case.Sur.ustar**3.0
        k_1 = grid.first_interior(Zmin())
        apply_bcs(grid, q, tmp, UpdVar, Case, params)
        if ws > 0.0:
            for k in grid.over_elems_real(Center())[1:]:
                z = grid.z_half[k]
                temp = ws * 1.3 * np.cbrt(us3/ws3 + 0.6 * z/zs) * np.sqrt(np.fmax(1.0-z/zs,0.0))
                q['tke', gm][k] = temp
            compute_mixing_length(grid, q, tmp, Case.Sur.obukhov_length, params)
        # print('')
        # print('zi          = ',self.zi)
        # print('wstar       = ',params.wstar)
        # print('surface_tke = ', q['tke', gm][k_1])
        # raise NameError("Done")
        return

    def pre_compute_vars(self, grid, q, q_tendencies, tmp, tmp_O2, UpdVar, Case, TS, tri_diag, params):
        gm, en, ud, sd, al = q.idx.allcombinations()

        diagnose_environment(grid, q)
        saturation_adjustment_sd(grid, q, tmp)

        update_dt(grid, TS, q)

        compute_entrainment_detrainment(grid, UpdVar, Case, tmp, q, params)
        compute_cloud_phys(grid, q, tmp)
        compute_buoyancy(grid, q, tmp, params)

        filter_scalars(grid, q, tmp, params)

        for k in grid.over_elems_real(Center()):
            ts = ActiveThermoState(q, tmp, gm, k)
            tmp['θ_ρ'][k] = virtual_pottemp(ts)
        params.zi = compute_inversion_height(grid, q, tmp, params.Ri_bulk_crit)

        params.wstar = compute_convective_velocity(Case.Sur.bflux, params.zi)
        compute_cv_gm(grid, q, 'w'    , 'w'    , 'tke'           , 0.5, Half.Identity)
        compute_mf_gm(grid, q, TS, tmp)
        compute_eddy_diffusivities_tke(grid, q, tmp, Case, params)

        compute_tke_buoy(grid, q, tmp, tmp_O2, 'tke')
        compute_covariance_entr(grid, q, tmp, tmp_O2, 'w'    , 'w'    , 'tke'           , 0.5, Half.Identity)
        compute_covariance_shear(grid, q, tmp, tmp_O2, 'w'    , 'w'    , 'tke')
        compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, 'w'    , 'w'    , 'tke'           , 0.5, Half.Identity)
        compute_tke_pressure(grid, q, tmp, tmp_O2, 'tke', params)

        apply_bcs(grid, q, tmp, UpdVar, Case, params)

        compute_cv_env(grid, q, tmp, tmp_O2, 'w'    , 'w'    , 'tke'           , 0.5, Half.Identity)

        cleanup_covariance(grid, q)

    def update(self, grid, q_new, q, q_tendencies, tmp, tmp_O2, UpdVar, Case, TS, tri_diag, params):

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
