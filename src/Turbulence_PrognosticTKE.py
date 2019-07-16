import numpy as np
from parameters import *
import sys
from Operators import advect, grad, Laplacian, grad_pos, grad_neg
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann

from TriDiagSolver import solve_tridiag_wrapper, construct_tridiag_diffusion_O1, construct_tridiag_diffusion_O2
from Surface import SurfaceBase
from Cases import  CasesBase
from TimeStepping import TimeStepping
from NetCDFIO import NetCDFIO_Stats
from EDMF_Updrafts import *
from funcs_EDMF import *
from funcs_thermo import  *
from funcs_turbulence import *

def compute_entrainment_detrainment(grid, UpdVar, Case, tmp, q, entr_detr_fp, wstar, tke_ed_coeff, entrainment_factor, detrainment_factor):
    quadrature_order = 3
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    compute_cloud_base_top_cover(grid, q, tmp, UpdVar)
    n_updrafts = len(i_uds)

    input_st = type('', (), {})()
    input_st.wstar = wstar

    input_st.b_mean = 0
    input_st.dz = grid.dz
    input_st.zbl = compute_zbl_qt_grad(grid, q)
    for i in i_uds:
        input_st.zi = UpdVar[i].cloud_base
        for k in grid.over_elems_real(Center()):
            input_st.quadrature_order = quadrature_order
            input_st.z                = grid.z_half[k]
            input_st.ml               = tmp['l_mix'][k]
            input_st.b                = tmp['B', i][k]
            input_st.w                = q['w', i].Mid(k)
            input_st.af               = q['a', i][k]
            input_st.tke              = q['tke', i_env][k]
            input_st.qt_env           = q['q_tot', i_env][k]
            input_st.q_liq_env        = tmp['q_liq', i_env][k]
            input_st.θ_liq_env        = q['θ_liq', i_env][k]
            input_st.b_env            = tmp['B', i_env][k]
            input_st.w_env            = q['w', i_env].values[k]
            input_st.θ_liq_up         = q['θ_liq', i][k]
            input_st.qt_up            = q['q_tot', i][k]
            input_st.q_liq_up         = tmp['q_liq', i][k]
            input_st.env_Hvar         = q['cv_θ_liq', i_env][k]
            input_st.env_QTvar        = q['cv_q_tot', i_env][k]
            input_st.env_HQTcov       = q['cv_θ_liq_q_tot', i_env][k]
            input_st.p0               = tmp['p_0_half'][k]
            input_st.alpha0           = tmp['α_0_half'][k]
            input_st.tke              = q['tke', i_env][k]
            input_st.tke_ed_coeff     = tke_ed_coeff
            input_st.L                = 20000.0 # need to define the scale of the GCM grid resolution
            input_st.n_up             = n_updrafts

            w_cut = q['w', i].DualCut(k)
            w_env_cut = q['w', i_env].DualCut(k)
            a_cut = q['a', i].Cut(k)
            a_env_cut = (1.0-q['a', i].Cut(k))
            aw_cut = a_cut * w_cut + a_env_cut * w_env_cut

            input_st.dwdz = grad(aw_cut, grid)

            if input_st.zbl-UpdVar[i].cloud_base > 0.0:
                input_st.poisson = np.random.poisson(grid.dz/((input_st.zbl-UpdVar[i].cloud_base)/10.0))
            else:
                input_st.poisson = 0.0
            ret = entr_detr_fp(input_st)
            tmp['entr_sc', i][k] = ret.entr_sc * entrainment_factor
            tmp['detr_sc', i][k] = ret.detr_sc * detrainment_factor

    return

class EDMF_PrognosticTKE:
    def __init__(self, namelist, paramlist, grid):

        self.n_updrafts             = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']
        self.use_local_micro        = namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro']
        self.similarity_diffusivity = namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity']
        self.prandtl_number         = paramlist['turbulence']['prandtl_number']
        self.Ri_bulk_crit           = paramlist['turbulence']['Ri_bulk_crit']
        self.surface_area           = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']
        self.max_area_factor        = paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor']
        self.entrainment_factor     = paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor']
        self.detrainment_factor     = paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor']
        self.pressure_buoy_coeff    = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff']
        self.pressure_drag_coeff    = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_drag_coeff']
        self.pressure_plume_spacing = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_plume_spacing']
        self.tke_ed_coeff           = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff']
        self.tke_diss_coeff         = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff']
        self.max_supersaturation    = paramlist['turbulence']['updraft_microphysics']['max_supersaturation']
        self.updraft_fraction       = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']

        entr_src = namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']
        if str(entr_src) == 'inverse_z':        self.entr_detr_fp = entr_detr_inverse_z
        elif str(entr_src) == 'dry':            self.entr_detr_fp = entr_detr_dry
        elif str(entr_src) == 'inverse_w':      self.entr_detr_fp = entr_detr_inverse_w
        elif str(entr_src) == 'b_w2':           self.entr_detr_fp = entr_detr_b_w2
        elif str(entr_src) == 'entr_detr_tke':  self.entr_detr_fp = entr_detr_tke
        elif str(entr_src) == 'entr_detr_tke2': self.entr_detr_fp = entr_detr_tke2
        elif str(entr_src) == 'suselj':         self.entr_detr_fp = entr_detr_suselj
        elif str(entr_src) == 'none':           self.entr_detr_fp = entr_detr_none
        else: raise ValueError('Bad entr_detr_fp in Turbulence_PrognosticTKE.py')

        self.vel_pressure_coeff = self.pressure_drag_coeff/self.pressure_plume_spacing
        self.vel_buoy_coeff = 1.0-self.pressure_buoy_coeff
        self.minimum_area = 1e-3
        return

    def initialize_vars(self, grid, q, q_tendencies, tmp, tmp_O2, UpdVar, Case, TS, tri_diag):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        self.zi = compute_inversion(grid, q, Case.inversion_option, tmp, self.Ri_bulk_crit, tmp['temp_C'])
        zs = self.zi
        self.wstar = compute_convective_velocity(Case.Sur.bflux, zs)
        ws = self.wstar
        ws3 = ws**3.0
        us3 = Case.Sur.ustar**3.0
        k_1 = grid.first_interior(Zmin())
        cv_θ_liq_1 = q['cv_θ_liq', i_gm][k_1]
        cv_q_tot_1 = q['cv_q_tot', i_gm][k_1]
        cv_θ_liq_q_tot_1 = q['cv_θ_liq_q_tot', i_gm][k_1]
        reset_surface_covariance(grid, q, tmp, Case, ws)
        if ws > 0.0:
            for k in grid.over_elems(Center()):
                z = grid.z_half[k]
                temp = ws * 1.3 * np.cbrt(us3/ws3 + 0.6 * z/zs) * np.sqrt(np.fmax(1.0-z/zs,0.0))
                q['tke', i_gm][k] = temp
                q['cv_θ_liq', i_gm][k]       = cv_θ_liq_1 * temp
                q['cv_q_tot', i_gm][k]       = cv_q_tot_1 * temp
                q['cv_θ_liq_q_tot', i_gm][k] = cv_θ_liq_q_tot_1 * temp
            reset_surface_covariance(grid, q, tmp, Case, ws)
            compute_mixing_length(grid, q, tmp, Case.Sur.obukhov_length, self.zi, self.wstar)
        self.pre_compute_vars(grid, q, q_tendencies, tmp, tmp_O2, UpdVar, Case, TS, tri_diag)
        return

    def set_updraft_surface_bc(self, grid, q, tmp, UpdVar, Case):
        i_gm, i_env, i_uds, i_sd = tmp.domain_idx()
        k_1 = grid.first_interior(Zmin())
        zLL = grid.z_half[k_1]
        θ_liq_1 = q['θ_liq', i_gm][k_1]
        q_tot_1 = q['q_tot', i_gm][k_1]
        alpha0LL  = tmp['α_0_half'][k_1]
        S = Case.Sur
        cv_q_tot = surface_variance(S.rho_q_tot_flux*alpha0LL, S.rho_q_tot_flux*alpha0LL, S.ustar, zLL, S.obukhov_length)
        cv_θ_liq = surface_variance(S.rho_θ_liq_flux*alpha0LL, S.rho_θ_liq_flux*alpha0LL, S.ustar, zLL, S.obukhov_length)
        for i in i_uds:
            UpdVar[i].area_surface_bc = self.surface_area/self.n_updrafts
            UpdVar[i].w_surface_bc = 0.0
            UpdVar[i].θ_liq_surface_bc = (θ_liq_1 + UpdVar[i].surface_scalar_coeff * np.sqrt(cv_θ_liq))
            UpdVar[i].q_tot_surface_bc = (q_tot_1 + UpdVar[i].surface_scalar_coeff * np.sqrt(cv_q_tot))
        return

    def pre_compute_vars(self, grid, q, q_tendencies, tmp, tmp_O2, UpdVar, Case, TS, tri_diag):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        self.zi = compute_inversion(grid, q, Case.inversion_option, tmp, self.Ri_bulk_crit, tmp['temp_C'])
        self.wstar = compute_convective_velocity(Case.Sur.bflux, self.zi)
        diagnose_environment(grid, q)
        compute_cv_gm(grid, q, 'w'    , 'w'    , 'tke')
        compute_cv_gm(grid, q, 'θ_liq', 'θ_liq', 'cv_θ_liq')
        compute_cv_gm(grid, q, 'q_tot', 'q_tot', 'cv_q_tot')
        compute_cv_gm(grid, q, 'θ_liq', 'q_tot', 'cv_θ_liq_q_tot')
        update_GMV_MF(grid, q, TS, tmp)
        compute_eddy_diffusivities_tke(grid, q, tmp, Case, self.zi, self.wstar, self.prandtl_number, self.tke_ed_coeff, self.similarity_diffusivity)

        we = q['w', i_env]
        compute_tke_buoy(grid, q, tmp, tmp_O2, 'tke')
        compute_covariance_entr(grid, q, tmp, tmp_O2, 'w'    , 'w'    , 'tke')
        compute_covariance_shear(grid, q, tmp, tmp_O2, 'w'    , 'w'    , 'tke')
        compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, 'w'    , 'w'    , 'tke')
        compute_tke_pressure(grid, q, tmp, tmp_O2, self.pressure_buoy_coeff, self.pressure_drag_coeff, self.pressure_plume_spacing, 'tke')
        compute_covariance_entr(grid, q, tmp, tmp_O2, 'θ_liq', 'θ_liq', 'cv_θ_liq')
        compute_covariance_entr(grid, q, tmp, tmp_O2, 'q_tot', 'q_tot', 'cv_q_tot')
        compute_covariance_entr(grid, q, tmp, tmp_O2, 'θ_liq', 'q_tot', 'cv_θ_liq_q_tot')
        compute_covariance_shear(grid, q, tmp, tmp_O2, 'θ_liq', 'θ_liq', 'cv_θ_liq')
        compute_covariance_shear(grid, q, tmp, tmp_O2, 'q_tot', 'q_tot', 'cv_q_tot')
        compute_covariance_shear(grid, q, tmp, tmp_O2, 'θ_liq', 'q_tot', 'cv_θ_liq_q_tot')
        compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, 'θ_liq', 'θ_liq', 'cv_θ_liq')
        compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, 'q_tot', 'q_tot', 'cv_q_tot')
        compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, 'θ_liq', 'q_tot', 'cv_θ_liq_q_tot')
        compute_covariance_rain(grid, q, tmp, tmp_O2, TS, 'tke')
        compute_covariance_rain(grid, q, tmp, tmp_O2, TS, 'cv_θ_liq')
        compute_covariance_rain(grid, q, tmp, tmp_O2, TS, 'cv_q_tot')
        compute_covariance_rain(grid, q, tmp, tmp_O2, TS, 'cv_θ_liq_q_tot')

        reset_surface_covariance(grid, q, tmp, Case, self.wstar)

        compute_cv_env(grid, q, tmp, tmp_O2, 'w'    , 'w'    , 'tke')
        compute_cv_env(grid, q, tmp, tmp_O2, 'θ_liq', 'θ_liq', 'cv_θ_liq')
        compute_cv_env(grid, q, tmp, tmp_O2, 'q_tot', 'q_tot', 'cv_q_tot')
        compute_cv_env(grid, q, tmp, tmp_O2, 'θ_liq', 'q_tot', 'cv_θ_liq_q_tot')

        compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, 'tke')
        compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, 'cv_θ_liq')
        compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, 'cv_q_tot')
        compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, 'cv_θ_liq_q_tot')

        compute_tendencies_gm(grid, q_tendencies, q, Case, TS, tmp, tri_diag)

        cleanup_covariance(grid, q)
        self.set_updraft_surface_bc(grid, q, tmp, UpdVar, Case)

    def update(self, grid, q_new, q, q_tendencies, tmp, tmp_O2, UpdVar, Case, TS, tri_diag):

        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        self.pre_compute_vars(grid, q, q_tendencies, tmp, tmp_O2, UpdVar, Case, TS, tri_diag)

        assign_new_to_values(grid, q_new, q, tmp)

        self.compute_prognostic_updrafts(grid, q_new, q, q_tendencies, tmp, UpdVar, Case, TS)

        update_cv_env(grid, q, q_tendencies, tmp, tmp_O2, TS, 'tke'           , tri_diag, self.tke_diss_coeff)
        update_cv_env(grid, q, q_tendencies, tmp, tmp_O2, TS, 'cv_θ_liq'      , tri_diag, self.tke_diss_coeff)
        update_cv_env(grid, q, q_tendencies, tmp, tmp_O2, TS, 'cv_q_tot'      , tri_diag, self.tke_diss_coeff)
        update_cv_env(grid, q, q_tendencies, tmp, tmp_O2, TS, 'cv_θ_liq_q_tot', tri_diag, self.tke_diss_coeff)

        update_sol_gm(grid, q_new, q, q_tendencies, TS, tmp, tri_diag)

        for k in grid.over_elems_real(Center()):
            q['U', i_gm][k]     = q_new['U', i_gm][k]
            q['V', i_gm][k]     = q_new['V', i_gm][k]
            q['θ_liq', i_gm][k] = q_new['θ_liq', i_gm][k]
            q['q_tot', i_gm][k] = q_new['q_tot', i_gm][k]
            q['q_rai', i_gm][k] = q_new['q_rai', i_gm][k]

        apply_gm_bcs(grid, q)

        return

    def compute_prognostic_updrafts(self, grid, q_new, q, q_tendencies, tmp, UpdVar, Case, TS):
        time_elapsed = 0.0
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        u_max = np.max([q['w', i][k] for i in i_uds for k in grid.over_elems(Node())])
        self.dt_upd = np.minimum(TS.dt, 0.5 * grid.dz/np.fmax(u_max,1e-10))
        while time_elapsed < TS.dt:
            compute_entrainment_detrainment(grid, UpdVar, Case, tmp, q, self.entr_detr_fp, self.wstar, self.tke_ed_coeff, self.entrainment_factor, self.detrainment_factor)
            eos_update_SA_mean(grid, q, False, tmp, self.max_supersaturation)
            buoyancy(grid, q, tmp)
            compute_sources(grid, q, tmp, self.max_supersaturation)
            update_updraftvars(grid, q, tmp)

            self.solve_updraft_velocity_area(grid, q_new, q, q_tendencies, tmp, UpdVar, TS)
            self.solve_updraft_scalars(grid, q_new, q, q_tendencies, tmp, UpdVar, TS)
            assign_values_to_new(grid, q, q_new, tmp)
            for i in i_uds: q['θ_liq', i].apply_bc(grid, 0.0)
            for i in i_uds: q['q_tot', i].apply_bc(grid, 0.0)
            for i in i_uds: q['q_rai', i].apply_bc(grid, 0.0)
            q['w', i_env].apply_bc(grid, 0.0)
            q['θ_liq', i_env].apply_bc(grid, 0.0)
            q['q_tot', i_env].apply_bc(grid, 0.0)
            time_elapsed += self.dt_upd
            u_max = np.max([q['w', i][k] for i in i_uds for k in grid.over_elems(Node())])
            self.dt_upd = np.minimum(TS.dt-time_elapsed,  0.5 * grid.dz/np.fmax(u_max,1e-10))
            diagnose_environment(grid, q)
        eos_update_SA_mean(grid, q, True, tmp, self.max_supersaturation)
        buoyancy(grid, q, tmp)
        return

    def solve_updraft_velocity_area(self, grid, q_new, q, q_tendencies, tmp, UpdVar, TS):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        k_1 = grid.first_interior(Zmin())
        kb_1 = grid.boundary(Zmin())
        dzi = grid.dzi
        dti_ = 1.0/self.dt_upd
        dt_ = 1.0/dti_

        # Solve for area fraction
        for i in i_uds:
            au_lim = UpdVar[i].area_surface_bc * self.max_area_factor
            for k in grid.over_elems_real(Center()):

                a_k = q['a', i][k]
                α_0_kp = tmp['α_0_half'][k]
                w_k = q['w', i].Mid(k)

                w_cut = q['w', i].DualCut(k)
                a_cut = q['a', i].Cut(k)
                ρ_cut = tmp['ρ_0_half'].Cut(k)
                tendencies = 0.0

                ρaw_cut = ρ_cut*a_cut*w_cut
                adv = - α_0_kp * advect(ρaw_cut, w_cut, grid)
                tendencies+=adv

                ε_term = a_k * w_k * (+ tmp['entr_sc', i][k])
                tendencies+=ε_term
                δ_term = a_k * w_k * (- tmp['detr_sc', i][k])
                tendencies+=δ_term

                a_predict = a_k + dt_ * tendencies

                needs_limiter = a_predict>au_lim
                q_new['a', i][k] = np.fmin(np.fmax(a_predict, 0.0), au_lim)

                unsteady = (q_new['a', i][k]-a_k)*dti_
                # δ_limiter = unsteady - tendencies if needs_limiter else 0.0
                # tendencies+=δ_limiter
                # a_correct = a_k + dt_ * tendencies

                if needs_limiter:
                    δ_term_new = unsteady - adv - ε_term
                    if a_k > 0.0:
                        tmp['detr_sc', i][k] = δ_term_new/(-a_k  * w_k)
                    else:
                        tmp['detr_sc', i][k] = δ_term_new/(-au_lim  * w_k)

            tmp['entr_sc', i][k_1] = 2.0 * dzi
            tmp['detr_sc', i][k_1] = 0.0
            q_new['a', i][k_1] = UpdVar[i].area_surface_bc

        # Solve for updraft velocity
        for i in i_uds:
            q_new['a', i][kb_1] = UpdVar[i].w_surface_bc
            for k in grid.over_elems_real(Center()):
                a_new_k = q_new['a', i].Mid(k)
                if a_new_k >= self.minimum_area:

                    ρ_k = tmp['ρ_0'][k]
                    w_i = q['w', i][k]
                    w_env = q['w', i_env].values[k]
                    a_k = q['a', i].Mid(k)
                    entr_w = tmp['entr_sc', i].Mid(k)
                    detr_w = tmp['detr_sc', i].Mid(k)
                    B_k = tmp['B', i].Mid(k)

                    a_cut = q['a', i].DualCut(k)
                    ρ_cut = tmp['ρ_0'].Cut(k)
                    w_cut = q['w', i].Cut(k)

                    ρa_k = ρ_k * a_k
                    ρa_new_k = ρ_k * a_new_k
                    ρaw_k = ρa_k * w_i
                    ρaww_cut = ρ_cut*a_cut*w_cut*w_cut

                    adv = -advect(ρaww_cut, w_cut, grid)
                    exch = ρaw_k * (- detr_w * w_i + entr_w * w_env)
                    buoy = ρa_k * B_k
                    press_buoy = - ρa_k * B_k * self.pressure_buoy_coeff
                    press_drag = - ρa_k * (self.pressure_drag_coeff/self.pressure_plume_spacing * (w_i - w_env)**2.0/np.sqrt(np.fmax(a_k, self.minimum_area)))
                    nh_press = press_buoy + press_drag

                    q_new['w', i][k] = ρaw_k/ρa_new_k + dt_/ρa_new_k*(adv + exch + buoy + nh_press)

        # Filter results
        for i in i_uds:
            for k in grid.over_elems_real(Center()):
                if q_new['a', i].Mid(k) >= self.minimum_area:
                    if q_new['w', i][k] <= 0.0:
                        q_new['w', i][k:] = 0.0
                        q_new['a', i][k+1:] = 0.0
                        break
                else:
                    q_new['w', i][k:] = 0.0
                    q_new['a', i][k+1:] = 0.0
                    break

        return

    def solve_updraft_scalars(self, grid, q_new, q, q_tendencies, tmp, UpdVar, TS):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        dzi = grid.dzi
        dti_ = 1.0/self.dt_upd
        k_1 = grid.first_interior(Zmin())

        for i in i_uds:
            q_new['θ_liq', i][k_1] = UpdVar[i].θ_liq_surface_bc
            q_new['q_tot', i][k_1] = UpdVar[i].q_tot_surface_bc

            for k in grid.over_elems_real(Center())[1:]:
                dt_ = 1.0/dti_
                θ_liq_env = q['θ_liq', i_env][k]
                q_tot_env = q['q_tot', i_env][k]

                if q_new['a', i][k] >= self.minimum_area:
                    a_k = q['a', i][k]
                    a_cut = q['a', i].Cut(k)
                    a_k_new = q_new['a', i][k]
                    θ_liq_cut = q['θ_liq', i].Cut(k)
                    q_tot_cut = q['q_tot', i].Cut(k)
                    ρ_k = tmp['ρ_0_half'][k]
                    ρ_cut = tmp['ρ_0_half'].Cut(k)
                    w_cut = q['w', i].DualCut(k)
                    ε_sc = tmp['entr_sc', i][k]
                    δ_sc = tmp['detr_sc', i][k]
                    ρa_k = ρ_k*a_k

                    ρaw_cut = ρ_cut * a_cut * w_cut
                    ρawθ_liq_cut = ρaw_cut * θ_liq_cut
                    ρawq_tot_cut = ρaw_cut * q_tot_cut
                    ρa_new_k = ρ_k * a_k_new

                    tendencies_θ_liq = -advect(ρawθ_liq_cut, w_cut, grid) + ρaw_cut[1] * (ε_sc * θ_liq_env - δ_sc * θ_liq_cut[1])
                    tendencies_q_tot = -advect(ρawq_tot_cut, w_cut, grid) + ρaw_cut[1] * (ε_sc * q_tot_env - δ_sc * q_tot_cut[1])

                    q_new['θ_liq', i][k] = ρa_k/ρa_new_k * θ_liq_cut[1] + dt_*tendencies_θ_liq/ρa_new_k
                    q_new['q_tot', i][k] = ρa_k/ρa_new_k * q_tot_cut[1] + dt_*tendencies_q_tot/ρa_new_k
                else:
                    q_new['θ_liq', i][k] = q['θ_liq', i_gm][k]
                    q_new['q_tot', i][k] = q['q_tot', i_gm][k]

        if self.use_local_micro:
            for i in i_uds:
                for k in grid.over_elems_real(Center()):
                    θ_liq = q_new['θ_liq', i][k]
                    q_tot = q_new['q_tot', i][k]
                    p_0 = tmp['p_0_half'][k]
                    T, q_liq = eos(p_0, q_tot, θ_liq)
                    tmp['T', i][k] = T
                    tmp_qr = acnv_instant(q_liq, q_tot, self.max_supersaturation, T, p_0)
                    s = -tmp_qr
                    tmp['prec_src_q_tot', i][k] = s
                    r_src = rain_source_to_thetal(p_0, T, q_tot, q_liq, 0.0, tmp_qr)
                    tmp['prec_src_θ_liq', i][k] = r_src
                    q_new['q_tot', i][k] += s
                    q_new['q_rai', i][k] -= s
                    q_new['θ_liq', i][k] += r_src
                    tmp['q_liq', i][k] = q_liq + s
                q_new['q_rai', i][k_1] = 0.0

        return
