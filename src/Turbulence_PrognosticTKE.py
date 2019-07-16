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
from funcs_utility import *

def compute_grid_means(grid, q, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]
    for k in grid.over_elems_real(Center()):
        tmp['q_liq', i_gm][k] = ae[k] * tmp['q_liq', i_env][k] + sum([ q['a_tmp', i][k] * tmp['q_liq', i][k] for i in i_uds])
        q['q_rai', i_gm][k]   = ae[k] * q['q_rai', i_env][k]   + sum([ q['a_tmp', i][k] * q['q_rai_tmp', i][k] for i in i_uds])
        tmp['T', i_gm][k]     = ae[k] * tmp['T', i_env][k]     + sum([ q['a_tmp', i][k] * tmp['T', i][k] for i in i_uds])
        tmp['B', i_gm][k]     = ae[k] * tmp['B', i_env][k]     + sum([ q['a_tmp', i][k] * tmp['B', i][k] for i in i_uds])
    return

def compute_cv_gm(grid, q, ϕ, ψ, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    is_tke = cv=='tke'
    tke_factor = 0.5 if is_tke else 1.0
    ae = q['a', i_env]
    for k in grid.over_elems(Center()):
        if is_tke:
            Δϕ = q[ϕ, i_env].Mid(k) - q[ϕ, i_gm].Mid(k)
            Δψ = q[ψ, i_env].Mid(k) - q[ψ, i_gm].Mid(k)
        else:
            Δϕ = q[ϕ, i_env][k]-q[ϕ, i_gm][k]
            Δψ = q[ψ, i_env][k]-q[ψ, i_gm][k]

        q[cv, i_gm][k] = tke_factor * ae[k] * Δϕ * Δψ + ae[k] * q[cv, i_env][k]
        for i in i_uds:
            if is_tke:
                Δϕ = q[ϕ, i].Mid(k) - q[ϕ, i_gm].Mid(k)
                Δψ = q[ψ, i].Mid(k) - q[ψ, i_gm].Mid(k)
            else:
                Δϕ = q[ϕ, i][k]-q[ϕ, i_gm][k]
                Δψ = q[ψ, i][k]-q[ψ, i_gm][k]
            q[cv, i_gm][k] += tke_factor * q['a_tmp', i][k] * Δϕ * Δψ
    return

def compute_covariance_entr(grid, q, tmp, tmp_O2, ϕ, ψ, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    is_tke = cv=='tke'
    tke_factor = 0.5 if is_tke else 1.0
    for k in grid.over_elems_real(Center()):
        tmp_O2[cv]['entr_gain'][k] = 0.0
        for i in i_uds:
            if is_tke:
                ϕ_u = q[ϕ, i].Mid(k)
                ψ_u = q[ψ, i].Mid(k)
                ϕ_e = q[ϕ, i_env].Mid(k)
                ψ_e = q[ψ, i_env].Mid(k)
            else:
                ϕ_u = q[ϕ, i][k]
                ψ_u = q[ψ, i][k]
                ϕ_e = q[ϕ, i_env][k]
                ψ_e = q[ψ, i_env][k]
            w_u = q['w_tmp', i].Mid(k)
            tmp_O2[cv]['entr_gain'][k] += tke_factor*q['a_tmp', i][k] * np.fabs(w_u) * tmp['detr_sc', i][k] * \
                                         (ϕ_u - ϕ_e) * (ψ_u - ψ_e)
        tmp_O2[cv]['entr_gain'][k] *= tmp['ρ_0_half'][k]
    return

def compute_covariance_shear(grid, q, tmp, tmp_O2, ϕ, ψ, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]
    is_tke = cv=='tke'
    tke_factor = 0.5 if is_tke else 1.0
    grad_u = 0.0
    grad_v = 0.0
    for k in grid.over_elems_real(Center()):
        if is_tke:
            grad_u = grad_neg(q['U', i_gm].Cut(k), grid)
            grad_v = grad_neg(q['V', i_gm].Cut(k), grid)
            grad_ϕ = grad_neg(q[ϕ, i_env].Cut(k), grid)
            grad_ψ = grad_neg(q[ψ, i_env].Cut(k), grid)
        else:
            grad_ϕ = grad(q[ϕ, i_env].Cut(k), grid)
            grad_ψ = grad(q[ψ, i_env].Cut(k), grid)
        ρaK = tmp['ρ_0_half'][k] * ae[k] * tmp['K_h'][k]
        tmp_O2[cv]['shear'][k] = tke_factor*2.0*ρaK * (grad_ϕ*grad_ψ + grad_u**2.0 + grad_v**2.0)
    return

def compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, ϕ, ψ, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    is_tke = cv=='tke'
    tke_factor = 0.5 if is_tke else 1.0
    for k in grid.over_elems(Center()):
        tmp_O2[cv]['interdomain'][k] = 0.0
        for i in i_uds:
            if is_tke:
                Δϕ = q[ϕ, i].Mid(k) - q[ϕ, i_env].Mid(k)
                Δψ = q[ψ, i].Mid(k) - q[ψ, i_env].Mid(k)
            else:
                Δϕ = q[ϕ, i][k]-q[ϕ, i_env][k]
                Δψ = q[ψ, i][k]-q[ψ, i_env][k]
            tmp_O2[cv]['interdomain'][k] += tke_factor*q['a_tmp', i][k] * (1.0-q['a_tmp', i][k]) * Δϕ * Δψ
    return

def compute_covariance_detr(grid, q, tmp, tmp_O2, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]

    for k in grid.over_elems_real(Center()):
        tmp_O2[cv]['detr_loss'][k] = 0.0
        for i in i_uds:
            w_u = q['w_tmp', i].Mid(k)
            tmp_O2[cv]['detr_loss'][k] += q['a_tmp', i][k] * np.fabs(w_u) * tmp['entr_sc', i][k]
        tmp_O2[cv]['detr_loss'][k] *= tmp['ρ_0_half'][k] * q[cv, i_env][k]
    return

def compute_tke_pressure(grid, q, tmp, tmp_O2, pressure_buoy_coeff, pressure_drag_coeff, pressure_plume_spacing, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems_real(Center()):
        tmp_O2[cv]['press'][k] = 0.0
        for i in i_uds:
            wu_half = q['w_tmp', i].Mid(k)
            we_half = q['w', i_env].Mid(k)
            a_i = q['a_tmp', i][k]
            ρ_0_k = tmp['ρ_0_half'][k]
            press_buoy = (-1.0 * ρ_0_k * a_i * tmp['B', i][k] * pressure_buoy_coeff)
            press_drag_coeff = -1.0 * ρ_0_k * np.sqrt(a_i) * pressure_drag_coeff/pressure_plume_spacing
            press_drag = press_drag_coeff * (wu_half - we_half)*np.fabs(wu_half - we_half)
            tmp_O2[cv]['press'][k] += (we_half - wu_half) * (press_buoy + press_drag)
    return

def compute_cv_env(grid, q, tmp, tmp_O2, ϕ, ψ, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    is_tke = cv=='tke'
    tke_factor = 0.5 if is_tke else 1.0
    ae = q['a', i_env]

    for k in grid.over_elems(Center()):
        if ae[k] > 0.0:
            if is_tke:
                Δϕ = q[ϕ, i_env].Mid(k) - q[ϕ, i_gm].Mid(k)
                Δψ = q[ψ, i_env].Mid(k) - q[ψ, i_gm].Mid(k)
            else:
                Δϕ = q[ϕ, i_env][k] - q[ϕ, i_gm][k]
                Δψ = q[ψ, i_env][k] - q[ψ, i_gm][k]

            q[cv, i_env][k] = q[cv, i_gm][k] - tke_factor * ae[k] * Δϕ * Δψ
            for i in i_uds:
                if is_tke:
                    Δϕ = q[ϕ, i].Mid(k) - q[ϕ, i_gm].Mid(k)
                    Δψ = q[ψ, i].Mid(k) - q[ψ, i_gm].Mid(k)
                else:
                    Δϕ = q[ϕ, i][k] - q[ϕ, i_gm][k]
                    Δψ = q[ψ, i][k] - q[ψ, i_gm][k]

                q[cv, i_env][k] -= tke_factor * q['a_tmp', i][k] * Δϕ * Δψ
            q[cv, i_env][k] = q[cv, i_env][k]/ae[k]
        else:
            q[cv, i_env][k] = 0.0
    return

def diagnose_environment(grid, q):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems(Center()):
        a_env = q['a', i_env][k]
        q['q_tot', i_env][k] = (q['q_tot', i_gm][k] - sum([q['a', i][k]*q['q_tot_tmp', i][k] for i in i_uds]))/a_env
        q['θ_liq', i_env][k] = (q['θ_liq', i_gm][k] - sum([q['a', i][k]*q['θ_liq_tmp', i][k] for i in i_uds]))/a_env
        # Assuming q['w', i_gm] = 0!
        a_env = q['a', i_env].Mid(k)
        q['w', i_env][k] = (0.0 - sum([q['a', i][k]*q['w_tmp', i][k] for i in i_uds]))/a_env
    return

def compute_tendencies_gm(grid, q_tendencies, q, Case, TS, tmp, tri_diag):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    k_1 = grid.first_interior(Zmin())
    dzi = grid.dzi
    α_1 = tmp['α_0_half'][k_1]
    ae_1 = q['a', i_env][k_1]
    slice_all_c = grid.slice_all(Center())

    q_tendencies['q_tot', i_gm][slice_all_c] += [tmp['mf_tend_q_tot'][k] + tmp['prec_src_q_tot', i_gm][k]*TS.dti for k in grid.over_elems(Center())]
    q_tendencies['q_tot', i_gm][k_1] += Case.Sur.rho_q_tot_flux * dzi * α_1/ae_1

    q_tendencies['θ_liq', i_gm][slice_all_c] += [tmp['mf_tend_θ_liq'][k] + tmp['prec_src_θ_liq', i_gm][k]*TS.dti for k in grid.over_elems(Center())]
    q_tendencies['θ_liq', i_gm][k_1] += Case.Sur.rho_θ_liq_flux * dzi * α_1/ae_1

    q_tendencies['U', i_gm][k_1] += Case.Sur.rho_uflux * dzi * α_1/ae_1
    q_tendencies['V', i_gm][k_1] += Case.Sur.rho_vflux * dzi * α_1/ae_1
    return

def update_cv_env(grid, q, q_tendencies, tmp, tmp_O2, TS, cv, tri_diag, tke_diss_coeff):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    construct_tridiag_diffusion_O2(grid, q, tmp, TS, tri_diag, tke_diss_coeff)
    dti = TS.dti
    k_1 = grid.first_interior(Zmin())

    slice_all_c = grid.slice_all(Center())
    a_e = q['a', i_env]
    tri_diag.f[slice_all_c] = [tmp['ρ_0_half'][k] * a_e[k] * q[cv, i_env][k] * dti + q_tendencies[cv, i_env][k] for k in grid.over_elems(Center())]
    tri_diag.f[k_1] = tmp['ρ_0_half'][k_1] * a_e[k_1] * q[cv, i_env][k_1] * dti + q[cv, i_env][k_1]
    solve_tridiag_wrapper(grid, q[cv, i_env], tri_diag)

    return

def update_GMV_MF(grid, q, TS, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    slice_all_c = grid.slice_real(Center())

    for i in i_uds:
        tmp['mf_tmp', i][slice_all_c] = [((q['w_tmp', i][k] - q['w', i_env].values[k]) * tmp['ρ_0'][k]
                       * q['a_tmp', i].Mid(k)) for k in grid.over_elems_real(Center())]

    for k in grid.over_elems_real(Center()):
        tmp['mf_θ_liq'][k] = np.sum([tmp['mf_tmp', i][k] * (q['θ_liq_tmp', i].Mid(k) - q['θ_liq', i_env].Mid(k)) for i in i_uds])
        tmp['mf_q_tot'][k] = np.sum([tmp['mf_tmp', i][k] * (q['q_tot_tmp', i].Mid(k) - q['q_tot', i_env].Mid(k)) for i in i_uds])

    tmp['mf_tend_θ_liq'][slice_all_c] = [-tmp['α_0_half'][k]*grad(tmp['mf_θ_liq'].Dual(k), grid) for k in grid.over_elems_real(Center())]
    tmp['mf_tend_q_tot'][slice_all_c] = [-tmp['α_0_half'][k]*grad(tmp['mf_q_tot'].Dual(k), grid) for k in grid.over_elems_real(Center())]
    return

def assign_new_to_values(grid, q_new, q, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    slice_all_c = grid.slice_all(Center())
    slice_all_n = grid.slice_all(Node())
    for i in i_uds:
        q_new['w', i][slice_all_n] = [q['w_tmp', i][k] for k in grid.over_elems(Node())]
        q_new['a', i][slice_all_c] = [q['a_tmp', i][k] for k in grid.over_elems(Center())]
        q_new['q_tot', i][slice_all_c] = [q['q_tot_tmp', i][k] for k in grid.over_elems(Center())]
        q_new['q_rai', i][slice_all_c] = [q['q_rai_tmp', i][k] for k in grid.over_elems(Center())]
        q_new['θ_liq', i][slice_all_c] = [q['θ_liq_tmp', i][k] for k in grid.over_elems(Center())]
    return

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
        input_st.zi = UpdVar.cloud_base[i]
        for k in grid.over_elems_real(Center()):
            input_st.quadrature_order = quadrature_order
            input_st.z = grid.z_half[k]
            input_st.ml = tmp['l_mix'][k]
            input_st.b = tmp['B', i][k]
            input_st.w = q['w_tmp', i].Mid(k)
            input_st.af = q['a_tmp', i][k]
            input_st.tke = q['tke', i_env][k]
            input_st.qt_env = q['q_tot', i_env][k]
            input_st.q_liq_env = tmp['q_liq', i_env][k]
            input_st.θ_liq_env = q['θ_liq', i_env][k]
            input_st.b_env = tmp['B', i_env][k]
            input_st.w_env = q['w', i_env].values[k]
            input_st.θ_liq_up = q['θ_liq_tmp', i][k]
            input_st.qt_up = q['q_tot_tmp', i][k]
            input_st.q_liq_up = tmp['q_liq', i][k]
            input_st.env_Hvar = q['cv_θ_liq', i_env][k]
            input_st.env_QTvar = q['cv_q_tot', i_env][k]
            input_st.env_HQTcov = q['cv_θ_liq_q_tot', i_env][k]
            input_st.p0 = tmp['p_0_half'][k]
            input_st.alpha0 = tmp['α_0_half'][k]
            input_st.tke = q['tke', i_env][k]
            input_st.tke_ed_coeff  = tke_ed_coeff

            input_st.L = 20000.0 # need to define the scale of the GCM grid resolution
            input_st.n_up = n_updrafts

            w_cut = q['w_tmp', i].DualCut(k)
            w_env_cut = q['w', i_env].DualCut(k)
            a_cut = q['a_tmp', i].Cut(k)
            a_env_cut = (1.0-q['a_tmp', i].Cut(k))
            aw_cut = a_cut * w_cut + a_env_cut * w_env_cut

            input_st.dwdz = grad(aw_cut, grid)

            if input_st.zbl-UpdVar.cloud_base[i] > 0.0:
                input_st.poisson = np.random.poisson(grid.dz/((input_st.zbl-UpdVar.cloud_base[i])/10.0))
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

        a_ = self.surface_area/self.n_updrafts
        i_uds = range(self.n_updrafts)
        self.surface_scalar_coeff = np.zeros((self.n_updrafts,), dtype=np.double, order='c')
        # i_gm, i_env, i_ud = tmp.domain_idx()
        for i in i_uds:
            self.surface_scalar_coeff[i] = percentile_bounds_mean_norm(1.0-self.surface_area+i*a_,
                                                                       1.0-self.surface_area + (i+1)*a_ , 1000)

        # Near-surface BC of updraft area fraction
        self.area_surface_bc  = np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.w_surface_bc     = np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.θ_liq_surface_bc = np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.q_tot_surface_bc = np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        return

    def initialize(self, UpdVar, tmp, q):
        UpdVar.initialize(tmp, q)
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
        self.pre_compute_vars(grid, q, q_tendencies, tmp, tmp_O2, Case, TS, tri_diag)
        return

    def pre_export_data_compute(self, grid, q, tmp, tmp_O2, Stats):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        for k in grid.over_elems_real(Center()):
            tmp['mf_θ_liq_half'][k] = tmp['mf_θ_liq'].Mid(k)
            tmp['mf_q_tot_half'][k] = tmp['mf_q_tot'].Mid(k)
            tmp['massflux_half'][k] = tmp['mf_tmp', 0].Mid(k)
            a_bulk = sum([q['a', i][k] for i in i_uds])
            if a_bulk > 0.0:
                for i in i_uds:
                    tmp['mean_entr_sc'][k] += q['a', i][k] * tmp['entr_sc', i][k]/a_bulk
                    tmp['mean_detr_sc'][k] += q['a', i][k] * tmp['detr_sc', i][k]/a_bulk

        compute_covariance_dissipation(grid, q, tmp, tmp_O2, self.tke_diss_coeff, 'tke')
        compute_covariance_detr(grid, q, tmp, tmp_O2, 'tke')
        compute_covariance_dissipation(grid, q, tmp, tmp_O2, self.tke_diss_coeff, 'cv_θ_liq')
        compute_covariance_dissipation(grid, q, tmp, tmp_O2, self.tke_diss_coeff, 'cv_q_tot')
        compute_covariance_dissipation(grid, q, tmp, tmp_O2, self.tke_diss_coeff, 'cv_θ_liq_q_tot')
        compute_covariance_detr(grid, q, tmp, tmp_O2, 'cv_θ_liq')
        compute_covariance_detr(grid, q, tmp, tmp_O2, 'cv_q_tot')
        compute_covariance_detr(grid, q, tmp, tmp_O2, 'cv_θ_liq_q_tot')

    def initialize_io(self, Stats, UpdVar):
        UpdVar.initialize_io(Stats)
        return

    def export_data(self, grid, q, tmp, tmp_O2, Stats, UpdVar):
        self.pre_export_data_compute(grid, q, tmp, tmp_O2, Stats)
        UpdVar.export_data(grid, q, tmp, Stats)
        return

    def set_updraft_surface_bc(self, grid, q, Case, tmp):
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
            self.area_surface_bc[i] = self.surface_area/self.n_updrafts
            self.w_surface_bc[i] = 0.0
            self.θ_liq_surface_bc[i] = (θ_liq_1 + self.surface_scalar_coeff[i] * np.sqrt(cv_θ_liq))
            self.q_tot_surface_bc[i] = (q_tot_1 + self.surface_scalar_coeff[i] * np.sqrt(cv_q_tot))
        return

    def pre_compute_vars(self, grid, q, q_tendencies, tmp, tmp_O2, Case, TS, tri_diag):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        self.zi = compute_inversion(grid, q, Case.inversion_option, tmp, self.Ri_bulk_crit, tmp['temp_C'])
        self.wstar = compute_convective_velocity(Case.Sur.bflux, self.zi)
        diagnose_environment(grid, q)
        compute_cv_gm(grid, q, 'w_tmp'    , 'w_tmp'    , 'tke')
        compute_cv_gm(grid, q, 'θ_liq_tmp', 'θ_liq_tmp', 'cv_θ_liq')
        compute_cv_gm(grid, q, 'q_tot_tmp', 'q_tot_tmp', 'cv_q_tot')
        compute_cv_gm(grid, q, 'θ_liq_tmp', 'q_tot_tmp', 'cv_θ_liq_q_tot')
        update_GMV_MF(grid, q, TS, tmp)
        compute_eddy_diffusivities_tke(grid, q, tmp, Case, self.zi, self.wstar, self.prandtl_number, self.tke_ed_coeff, self.similarity_diffusivity)

        we = q['w', i_env]
        compute_tke_buoy(grid, q, tmp, tmp_O2, 'tke')
        compute_covariance_entr(grid, q, tmp, tmp_O2, 'w_tmp'    , 'w_tmp'    , 'tke')
        compute_covariance_shear(grid, q, tmp, tmp_O2, 'w_tmp'    , 'w_tmp'    , 'tke')
        compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, 'w_tmp'    , 'w_tmp'    , 'tke')
        compute_tke_pressure(grid, q, tmp, tmp_O2, self.pressure_buoy_coeff, self.pressure_drag_coeff, self.pressure_plume_spacing, 'tke')
        compute_covariance_entr(grid, q, tmp, tmp_O2, 'θ_liq_tmp', 'θ_liq_tmp', 'cv_θ_liq')
        compute_covariance_entr(grid, q, tmp, tmp_O2, 'q_tot_tmp', 'q_tot_tmp', 'cv_q_tot')
        compute_covariance_entr(grid, q, tmp, tmp_O2, 'θ_liq_tmp', 'q_tot_tmp', 'cv_θ_liq_q_tot')
        compute_covariance_shear(grid, q, tmp, tmp_O2, 'θ_liq_tmp', 'θ_liq_tmp', 'cv_θ_liq')
        compute_covariance_shear(grid, q, tmp, tmp_O2, 'q_tot_tmp', 'q_tot_tmp', 'cv_q_tot')
        compute_covariance_shear(grid, q, tmp, tmp_O2, 'θ_liq_tmp', 'q_tot_tmp', 'cv_θ_liq_q_tot')
        compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, 'θ_liq_tmp', 'θ_liq_tmp', 'cv_θ_liq')
        compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, 'q_tot_tmp', 'q_tot_tmp', 'cv_q_tot')
        compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, 'θ_liq_tmp', 'q_tot_tmp', 'cv_θ_liq_q_tot')
        compute_covariance_rain(grid, q, tmp, tmp_O2, TS, 'tke')
        compute_covariance_rain(grid, q, tmp, tmp_O2, TS, 'cv_θ_liq')
        compute_covariance_rain(grid, q, tmp, tmp_O2, TS, 'cv_q_tot')
        compute_covariance_rain(grid, q, tmp, tmp_O2, TS, 'cv_θ_liq_q_tot')

        reset_surface_covariance(grid, q, tmp, Case, self.wstar)

        compute_cv_env(grid, q, tmp, tmp_O2, 'w_tmp'    , 'w_tmp'    , 'tke')
        compute_cv_env(grid, q, tmp, tmp_O2, 'θ_liq_tmp', 'θ_liq_tmp', 'cv_θ_liq')
        compute_cv_env(grid, q, tmp, tmp_O2, 'q_tot_tmp', 'q_tot_tmp', 'cv_q_tot')
        compute_cv_env(grid, q, tmp, tmp_O2, 'θ_liq_tmp', 'q_tot_tmp', 'cv_θ_liq_q_tot')

        compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, 'tke')
        compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, 'cv_θ_liq')
        compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, 'cv_q_tot')
        compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, 'cv_θ_liq_q_tot')

        compute_tendencies_gm(grid, q_tendencies, q, Case, TS, tmp, tri_diag)

        cleanup_covariance(grid, q)
        self.set_updraft_surface_bc(grid, q, Case, tmp)

    def update(self, grid, q_new, q, q_tendencies, tmp, tmp_O2, UpdVar, Case, TS, tri_diag):

        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        self.pre_compute_vars(grid, q, q_tendencies, tmp, tmp_O2, Case, TS, tri_diag)

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
        u_max = np.max([q['w_tmp', i][k] for i in i_uds for k in grid.over_elems(Node())])
        self.dt_upd = np.minimum(TS.dt, 0.5 * grid.dz/np.fmax(u_max,1e-10))
        while time_elapsed < TS.dt:
            compute_entrainment_detrainment(grid, UpdVar, Case, tmp, q, self.entr_detr_fp, self.wstar, self.tke_ed_coeff, self.entrainment_factor, self.detrainment_factor)
            eos_update_SA_mean(grid, q, False, tmp, self.max_supersaturation)
            buoyancy(grid, q, tmp)
            compute_sources(grid, q, tmp, self.max_supersaturation)
            update_updraftvars(grid, q, tmp)

            self.solve_updraft_velocity_area(grid, q_new, q, q_tendencies, tmp, UpdVar, TS)
            self.solve_updraft_scalars(grid, q_new, q, q_tendencies, tmp, UpdVar, TS)
            for i in i_uds: q['θ_liq_tmp', i].apply_bc(grid, 0.0)
            for i in i_uds: q['q_tot_tmp', i].apply_bc(grid, 0.0)
            for i in i_uds: q['q_rai_tmp', i].apply_bc(grid, 0.0)
            q['w', i_env].apply_bc(grid, 0.0)
            q['θ_liq', i_env].apply_bc(grid, 0.0)
            q['q_tot', i_env].apply_bc(grid, 0.0)
            assign_values_to_new(grid, q, q_new, tmp)
            time_elapsed += self.dt_upd
            u_max = np.max([q['w_tmp', i][k] for i in i_uds for k in grid.over_elems(Node())])
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
            au_lim = self.area_surface_bc[i] * self.max_area_factor
            for k in grid.over_elems_real(Center()):

                a_k = q['a_tmp', i][k]
                α_0_kp = tmp['α_0_half'][k]
                w_k = q['w_tmp', i].Mid(k)

                w_cut = q['w_tmp', i].DualCut(k)
                a_cut = q['a_tmp', i].Cut(k)
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
            q_new['a', i][k_1] = self.area_surface_bc[i]


        for k in grid.over_elems(Center()):
            for i in i_uds:
                q['a', i][k] = q_new['a', i][k]
            q['a', i_env][k] = 1.0 - sum([q_new['a', i][k] for i in i_uds])

        # Solve for updraft velocity
        for i in i_uds:
            q_new['a', i][kb_1] = self.w_surface_bc[i]
            for k in grid.over_elems_real(Center()):
                a_new_k = q_new['a', i].Mid(k)
                if a_new_k >= self.minimum_area:

                    ρ_k = tmp['ρ_0'][k]
                    w_i = q['w_tmp', i][k]
                    w_env = q['w', i_env].values[k]
                    a_k = q['a_tmp', i].Mid(k)
                    entr_w = tmp['entr_sc', i].Mid(k)
                    detr_w = tmp['detr_sc', i].Mid(k)
                    B_k = tmp['B', i].Mid(k)

                    a_cut = q['a_tmp', i].DualCut(k)
                    ρ_cut = tmp['ρ_0'].Cut(k)
                    w_cut = q['w_tmp', i].Cut(k)

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
            q_new['θ_liq', i][k_1] = self.θ_liq_surface_bc[i]
            q_new['q_tot', i][k_1] = self.q_tot_surface_bc[i]

            for k in grid.over_elems_real(Center())[1:]:
                dt_ = 1.0/dti_
                θ_liq_env = q['θ_liq', i_env][k]
                q_tot_env = q['q_tot', i_env][k]

                if q_new['a', i][k] >= self.minimum_area:
                    a_k = q['a_tmp', i][k]
                    a_cut = q['a_tmp', i].Cut(k)
                    a_k_new = q_new['a', i][k]
                    θ_liq_cut = q['θ_liq_tmp', i].Cut(k)
                    q_tot_cut = q['q_tot_tmp', i].Cut(k)
                    ρ_k = tmp['ρ_0_half'][k]
                    ρ_cut = tmp['ρ_0_half'].Cut(k)
                    w_cut = q['w_tmp', i].DualCut(k)
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
