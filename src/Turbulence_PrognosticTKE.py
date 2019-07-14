import numpy as np
from parameters import *
import sys
from EDMF_Updrafts import *
from EDMF_Environment import *
from Operators import advect, grad, Laplacian, grad_pos, grad_neg
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann

from TriDiagSolver import solve_tridiag_wrapper, construct_tridiag_diffusion_O1, construct_tridiag_diffusion_O2
from Variables import VariablePrognostic, VariableDiagnostic, GridMeanVariables
from Surface import SurfaceBase
from Cases import  CasesBase
from TimeStepping import TimeStepping
from NetCDFIO import NetCDFIO_Stats
from funcs_thermo import  *
from funcs_turbulence import *
from funcs_utility import *

def compute_grid_means(grid, q, tmp, GMV, EnvVar, UpdVar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]
    au = UpdVar.Area.values
    for k in grid.over_elems_real(Center()):
        GMV.q_liq.values[k] = ae[k] * EnvVar.q_liq.values[k] + sum([ au[i][k] * UpdVar.q_liq.values[i][k] for i in i_uds])
        GMV.q_rai.values[k] = ae[k] * EnvVar.q_rai.values[k] + sum([ au[i][k] * UpdVar.q_rai.values[i][k] for i in i_uds])
        GMV.T.values[k]     = ae[k] * EnvVar.T.values[k]     + sum([ au[i][k] * UpdVar.T.values[i][k] for i in i_uds])
        GMV.B.values[k]     = ae[k] * EnvVar.B.values[k]     + sum([ au[i][k] * UpdVar.B.values[i][k] for i in i_uds])
    return

def get_GMV_CoVar(grid, q, au, phi_u, psi_u, phi_e,  psi_e, covar_e, gmv_phi, gmv_psi, gmv_covar, name):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    tke_factor = 0.5 if name == 'tke' else 1.0
    ae = q['a', i_env]
    for k in grid.over_elems(Center()):
        if name == 'tke':
            phi_diff = phi_e.Mid(k) - gmv_phi.Mid(k)
            psi_diff = psi_e.Mid(k) - gmv_psi.Mid(k)
        else:
            phi_diff = phi_e[k]-gmv_phi[k]
            psi_diff = psi_e[k]-gmv_psi[k]

        gmv_covar[k] = tke_factor * ae[k] * phi_diff * psi_diff + ae[k] * covar_e[k]
        for i in i_uds:
            if name == 'tke':
                phi_diff = phi_u[i].Mid(k) - gmv_phi.Mid(k)
                psi_diff = psi_u[i].Mid(k) - gmv_psi.Mid(k)
            else:
                phi_diff = phi_u[i][k]-gmv_phi[k]
                psi_diff = psi_u[i][k]-gmv_psi[k]

            gmv_covar[k] += tke_factor * au[i][k] * phi_diff * psi_diff
    return

def compute_covariance_entr(grid, q, tmp, tmp_O2, UpdVar, Covar, UpdVar1, UpdVar2, EnvVar1, EnvVar2, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    is_tke = cv=='tke'
    tke_factor = 0.5 if is_tke else 1.0
    for k in grid.over_elems_real(Center()):
        Covar.entr_gain[k] = 0.0
        for i in i_uds:
            if is_tke:
                updvar1 = UpdVar1.values[i].Mid(k)
                updvar2 = UpdVar2.values[i].Mid(k)
                envvar1 = EnvVar1.Mid(k)
                envvar2 = EnvVar2.Mid(k)
            else:
                updvar1 = UpdVar1.values[i][k]
                updvar2 = UpdVar2.values[i][k]
                envvar1 = EnvVar1[k]
                envvar2 = EnvVar2[k]
            w_u = UpdVar.W.values[i].Mid(k)
            Covar.entr_gain[k] += tke_factor*UpdVar.Area.values[i][k] * np.fabs(w_u) * tmp['detr_sc', i][k] * \
                                         (updvar1 - envvar1) * (updvar2 - envvar2)
        Covar.entr_gain[k] *= tmp['ρ_0_half'][k]
    return

def compute_covariance_shear(grid, q, tmp, tmp_O2, GMV, Covar, UpdVar1, UpdVar2, EnvVar1, EnvVar2, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]
    is_tke = cv=='tke'
    tke_factor = 0.5 if is_tke else 1.0
    grad_u = 0.0
    grad_v = 0.0
    for k in grid.over_elems_real(Center()):
        if is_tke:
            grad_u = grad_neg(GMV.U.values.Cut(k), grid)
            grad_v = grad_neg(GMV.V.values.Cut(k), grid)
            grad_var2 = grad_neg(EnvVar2.Cut(k), grid)
            grad_var1 = grad_neg(EnvVar1.Cut(k), grid)
        else:
            grad_var2 = grad(EnvVar2.Cut(k), grid)
            grad_var1 = grad(EnvVar1.Cut(k), grid)
        ρaK = tmp['ρ_0_half'][k] * ae[k] * tmp['K_h'][k]
        Covar.shear[k] = tke_factor*2.0*ρaK * (grad_var1*grad_var2 + grad_u**2.0 + grad_v**2.0)
    return

def compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, au, phi_u, psi_u, phi_e, psi_e, Covar, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    is_tke = cv=='tke'
    tke_factor = 0.5 if is_tke else 1.0
    for k in grid.over_elems(Center()):
        Covar.interdomain[k] = 0.0
        for i in i_uds:
            if is_tke:
                phi_diff = phi_u.values[i].Mid(k) - phi_e.Mid(k)
                psi_diff = psi_u.values[i].Mid(k) - psi_e.Mid(k)
            else:
                phi_diff = phi_u.values[i][k]-phi_e[k]
                psi_diff = psi_u.values[i][k]-psi_e[k]

            Covar.interdomain[k] += tke_factor*au.values[i][k] * (1.0-au.values[i][k]) * phi_diff * psi_diff
    return

def compute_covariance_detr(grid, q, tmp, tmp_O2, Covar, UpdVar, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]

    for k in grid.over_elems_real(Center()):
        Covar.detr_loss[k] = 0.0
        for i in i_uds:
            w_u = UpdVar.W.values[i].Mid(k)
            Covar.detr_loss[k] += UpdVar.Area.values[i][k] * np.fabs(w_u) * tmp['entr_sc', i][k]
        Covar.detr_loss[k] *= tmp['ρ_0_half'][k] * Covar.values[k]
    return

def compute_covariance_rain(grid, q, tmp, tmp_O2, TS, GMV, EnvVar, EnvThermo, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]
    for k in grid.over_elems_real(Center()):
        ρa_0 = tmp['ρ_0_half'][k]*ae[k]
        if cv=='tke':            EnvVar.tke.rain_src[k] = 0.0
        if cv=='cv_θ_liq':       EnvVar.cv_θ_liq.rain_src[k]       = ρa_0 * 2. * EnvThermo.cv_θ_liq_rain_dt[k]       * TS.dti
        if cv=='cv_q_tot':       EnvVar.cv_q_tot.rain_src[k]       = ρa_0 * 2. * EnvThermo.cv_q_tot_rain_dt[k]       * TS.dti
        if cv=='cv_θ_liq_q_tot': EnvVar.cv_θ_liq_q_tot.rain_src[k] = ρa_0 *      EnvThermo.cv_θ_liq_q_tot_rain_dt[k] * TS.dti
    return

def compute_covariance_dissipation(grid, q, tmp, tmp_O2, Covar, EnvVar, tke_diss_coeff, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]
    for k in grid.over_elems_real(Center()):
        l_mix = np.fmax(tmp['l_mix'][k], 1.0)
        tke_env = np.fmax(EnvVar.tke.values[k], 0.0)

        Covar.dissipation[k] = (tmp['ρ_0_half'][k] * ae[k] * Covar.values[k] * pow(tke_env, 0.5)/l_mix * tke_diss_coeff)
    return

def compute_tke_pressure(grid, q, tmp, tmp_O2, EnvVar, UpdVar, pressure_buoy_coeff, pressure_drag_coeff, pressure_plume_spacing, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems_real(Center()):
        EnvVar.tke.press[k] = 0.0
        for i in i_uds:
            wu_half = UpdVar.W.values[i].Mid(k)
            we_half = q['w', i_env].Mid(k)
            a_i = UpdVar.Area.values[i][k]
            ρ_0_k = tmp['ρ_0_half'][k]
            press_buoy = (-1.0 * ρ_0_k * a_i * UpdVar.B.values[i][k] * pressure_buoy_coeff)
            press_drag_coeff = -1.0 * ρ_0_k * np.sqrt(a_i) * pressure_drag_coeff/pressure_plume_spacing
            press_drag = press_drag_coeff * (wu_half - we_half)*np.fabs(wu_half - we_half)
            EnvVar.tke.press[k] += (we_half - wu_half) * (press_buoy + press_drag)
    return

def cleanup_covariance(grid, q, GMV, EnvVar, UpdVar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    tmp_eps = 1e-18
    for k in grid.over_elems_real(Center()):
        if GMV.tke.values[k] < tmp_eps:                        GMV.tke.values[k]               = 0.0
        if GMV.cv_θ_liq.values[k] < tmp_eps:                   GMV.cv_θ_liq.values[k]          = 0.0
        if GMV.cv_q_tot.values[k] < tmp_eps:                   GMV.cv_q_tot.values[k]          = 0.0
        if np.fabs(GMV.cv_θ_liq_q_tot.values[k]) < tmp_eps:    GMV.cv_θ_liq_q_tot.values[k]    = 0.0
        if EnvVar.cv_θ_liq.values[k] < tmp_eps:                EnvVar.cv_θ_liq.values[k]       = 0.0
        if EnvVar.tke.values[k] < tmp_eps:                     EnvVar.tke.values[k]            = 0.0
        if EnvVar.cv_q_tot.values[k] < tmp_eps:                EnvVar.cv_q_tot.values[k]       = 0.0
        if np.fabs(EnvVar.cv_θ_liq_q_tot.values[k]) < tmp_eps: EnvVar.cv_θ_liq_q_tot.values[k] = 0.0

def get_env_covar_from_GMV(grid, q, tmp, tmp_O2, au, phi_u, psi_u, phi_e, psi_e, covar_e, gmv_phi, gmv_psi, gmv_covar, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    is_tke = cv=='tke'
    tke_factor = 0.5 if is_tke else 1.0
    ae = q['a', i_env]

    for k in grid.over_elems(Center()):
        if ae[k] > 0.0:
            if is_tke:
                phi_diff = phi_e.Mid(k) - gmv_phi.Mid(k)
                psi_diff = psi_e.Mid(k) - gmv_psi.Mid(k)
            else:
                phi_diff = phi_e[k] - gmv_phi[k]
                psi_diff = psi_e[k] - gmv_psi[k]

            covar_e[k] = gmv_covar[k] - tke_factor * ae[k] * phi_diff * psi_diff
            for i in i_uds:
                if is_tke:
                    phi_diff = phi_u.values[i].Mid(k) - gmv_phi.Mid(k)
                    psi_diff = psi_u.values[i].Mid(k) - gmv_psi.Mid(k)
                else:
                    phi_diff = phi_u.values[i][k] - gmv_phi[k]
                    psi_diff = psi_u.values[i][k] - gmv_psi[k]

                covar_e[k] -= tke_factor * au.values[i][k] * phi_diff * psi_diff
            covar_e[k] = covar_e[k]/ae[k]
        else:
            covar_e[k] = 0.0
    return


def reset_surface_covariance(grid, q, tmp, GMV, Case, wstar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    flux1 = Case.Sur.rho_θ_liq_flux
    flux2 = Case.Sur.rho_q_tot_flux
    k_1 = grid.first_interior(Zmin())
    zLL = grid.z_half[k_1]
    alpha0LL  = tmp['α_0_half'][k_1]
    ustar = Case.Sur.ustar
    oblength = Case.Sur.obukhov_length
    GMV.tke.values[k_1]            = get_surface_tke(Case.Sur.ustar, wstar, zLL, Case.Sur.obukhov_length)
    GMV.cv_θ_liq.values[k_1]       = get_surface_variance(flux1*alpha0LL,flux1*alpha0LL, ustar, zLL, oblength)
    GMV.cv_q_tot.values[k_1]       = get_surface_variance(flux2*alpha0LL,flux2*alpha0LL, ustar, zLL, oblength)
    GMV.cv_θ_liq_q_tot.values[k_1] = get_surface_variance(flux1*alpha0LL,flux2*alpha0LL, ustar, zLL, oblength)
    return

# Find values of environmental variables by subtracting updraft values from grid mean values
def decompose_environment(grid, q, GMV, EnvVar, UpdVar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems(Center()):
        a_env = q['a', i_env][k]
        EnvVar.q_tot.values[k] = (GMV.q_tot.values[k] - sum([q['a', i][k]*UpdVar.q_tot.values[i][k] for i in i_uds]))/a_env
        EnvVar.θ_liq.values[k] = (GMV.θ_liq.values[k] - sum([q['a', i][k]*UpdVar.θ_liq.values[i][k] for i in i_uds]))/a_env
        # Assuming GMV.W = 0!
        a_env = q['a', i_env].Mid(k)
        q['w', i_env][k] = (0.0 - sum([q['a', i][k]*UpdVar.W.values[i][k] for i in i_uds]))/a_env
    return

def compute_tendencies_gm(grid, q_tendencies, q, GMV, UpdMicro, Case, TS, tmp, tri_diag):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    k_1 = grid.first_interior(Zmin())
    dzi = grid.dzi
    α_1 = tmp['α_0_half'][k_1]
    ae_1 = q['a', i_env][k_1]
    slice_all_c = grid.slice_all(Center())

    q_tendencies['q_tot', i_gm][slice_all_c] += [tmp['mf_tend_q_tot'][k] + UpdMicro.prec_src_q_tot_tot[k]*TS.dti for k in grid.over_elems(Center())]
    q_tendencies['q_tot', i_gm][k_1] += Case.Sur.rho_q_tot_flux * dzi * α_1/ae_1

    q_tendencies['θ_liq', i_gm][slice_all_c] += [tmp['mf_tend_θ_liq'][k] + UpdMicro.prec_src_θ_liq_tot[k]*TS.dti for k in grid.over_elems(Center())]
    q_tendencies['θ_liq', i_gm][k_1] += Case.Sur.rho_θ_liq_flux * dzi * α_1/ae_1

    q_tendencies['U', i_gm][k_1] += Case.Sur.rho_uflux * dzi * α_1/ae_1
    q_tendencies['V', i_gm][k_1] += Case.Sur.rho_vflux * dzi * α_1/ae_1
    return

def update_sol_gm(grid, q, q_tendencies, GMV, TS, tmp, tri_diag):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ρ_0_half = tmp['ρ_0_half']
    ae = q['a', i_env]
    slice_real_n = grid.slice_real(Node())
    slice_all_c = grid.slice_all(Center())

    tri_diag.ρaK[slice_real_n] = [ae.Mid(k)*tmp['K_h'].Mid(k)*ρ_0_half.Mid(k) for k in grid.over_elems_real(Node())]
    construct_tridiag_diffusion_O1(grid, TS.dt, tri_diag, ρ_0_half, ae)
    tri_diag.f[slice_all_c] = [GMV.q_tot.values[k] + TS.dt*q_tendencies['q_tot', i_gm][k] for k in grid.over_elems(Center())]
    solve_tridiag_wrapper(grid, GMV.q_tot.new, tri_diag)
    tri_diag.f[slice_all_c] = [GMV.θ_liq.values[k] + TS.dt*q_tendencies['θ_liq', i_gm][k] for k in grid.over_elems(Center())]
    solve_tridiag_wrapper(grid, GMV.θ_liq.new, tri_diag)

    tri_diag.ρaK[slice_real_n] = [ae.Mid(k)*tmp['K_m'].Mid(k)*ρ_0_half.Mid(k) for k in grid.over_elems_real(Node())]
    construct_tridiag_diffusion_O1(grid, TS.dt, tri_diag, ρ_0_half, ae)
    tri_diag.f[slice_all_c] = [GMV.U.values[k] + TS.dt*q_tendencies['U', i_gm][k] for k in grid.over_elems(Center())]
    solve_tridiag_wrapper(grid, GMV.U.new, tri_diag)
    tri_diag.f[slice_all_c] = [GMV.V.values[k] + TS.dt*q_tendencies['V', i_gm][k] for k in grid.over_elems(Center())]
    solve_tridiag_wrapper(grid, GMV.V.new, tri_diag)
    return

def compute_zbl_qt_grad(grid, q, GMV):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    # computes inversion height as z with max gradient of q_tot
    zbl_q_tot = 0.0
    q_tot_grad = 0.0
    for k in grid.over_elems_real(Center()):
        q_tot_grad_new = grad(GMV.q_tot.values.Dual(k), grid)
        if np.fabs(q_tot_grad) > q_tot_grad:
            q_tot_grad = np.fabs(q_tot_grad_new)
            zbl_q_tot = grid.z_half[k]
    return zbl_q_tot

def compute_inversion(grid, q, GMV, option, tmp, Ri_bulk_crit, temp_C):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    maxgrad = 0.0
    theta_rho_bl = temp_C.first_interior(grid)
    for k in grid.over_elems_real(Center()):
        q_tot = GMV.q_tot.values[k]
        q_vap = q_tot - GMV.q_liq.values[k]
        temp_C[k] = theta_rho_c(tmp['p_0_half'][k], GMV.T.values[k], q_tot, q_vap)
    if option == 'theta_rho':
        for k in grid.over_elems_real(Center()):
            if temp_C[k] > theta_rho_bl:
                zi = grid.z_half[k]
                break
    elif option == 'thetal_maxgrad':
        for k in grid.over_elems_real(Center()):
            grad_TH = grad(GMV.θ_liq.values.Dual(k), grid)
            if grad_TH > maxgrad:
                maxgrad = grad_TH
                zi = grid.z[k]
    elif option == 'critical_Ri':
        zi = get_inversion(temp_C, GMV.U.values, GMV.V.values, grid, Ri_bulk_crit)
    else:
        print('INVERSION HEIGHT OPTION NOT RECOGNIZED')
    return zi

def compute_mixing_length(grid, q, tmp, obukhov_length, EnvVar, zi, wstar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    tau = get_mixing_tau(zi, wstar)
    for k in grid.over_elems_real(Center()):
        l1 = tau * np.sqrt(np.fmax(EnvVar.tke.values[k],0.0))
        z_ = grid.z_half[k]
        if obukhov_length < 0.0: #unstable
            l2 = vkb * z_ * ( (1.0 - 100.0 * z_/obukhov_length)**0.2 )
        elif obukhov_length > 0.0: #stable
            l2 = vkb * z_ /  (1. + 2.7 *z_/obukhov_length)
        else:
            l2 = vkb * z_
        tmp['l_mix'][k] = np.fmax( 1.0/(1.0/np.fmax(l1,1e-10) + 1.0/l2), 1e-3)
    return

def compute_eddy_diffusivities_tke(grid, q, tmp, GMV, EnvVar, Case, zi, wstar, prandtl_number, tke_ed_coeff, similarity_diffusivity):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    compute_mixing_length(grid, q, tmp, Case.Sur.obukhov_length, EnvVar, zi, wstar)
    if similarity_diffusivity:
        compute_eddy_diffusivities_similarity_Siebesma2007(grid, Case, tmp, zi, wstar, prandtl_number)
    else:
        for k in grid.over_elems_real(Center()):
            lm = tmp['l_mix'][k]
            K_m_k = tke_ed_coeff * lm * np.sqrt(np.fmax(EnvVar.tke.values[k],0.0) )
            tmp['K_m'][k] = K_m_k
            tmp['K_h'][k] = K_m_k / prandtl_number
    return

def compute_eddy_diffusivities_similarity_Siebesma2007(grid, Case, tmp, zi, wstar, prandtl_number):
    ustar = Case.Sur.ustar
    for k in grid.over_elems_real(Center()):
        zzi = grid.z_half[k]/zi
        tmp['K_h'][k] = 0.0
        tmp['K_m'][k] = 0.0
        if zzi <= 1.0 and not (wstar<1e-6):
            tmp['K_h'][k] = vkb * ( (ustar/wstar)**3.0 + 39.0*vkb*zzi)**(1.0/3.0) * zzi * (1.0-zzi) * (1.0-zzi) * wstar * zi
            tmp['K_m'][k] = tmp['K_h'][k] * prandtl_number
    return

def compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, Covar, name):
    i_gm, i_env, i_uds, i_sd = q_tendencies.domain_idx()
    k_1 = grid.first_interior(Zmin())

    for k in grid.over_elems_real(Center()):
        q_tendencies[name, i_env][k] = Covar.press[k] + Covar.buoy[k] + Covar.shear[k] + Covar.entr_gain[k] + Covar.rain_src[k]
    q_tendencies[name, i_env][k_1] = 0.0
    return

def update_cv_env(grid, q, q_tendencies, tmp, tmp_O2, Covar, EnvVar, UpdVar, TS, name, tri_diag, tke_diss_coeff):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    construct_tridiag_diffusion_O2(grid, q, tmp, TS, UpdVar, EnvVar, tri_diag, tke_diss_coeff)
    dti = TS.dti
    k_1 = grid.first_interior(Zmin())

    slice_all_c = grid.slice_all(Center())
    a_e = q['a', i_env]
    tri_diag.f[slice_all_c] = [tmp['ρ_0_half'][k] * a_e[k] * Covar.values[k] * dti + q_tendencies[name, i_env][k] for k in grid.over_elems(Center())]
    tri_diag.f[k_1] = tmp['ρ_0_half'][k_1] * a_e[k_1] * Covar.values[k_1] * dti + Covar.values[k_1]
    solve_tridiag_wrapper(grid, Covar.values, tri_diag)

    return

def update_GMV_MF(grid, q, GMV, EnvVar, UpdVar, TS, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    slice_all_c = grid.slice_real(Center())

    for i in i_uds:
        tmp['mf_tmp', i][slice_all_c] = [((UpdVar.W.values[i][k] - q['w', i_env].values[k]) * tmp['ρ_0'][k]
                       * UpdVar.Area.values[i].Mid(k)) for k in grid.over_elems_real(Center())]

    for k in grid.over_elems_real(Center()):
        tmp['mf_θ_liq'][k] = np.sum([tmp['mf_tmp', i][k] * (UpdVar.θ_liq.values[i].Mid(k) - EnvVar.θ_liq.values.Mid(k)) for i in i_uds])
        tmp['mf_q_tot'][k] = np.sum([tmp['mf_tmp', i][k] * (UpdVar.q_tot.values[i].Mid(k) - EnvVar.q_tot.values.Mid(k)) for i in i_uds])

    tmp['mf_tend_θ_liq'][slice_all_c] = [-tmp['α_0_half'][k]*grad(tmp['mf_θ_liq'].Dual(k), grid) for k in grid.over_elems_real(Center())]
    tmp['mf_tend_q_tot'][slice_all_c] = [-tmp['α_0_half'][k]*grad(tmp['mf_q_tot'].Dual(k), grid) for k in grid.over_elems_real(Center())]
    return

def compute_tke_buoy(grid, q, tmp, tmp_O2, GMV, EnvVar, EnvThermo, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]

    # Note that source terms at the first interior point are not really used because that is where tke boundary condition is
    # enforced (according to MO similarity). Thus here I am being sloppy about lowest grid point
    for k in grid.over_elems_real(Center()):
        q_tot_dry = EnvThermo.q_tot_dry[k]
        θ_dry = EnvThermo.θ_dry[k]
        t_cloudy = EnvThermo.t_cloudy[k]
        q_vap_cloudy = EnvThermo.q_vap_cloudy[k]
        q_tot_cloudy = EnvThermo.q_tot_cloudy[k]
        θ_cloudy = EnvThermo.θ_cloudy[k]
        p_0 = tmp['p_0_half'][k]

        lh = latent_heat(t_cloudy)
        cpm = cpm_c(q_tot_cloudy)
        grad_θ_liq = grad_neg(EnvVar.θ_liq.values.Cut(k), grid)
        grad_q_tot = grad_neg(EnvVar.q_tot.values.Cut(k), grid)

        prefactor = Rd * exner_c(p_0)/p_0

        d_alpha_θ_liq_dry = prefactor * (1.0 + (eps_vi - 1.0) * q_tot_dry)
        d_alpha_q_tot_dry = prefactor * θ_dry * (eps_vi - 1.0)
        CF_env = EnvVar.CF.values[k]

        if CF_env > 0.0:
            d_alpha_θ_liq_cloudy = (prefactor * (1.0 + eps_vi * (1.0 + lh / Rv / t_cloudy) * q_vap_cloudy - q_tot_cloudy )
                                     / (1.0 + lh * lh / cpm / Rv / t_cloudy / t_cloudy * q_vap_cloudy))
            d_alpha_q_tot_cloudy = (lh / cpm / t_cloudy * d_alpha_θ_liq_cloudy - prefactor) * θ_cloudy
        else:
            d_alpha_θ_liq_cloudy = 0.0
            d_alpha_q_tot_cloudy = 0.0

        d_alpha_θ_liq_total = (CF_env * d_alpha_θ_liq_cloudy + (1.0-CF_env) * d_alpha_θ_liq_dry)
        d_alpha_q_tot_total = (CF_env * d_alpha_q_tot_cloudy + (1.0-CF_env) * d_alpha_q_tot_dry)

        K_h_k = tmp['K_h'][k]
        term_1 = - K_h_k * grad_θ_liq * d_alpha_θ_liq_total
        term_2 = - K_h_k * grad_q_tot * d_alpha_q_tot_total

        # TODO - check
        EnvVar.tke.buoy[k] = g / tmp['α_0_half'][k] * ae[k] * tmp['ρ_0_half'][k] * (term_1 + term_2)
    return

def compute_entrainment_detrainment(grid, GMV, EnvVar, UpdVar, Case, tmp, q, entr_detr_fp, wstar, tke_ed_coeff, entrainment_factor, detrainment_factor):
    quadrature_order = 3
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    UpdVar.get_cloud_base_top_cover(grid, q, tmp)
    n_updrafts = len(i_uds)

    input_st = type('', (), {})()
    input_st.wstar = wstar

    input_st.b_mean = 0
    input_st.dz = grid.dz
    input_st.zbl = compute_zbl_qt_grad(grid, q, GMV)
    for i in i_uds:
        input_st.zi = UpdVar.cloud_base[i]
        for k in grid.over_elems_real(Center()):
            input_st.quadrature_order = quadrature_order
            input_st.z = grid.z_half[k]
            input_st.ml = tmp['l_mix'][k]
            input_st.b = UpdVar.B.values[i][k]
            input_st.w = UpdVar.W.values[i].Mid(k)
            input_st.af = UpdVar.Area.values[i][k]
            input_st.tke = EnvVar.tke.values[k]
            input_st.qt_env = EnvVar.q_tot.values[k]
            input_st.q_liq_env = EnvVar.q_liq.values[k]
            input_st.θ_liq_env = EnvVar.θ_liq.values[k]
            input_st.b_env = EnvVar.B.values[k]
            input_st.w_env = q['w', i_env].values[k]
            input_st.θ_liq_up = UpdVar.θ_liq.values[i][k]
            input_st.qt_up = UpdVar.q_tot.values[i][k]
            input_st.q_liq_up = UpdVar.q_liq.values[i][k]
            input_st.env_Hvar = EnvVar.cv_θ_liq.values[k]
            input_st.env_QTvar = EnvVar.cv_q_tot.values[k]
            input_st.env_HQTcov = EnvVar.cv_θ_liq_q_tot.values[k]
            input_st.p0 = tmp['p_0_half'][k]
            input_st.alpha0 = tmp['α_0_half'][k]
            input_st.tke = EnvVar.tke.values[k]
            input_st.tke_ed_coeff  = tke_ed_coeff

            input_st.L = 20000.0 # need to define the scale of the GCM grid resolution
            input_st.n_up = n_updrafts

            w_cut = UpdVar.W.values[i].DualCut(k)
            w_env_cut = q['w', i_env].DualCut(k)
            a_cut = UpdVar.Area.values[i].Cut(k)
            a_env_cut = (1.0-UpdVar.Area.values[i].Cut(k))
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

def assign_new_to_values(grid, q, tmp, UpdVar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for i in i_uds:
        for k in grid.over_elems(Center()):
            UpdVar.W.new[i][k] = UpdVar.W.values[i][k]
            UpdVar.Area.new[i][k] = UpdVar.Area.values[i][k]
            UpdVar.q_tot.new[i][k] = UpdVar.q_tot.values[i][k]
            UpdVar.q_liq.new[i][k] = UpdVar.q_liq.values[i][k]
            UpdVar.q_rai.new[i][k] = UpdVar.q_rai.values[i][k]
            UpdVar.θ_liq.new[i][k] = UpdVar.θ_liq.values[i][k]
            UpdVar.T.new[i][k] = UpdVar.T.values[i][k]
            UpdVar.B.new[i][k] = UpdVar.B.values[i][k]
    return

def apply_bcs(grid, q, GMV, UpdVar, EnvVar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    GMV.U.set_bcs(grid)
    GMV.V.set_bcs(grid)
    GMV.θ_liq.set_bcs(grid)
    GMV.q_tot.set_bcs(grid)
    GMV.q_rai.set_bcs(grid)
    GMV.tke.set_bcs(grid)
    GMV.cv_q_tot.set_bcs(grid)
    GMV.cv_θ_liq.set_bcs(grid)
    GMV.cv_θ_liq_q_tot.set_bcs(grid)

class EDMF_PrognosticTKE:
    def __init__(self, namelist, paramlist, grid):

        self.prandtl_number = paramlist['turbulence']['prandtl_number']
        self.Ri_bulk_crit = paramlist['turbulence']['Ri_bulk_crit']

        self.n_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']

        try:
            self.use_local_micro = namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro']
        except:
            self.use_local_micro = True
            print('Turbulence--EDMF_PrognosticTKE: defaulting to local (level-by-level) microphysics')

        try:
            if str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'inverse_z':
                self.entr_detr_fp = entr_detr_inverse_z
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'dry':
                self.entr_detr_fp = entr_detr_dry
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'inverse_w':
                self.entr_detr_fp = entr_detr_inverse_w
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'b_w2':
                self.entr_detr_fp = entr_detr_b_w2
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'entr_detr_tke':
                self.entr_detr_fp = entr_detr_tke
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'entr_detr_tke2':
                self.entr_detr_fp = entr_detr_tke2
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'suselj':
                self.entr_detr_fp = entr_detr_suselj
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'none':
                self.entr_detr_fp = entr_detr_none
            else:
                print('Turbulence--EDMF_PrognosticTKE: Entrainment rate namelist option is not recognized')
        except:
            self.entr_detr_fp = entr_detr_b_w2
            print('Turbulence--EDMF_PrognosticTKE: defaulting to cloudy entrainment formulation')

        try:
            self.similarity_diffusivity = namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity']
        except:
            self.similarity_diffusivity = False
            print('Turbulence--EDMF_PrognosticTKE: defaulting to tke-based eddy diffusivity')

        # Get values from paramlist
        # set defaults at some point?
        self.surface_area           = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']
        self.max_area_factor        = paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor']
        self.entrainment_factor     = paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor']
        self.detrainment_factor     = paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor']
        self.pressure_buoy_coeff    = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff']
        self.pressure_drag_coeff    = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_drag_coeff']
        self.pressure_plume_spacing = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_plume_spacing']
        self.vel_pressure_coeff = self.pressure_drag_coeff/self.pressure_plume_spacing
        self.vel_buoy_coeff = 1.0-self.pressure_buoy_coeff
        self.tke_ed_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff']
        self.tke_diss_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff']

        # Need to code up as paramlist option?
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

    def initialize(self, GMV, UpdVar, tmp, q):
        UpdVar.initialize(GMV, tmp, q)
        return

    def initialize_vars(self, grid, q, q_tendencies, tmp, tmp_O2, GMV, EnvVar, UpdVar, UpdMicro, EnvThermo, UpdThermo, Case, TS, tri_diag):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        self.zi = compute_inversion(grid, q, GMV, Case.inversion_option, tmp, self.Ri_bulk_crit, tmp['temp_C'])
        zs = self.zi
        self.wstar = get_wstar(Case.Sur.bflux, zs)
        ws = self.wstar
        ws3 = ws**3.0
        us3 = Case.Sur.ustar**3.0
        k_1 = grid.first_interior(Zmin())
        cv_θ_liq_1 = GMV.cv_θ_liq.values[k_1]
        cv_q_tot_1 = GMV.cv_q_tot.values[k_1]
        cv_θ_liq_q_tot_1 = GMV.cv_θ_liq_q_tot.values[k_1]
        reset_surface_covariance(grid, q, tmp, GMV, Case, ws)
        if ws > 0.0:
            for k in grid.over_elems(Center()):
                z = grid.z_half[k]
                temp = ws * 1.3 * np.cbrt(us3/ws3 + 0.6 * z/zs) * np.sqrt(np.fmax(1.0-z/zs,0.0))
                GMV.tke.values[k] = temp
                GMV.cv_θ_liq.values[k]       = cv_θ_liq_1 * temp
                GMV.cv_q_tot.values[k]       = cv_q_tot_1 * temp
                GMV.cv_θ_liq_q_tot.values[k] = cv_θ_liq_q_tot_1 * temp
            reset_surface_covariance(grid, q, tmp, GMV, Case, ws)
            compute_mixing_length(grid, q, tmp, Case.Sur.obukhov_length, EnvVar, self.zi, self.wstar)
        self.pre_compute_vars(grid, q, q_tendencies, tmp, tmp_O2, GMV, EnvVar, UpdVar, UpdMicro, EnvThermo, UpdThermo, Case, TS, tri_diag)
        return

    def initialize_io(self, Stats, EnvVar, UpdVar):

        UpdVar.initialize_io(Stats)
        EnvVar.initialize_io(Stats)

        Stats.add_profile('eddy_viscosity')
        Stats.add_profile('eddy_diffusivity')
        Stats.add_profile('mean_entr_sc')
        Stats.add_profile('mean_detr_sc')
        Stats.add_profile('massflux_half')
        Stats.add_profile('mf_θ_liq_half')
        Stats.add_profile('mf_q_tot_half')
        Stats.add_profile('mf_tend_θ_liq')
        Stats.add_profile('mf_tend_q_tot')
        Stats.add_profile('l_mix')
        Stats.add_profile('updraft_q_tot_precip')
        Stats.add_profile('updraft_θ_liq_precip')

        Stats.add_profile('tke_dissipation')
        Stats.add_profile('tke_entr_gain')
        Stats.add_profile('tke_detr_loss')
        Stats.add_profile('tke_shear')
        Stats.add_profile('tke_buoy')
        Stats.add_profile('tke_press')
        Stats.add_profile('tke_interdomain')

        Stats.add_profile('cv_θ_liq_dissipation')
        Stats.add_profile('cv_q_tot_dissipation')
        Stats.add_profile('cv_θ_liq_q_tot_dissipation')
        Stats.add_profile('cv_θ_liq_entr_gain')
        Stats.add_profile('cv_q_tot_entr_gain')
        Stats.add_profile('cv_θ_liq_q_tot_entr_gain')
        Stats.add_profile('cv_θ_liq_detr_loss')
        Stats.add_profile('cv_q_tot_detr_loss')
        Stats.add_profile('cv_θ_liq_q_tot_detr_loss')
        Stats.add_profile('cv_θ_liq_shear')
        Stats.add_profile('cv_q_tot_shear')
        Stats.add_profile('cv_θ_liq_q_tot_shear')
        Stats.add_profile('cv_θ_liq_rain_src')
        Stats.add_profile('cv_q_tot_rain_src')
        Stats.add_profile('cv_θ_liq_q_tot_rain_src')
        Stats.add_profile('cv_θ_liq_interdomain')
        Stats.add_profile('cv_q_tot_interdomain')
        Stats.add_profile('cv_θ_liq_q_tot_interdomain')
        return

    def export_data(self, grid, q, tmp, tmp_O2, Stats, EnvVar, UpdVar, UpdMicro):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()

        UpdVar.export_data(grid, q, tmp, Stats)
        EnvVar.export_data(grid, q, tmp, Stats)

        Stats.write_profile_new('eddy_viscosity'  , grid, tmp['K_m'])
        Stats.write_profile_new('eddy_diffusivity', grid, tmp['K_h'])
        for k in grid.over_elems_real(Center()):
            tmp['mf_θ_liq_half'][k] = tmp['mf_θ_liq'].Mid(k)
            tmp['mf_q_tot_half'][k] = tmp['mf_q_tot'].Mid(k)
            tmp['massflux_half'][k] = tmp['mf_tmp', 0].Mid(k)
            a_bulk = sum([q['a', i][k] for i in i_uds])
            if a_bulk > 0.0:
                for i in i_uds:
                    tmp['mean_entr_sc'][k] += q['a', i][k] * tmp['entr_sc', i][k]/a_bulk
                    tmp['mean_detr_sc'][k] += q['a', i][k] * tmp['detr_sc', i][k]/a_bulk

        compute_covariance_dissipation(grid, q, tmp, tmp_O2, EnvVar.tke, EnvVar, self.tke_diss_coeff, 'tke')
        compute_covariance_detr(grid, q, tmp, tmp_O2, EnvVar.tke, UpdVar, 'tke')
        compute_covariance_dissipation(grid, q, tmp, tmp_O2, EnvVar.cv_θ_liq, EnvVar, self.tke_diss_coeff, 'cv_θ_liq')
        compute_covariance_dissipation(grid, q, tmp, tmp_O2, EnvVar.cv_q_tot, EnvVar, self.tke_diss_coeff, 'cv_q_tot')
        compute_covariance_dissipation(grid, q, tmp, tmp_O2, EnvVar.cv_θ_liq_q_tot, EnvVar, self.tke_diss_coeff, 'cv_θ_liq_q_tot')
        compute_covariance_detr(grid, q, tmp, tmp_O2, EnvVar.cv_θ_liq, UpdVar, 'cv_θ_liq')
        compute_covariance_detr(grid, q, tmp, tmp_O2, EnvVar.cv_q_tot, UpdVar, 'cv_q_tot')
        compute_covariance_detr(grid, q, tmp, tmp_O2, EnvVar.cv_θ_liq_q_tot, UpdVar, 'cv_θ_liq_q_tot')

        Stats.write_profile_new('mean_entr_sc'  , grid, tmp['mean_entr_sc'])
        Stats.write_profile_new('mean_detr_sc'  , grid, tmp['mean_detr_sc'])
        Stats.write_profile_new('massflux_half' , grid, tmp['massflux_half'])
        Stats.write_profile_new('mf_θ_liq_half' , grid, tmp['mf_θ_liq_half'])
        Stats.write_profile_new('mf_q_tot_half' , grid, tmp['mf_q_tot_half'])
        Stats.write_profile_new('mf_tend_θ_liq' , grid, tmp['mf_tend_θ_liq'])
        Stats.write_profile_new('mf_tend_q_tot' , grid, tmp['mf_tend_q_tot'])
        Stats.write_profile_new('l_mix'         , grid, tmp['l_mix'])
        Stats.write_profile_new('updraft_q_tot_precip' , grid, UpdMicro.prec_src_q_tot_tot)
        Stats.write_profile_new('updraft_θ_liq_precip' , grid, UpdMicro.prec_src_θ_liq_tot)

        Stats.write_profile_new('tke_dissipation' , grid, EnvVar.tke.dissipation)
        Stats.write_profile_new('tke_entr_gain'   , grid, EnvVar.tke.entr_gain)
        Stats.write_profile_new('tke_detr_loss'   , grid, EnvVar.tke.detr_loss)
        Stats.write_profile_new('tke_shear'       , grid, EnvVar.tke.shear)
        Stats.write_profile_new('tke_buoy'        , grid, EnvVar.tke.buoy)
        Stats.write_profile_new('tke_press'       , grid, EnvVar.tke.press)
        Stats.write_profile_new('tke_interdomain' , grid, EnvVar.tke.interdomain)

        Stats.write_profile_new('cv_θ_liq_dissipation'       , grid, EnvVar.cv_θ_liq.dissipation)
        Stats.write_profile_new('cv_q_tot_dissipation'       , grid, EnvVar.cv_q_tot.dissipation)
        Stats.write_profile_new('cv_θ_liq_q_tot_dissipation' , grid, EnvVar.cv_θ_liq_q_tot.dissipation)
        Stats.write_profile_new('cv_θ_liq_entr_gain'         , grid, EnvVar.cv_θ_liq.entr_gain)
        Stats.write_profile_new('cv_q_tot_entr_gain'         , grid, EnvVar.cv_q_tot.entr_gain)
        Stats.write_profile_new('cv_θ_liq_q_tot_entr_gain'   , grid, EnvVar.cv_θ_liq_q_tot.entr_gain)
        Stats.write_profile_new('cv_θ_liq_detr_loss'         , grid, EnvVar.cv_θ_liq.detr_loss)
        Stats.write_profile_new('cv_q_tot_detr_loss'         , grid, EnvVar.cv_q_tot.detr_loss)
        Stats.write_profile_new('cv_θ_liq_q_tot_detr_loss'   , grid, EnvVar.cv_θ_liq_q_tot.detr_loss)
        Stats.write_profile_new('cv_θ_liq_shear'             , grid, EnvVar.cv_θ_liq.shear)
        Stats.write_profile_new('cv_q_tot_shear'             , grid, EnvVar.cv_q_tot.shear)
        Stats.write_profile_new('cv_θ_liq_q_tot_shear'       , grid, EnvVar.cv_θ_liq_q_tot.shear)
        Stats.write_profile_new('cv_θ_liq_rain_src'          , grid, EnvVar.cv_θ_liq.rain_src)
        Stats.write_profile_new('cv_q_tot_rain_src'          , grid, EnvVar.cv_q_tot.rain_src)
        Stats.write_profile_new('cv_θ_liq_q_tot_rain_src'    , grid, EnvVar.cv_θ_liq_q_tot.rain_src)
        Stats.write_profile_new('cv_θ_liq_interdomain'       , grid, EnvVar.cv_θ_liq.interdomain)
        Stats.write_profile_new('cv_q_tot_interdomain'       , grid, EnvVar.cv_q_tot.interdomain)
        Stats.write_profile_new('cv_θ_liq_q_tot_interdomain' , grid, EnvVar.cv_θ_liq_q_tot.interdomain)
        return

    def set_updraft_surface_bc(self, grid, q, GMV, Case, tmp):
        i_gm, i_env, i_uds, i_sd = tmp.domain_idx()
        k_1 = grid.first_interior(Zmin())
        zLL = grid.z_half[k_1]
        θ_liq_1 = GMV.θ_liq.values[k_1]
        q_tot_1 = GMV.q_tot.values[k_1]
        alpha0LL  = tmp['α_0_half'][k_1]
        S = Case.Sur
        cv_q_tot = get_surface_variance(S.rho_q_tot_flux*alpha0LL, S.rho_q_tot_flux*alpha0LL, S.ustar, zLL, S.obukhov_length)
        cv_θ_liq = get_surface_variance(S.rho_θ_liq_flux*alpha0LL, S.rho_θ_liq_flux*alpha0LL, S.ustar, zLL, S.obukhov_length)
        for i in i_uds:
            self.area_surface_bc[i] = self.surface_area/self.n_updrafts
            self.w_surface_bc[i] = 0.0
            self.θ_liq_surface_bc[i] = (θ_liq_1 + self.surface_scalar_coeff[i] * np.sqrt(cv_θ_liq))
            self.q_tot_surface_bc[i] = (q_tot_1 + self.surface_scalar_coeff[i] * np.sqrt(cv_q_tot))
        return

    def pre_compute_vars(self, grid, q, q_tendencies, tmp, tmp_O2, GMV, EnvVar, UpdVar, UpdMicro, EnvThermo, UpdThermo, Case, TS, tri_diag):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        self.zi = compute_inversion(grid, q, GMV, Case.inversion_option, tmp, self.Ri_bulk_crit, tmp['temp_C'])
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)
        decompose_environment(grid, q, GMV, EnvVar, UpdVar)
        get_GMV_CoVar(grid, q, UpdVar.Area.values, UpdVar.W.values,  UpdVar.W.values,  q['w', i_env],  q['w', i_env],  EnvVar.tke.values,    GMV.W.values,  GMV.W.values,  GMV.tke.values, 'tke')
        get_GMV_CoVar(grid, q, UpdVar.Area.values, UpdVar.θ_liq.values,  UpdVar.θ_liq.values, EnvVar.θ_liq.values, EnvVar.θ_liq.values, EnvVar.cv_θ_liq.values,       GMV.θ_liq.values, GMV.θ_liq.values, GMV.cv_θ_liq.values      , '')
        get_GMV_CoVar(grid, q, UpdVar.Area.values, UpdVar.q_tot.values,  UpdVar.q_tot.values, EnvVar.q_tot.values, EnvVar.q_tot.values, EnvVar.cv_q_tot.values,       GMV.q_tot.values, GMV.q_tot.values, GMV.cv_q_tot.values      , '')
        get_GMV_CoVar(grid, q, UpdVar.Area.values, UpdVar.θ_liq.values,  UpdVar.q_tot.values, EnvVar.θ_liq.values, EnvVar.q_tot.values, EnvVar.cv_θ_liq_q_tot.values, GMV.θ_liq.values, GMV.q_tot.values, GMV.cv_θ_liq_q_tot.values, '')
        update_GMV_MF(grid, q, GMV, EnvVar, UpdVar, TS, tmp)
        compute_eddy_diffusivities_tke(grid, q, tmp, GMV, EnvVar, Case, self.zi, self.wstar, self.prandtl_number, self.tke_ed_coeff, self.similarity_diffusivity)

        we = q['w', i_env]
        compute_tke_buoy(grid, q, tmp, tmp_O2, GMV, EnvVar, EnvThermo, 'tke')
        compute_covariance_entr(grid, q, tmp, tmp_O2, UpdVar, EnvVar.tke, UpdVar.W, UpdVar.W, we, we, 'tke')
        compute_covariance_shear(grid, q, tmp, tmp_O2, GMV, EnvVar.tke, UpdVar.W.values, UpdVar.W.values, we, we, 'tke')
        compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, UpdVar.Area, UpdVar.W, UpdVar.W, we, we, EnvVar.tke, 'tke')
        compute_tke_pressure(grid, q, tmp, tmp_O2, EnvVar, UpdVar, self.pressure_buoy_coeff, self.pressure_drag_coeff, self.pressure_plume_spacing, 'tke')
        compute_covariance_entr(grid, q, tmp, tmp_O2, UpdVar, EnvVar.cv_θ_liq,   UpdVar.θ_liq,  UpdVar.θ_liq,  EnvVar.θ_liq.values,  EnvVar.θ_liq.values, 'cv_θ_liq')
        compute_covariance_entr(grid, q, tmp, tmp_O2, UpdVar, EnvVar.cv_q_tot,  UpdVar.q_tot, UpdVar.q_tot, EnvVar.q_tot.values, EnvVar.q_tot.values, 'cv_q_tot')
        compute_covariance_entr(grid, q, tmp, tmp_O2, UpdVar, EnvVar.cv_θ_liq_q_tot, UpdVar.θ_liq,  UpdVar.q_tot, EnvVar.θ_liq.values,  EnvVar.q_tot.values, 'cv_θ_liq_q_tot')
        compute_covariance_shear(grid, q, tmp, tmp_O2, GMV, EnvVar.cv_θ_liq,   UpdVar.θ_liq.values,  UpdVar.θ_liq.values,  EnvVar.θ_liq.values,  EnvVar.θ_liq.values, 'cv_θ_liq')
        compute_covariance_shear(grid, q, tmp, tmp_O2, GMV, EnvVar.cv_q_tot,  UpdVar.q_tot.values, UpdVar.q_tot.values, EnvVar.q_tot.values, EnvVar.q_tot.values, 'cv_q_tot')
        compute_covariance_shear(grid, q, tmp, tmp_O2, GMV, EnvVar.cv_θ_liq_q_tot, UpdVar.θ_liq.values,  UpdVar.q_tot.values, EnvVar.θ_liq.values,  EnvVar.q_tot.values, 'cv_θ_liq_q_tot')
        compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, UpdVar.Area, UpdVar.θ_liq, UpdVar.θ_liq, EnvVar.θ_liq.values, EnvVar.θ_liq.values, EnvVar.cv_θ_liq, 'cv_θ_liq')
        compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, UpdVar.Area, UpdVar.q_tot, UpdVar.q_tot, EnvVar.q_tot.values, EnvVar.q_tot.values, EnvVar.cv_q_tot, 'cv_q_tot')
        compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, UpdVar.Area, UpdVar.θ_liq, UpdVar.q_tot, EnvVar.θ_liq.values, EnvVar.q_tot.values, EnvVar.cv_θ_liq_q_tot, 'cv_θ_liq_q_tot')
        compute_covariance_rain(grid, q, tmp, tmp_O2, TS, GMV, EnvVar, EnvThermo, 'tke')
        compute_covariance_rain(grid, q, tmp, tmp_O2, TS, GMV, EnvVar, EnvThermo, 'cv_θ_liq')
        compute_covariance_rain(grid, q, tmp, tmp_O2, TS, GMV, EnvVar, EnvThermo, 'cv_q_tot')
        compute_covariance_rain(grid, q, tmp, tmp_O2, TS, GMV, EnvVar, EnvThermo, 'cv_θ_liq_q_tot')

        reset_surface_covariance(grid, q, tmp, GMV, Case, self.wstar)

        get_env_covar_from_GMV(grid, q, tmp, tmp_O2, UpdVar.Area, UpdVar.W     , UpdVar.W    , we                  , we                 , EnvVar.tke.values           , GMV.W.values     , GMV.W.values     , GMV.tke.values           , 'tke'           )
        get_env_covar_from_GMV(grid, q, tmp, tmp_O2, UpdVar.Area, UpdVar.θ_liq , UpdVar.θ_liq, EnvVar.θ_liq.values , EnvVar.θ_liq.values, EnvVar.cv_θ_liq.values      , GMV.θ_liq.values , GMV.θ_liq.values , GMV.cv_θ_liq.values      , 'cv_θ_liq'      )
        get_env_covar_from_GMV(grid, q, tmp, tmp_O2, UpdVar.Area, UpdVar.q_tot , UpdVar.q_tot, EnvVar.q_tot.values , EnvVar.q_tot.values, EnvVar.cv_q_tot.values      , GMV.q_tot.values , GMV.q_tot.values , GMV.cv_q_tot.values      , 'cv_q_tot'      )
        get_env_covar_from_GMV(grid, q, tmp, tmp_O2, UpdVar.Area, UpdVar.θ_liq , UpdVar.q_tot, EnvVar.θ_liq.values , EnvVar.q_tot.values, EnvVar.cv_θ_liq_q_tot.values, GMV.θ_liq.values , GMV.q_tot.values , GMV.cv_θ_liq_q_tot.values, 'cv_θ_liq_q_tot')

        compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, EnvVar.tke           , 'tke')
        compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, EnvVar.cv_θ_liq      , 'cv_θ_liq')
        compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, EnvVar.cv_q_tot      , 'cv_q_tot')
        compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, EnvVar.cv_θ_liq_q_tot, 'cv_θ_liq_q_tot')

        compute_tendencies_gm(grid, q_tendencies, q, GMV, UpdMicro, Case, TS, tmp, tri_diag)

        for k in grid.over_elems_real(Center()):
            EnvVar.tke.values[k] = np.fmax(EnvVar.tke.values[k], 0.0)
            EnvVar.cv_θ_liq.values[k] = np.fmax(EnvVar.cv_θ_liq.values[k], 0.0)
            EnvVar.cv_q_tot.values[k] = np.fmax(EnvVar.cv_q_tot.values[k], 0.0)
            EnvVar.cv_θ_liq_q_tot.values[k] = np.fmax(EnvVar.cv_θ_liq_q_tot.values[k], np.sqrt(EnvVar.cv_θ_liq.values[k]*EnvVar.cv_q_tot.values[k]))
        cleanup_covariance(grid, q, GMV, EnvVar, UpdVar)
        self.set_updraft_surface_bc(grid, q, GMV, Case, tmp)

    def update(self, grid, q, q_tendencies, tmp, tmp_O2, GMV, EnvVar, UpdVar, UpdMicro, EnvThermo, UpdThermo, Case, TS, tri_diag):

        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        self.pre_compute_vars(grid, q, q_tendencies, tmp, tmp_O2, GMV, EnvVar, UpdVar, UpdMicro, EnvThermo, UpdThermo, Case, TS, tri_diag)

        assign_new_to_values(grid, q, tmp, UpdVar)

        self.compute_prognostic_updrafts(grid, q, q_tendencies, tmp, GMV, EnvVar, UpdVar, UpdMicro, EnvThermo, UpdThermo, Case, TS)

        update_cv_env(grid, q, q_tendencies, tmp, tmp_O2, EnvVar.tke           , EnvVar, UpdVar, TS, 'tke'           , tri_diag, self.tke_diss_coeff)
        update_cv_env(grid, q, q_tendencies, tmp, tmp_O2, EnvVar.cv_θ_liq      , EnvVar, UpdVar, TS, 'cv_θ_liq'      , tri_diag, self.tke_diss_coeff)
        update_cv_env(grid, q, q_tendencies, tmp, tmp_O2, EnvVar.cv_q_tot      , EnvVar, UpdVar, TS, 'cv_q_tot'      , tri_diag, self.tke_diss_coeff)
        update_cv_env(grid, q, q_tendencies, tmp, tmp_O2, EnvVar.cv_θ_liq_q_tot, EnvVar, UpdVar, TS, 'cv_θ_liq_q_tot', tri_diag, self.tke_diss_coeff)

        update_sol_gm(grid, q, q_tendencies, GMV, TS, tmp, tri_diag)

        for k in grid.over_elems_real(Center()):
            GMV.U.values[k]     = GMV.U.new[k]
            GMV.V.values[k]     = GMV.V.new[k]
            GMV.θ_liq.values[k] = GMV.θ_liq.new[k]
            GMV.q_tot.values[k] = GMV.q_tot.new[k]
            GMV.q_rai.values[k] = GMV.q_rai.new[k]

        apply_bcs(grid, q, GMV, UpdVar, EnvVar)

        return

    def compute_prognostic_updrafts(self, grid, q, q_tendencies, tmp, GMV, EnvVar, UpdVar, UpdMicro, EnvThermo, UpdThermo, Case, TS):
        time_elapsed = 0.0
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        self.dt_upd = np.minimum(TS.dt, 0.5 * grid.dz/np.fmax(np.max(UpdVar.W.values),1e-10))
        while time_elapsed < TS.dt:
            compute_entrainment_detrainment(grid, GMV, EnvVar, UpdVar, Case, tmp, q, self.entr_detr_fp, self.wstar, self.tke_ed_coeff, self.entrainment_factor, self.detrainment_factor)
            EnvThermo.eos_update_SA_mean(grid, q, EnvVar, False, tmp)
            UpdThermo.buoyancy(grid, q, tmp, UpdVar, EnvVar, GMV)
            UpdMicro.compute_sources(grid, q, UpdVar, tmp)
            UpdMicro.update_updraftvars(grid, q, tmp, UpdVar)

            self.solve_updraft_velocity_area(grid, q, q_tendencies, tmp, GMV, UpdVar, TS)
            self.solve_updraft_scalars(grid, q, q_tendencies, tmp, GMV, EnvVar, UpdVar, UpdMicro, TS)
            UpdVar.θ_liq.set_bcs(grid)
            UpdVar.q_tot.set_bcs(grid)
            UpdVar.q_rai.set_bcs(grid)
            q['w', i_env].apply_bc(grid, 0.0)
            EnvVar.θ_liq.set_bcs(grid)
            EnvVar.q_tot.set_bcs(grid)
            UpdVar.assign_values_to_new(grid, q, tmp)
            time_elapsed += self.dt_upd
            self.dt_upd = np.minimum(TS.dt-time_elapsed,  0.5 * grid.dz/np.fmax(np.max(UpdVar.W.values),1e-10))
            decompose_environment(grid, q, GMV, EnvVar, UpdVar)
        EnvThermo.eos_update_SA_mean(grid, q, EnvVar, True, tmp)
        UpdThermo.buoyancy(grid, q, tmp, UpdVar, EnvVar, GMV)
        return

    def solve_updraft_velocity_area(self, grid, q, q_tendencies, tmp, GMV, UpdVar, TS):
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

                a_k = UpdVar.Area.values[i][k]
                α_0_kp = tmp['α_0_half'][k]
                w_k = UpdVar.W.values[i].Mid(k)

                w_cut = UpdVar.W.values[i].DualCut(k)
                a_cut = UpdVar.Area.values[i].Cut(k)
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
                UpdVar.Area.new[i][k] = np.fmin(np.fmax(a_predict, 0.0), au_lim)

                unsteady = (UpdVar.Area.new[i][k]-a_k)*dti_
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
            UpdVar.Area.new[i][k_1] = self.area_surface_bc[i]


        for k in grid.over_elems(Center()):
            for i in i_uds:
                q['a', i][k] = UpdVar.Area.new[i][k]
            q['a', i_env][k] = 1.0 - sum([UpdVar.Area.new[i][k] for i in i_uds])

        # Solve for updraft velocity
        for i in i_uds:
            UpdVar.W.new[i][kb_1] = self.w_surface_bc[i]
            for k in grid.over_elems_real(Center()):
                a_new_k = UpdVar.Area.new[i].Mid(k)
                if a_new_k >= self.minimum_area:

                    ρ_k = tmp['ρ_0'][k]
                    w_i = UpdVar.W.values[i][k]
                    w_env = q['w', i_env].values[k]
                    a_k = UpdVar.Area.values[i].Mid(k)
                    entr_w = tmp['entr_sc', i].Mid(k)
                    detr_w = tmp['detr_sc', i].Mid(k)
                    B_k = UpdVar.B.values[i].Mid(k)

                    a_cut = UpdVar.Area.values[i].DualCut(k)
                    ρ_cut = tmp['ρ_0'].Cut(k)
                    w_cut = UpdVar.W.values[i].Cut(k)

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

                    UpdVar.W.new[i][k] = ρaw_k/ρa_new_k + dt_/ρa_new_k*(adv + exch + buoy + nh_press)

        # Filter results
        for i in i_uds:
            for k in grid.over_elems_real(Center()):
                if UpdVar.Area.new[i].Mid(k) >= self.minimum_area:
                    if UpdVar.W.new[i][k] <= 0.0:
                        UpdVar.W.new[i][k:] = 0.0
                        UpdVar.Area.new[i][k+1:] = 0.0
                        break
                else:
                    UpdVar.W.new[i][k:] = 0.0
                    UpdVar.Area.new[i][k+1:] = 0.0
                    break

        return

    def solve_updraft_scalars(self, grid, q, q_tendencies, tmp, GMV, EnvVar, UpdVar, UpdMicro, TS):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        dzi = grid.dzi
        dti_ = 1.0/self.dt_upd
        k_1 = grid.first_interior(Zmin())

        for i in i_uds:
            UpdVar.θ_liq.new[i][k_1] = self.θ_liq_surface_bc[i]
            UpdVar.q_tot.new[i][k_1] = self.q_tot_surface_bc[i]

            for k in grid.over_elems_real(Center())[1:]:
                dt_ = 1.0/dti_
                θ_liq_env = EnvVar.θ_liq.values[k]
                q_tot_env = EnvVar.q_tot.values[k]

                if UpdVar.Area.new[i][k] >= self.minimum_area:
                    a_k = UpdVar.Area.values[i][k]
                    a_cut = UpdVar.Area.values[i].Cut(k)
                    a_k_new = UpdVar.Area.new[i][k]
                    θ_liq_cut = UpdVar.θ_liq.values[i].Cut(k)
                    q_tot_cut = UpdVar.q_tot.values[i].Cut(k)
                    ρ_k = tmp['ρ_0_half'][k]
                    ρ_cut = tmp['ρ_0_half'].Cut(k)
                    w_cut = UpdVar.W.values[i].DualCut(k)
                    ε_sc = tmp['entr_sc', i][k]
                    δ_sc = tmp['detr_sc', i][k]
                    ρa_k = ρ_k*a_k

                    ρaw_cut = ρ_cut * a_cut * w_cut
                    ρawθ_liq_cut = ρaw_cut * θ_liq_cut
                    ρawq_tot_cut = ρaw_cut * q_tot_cut
                    ρa_new_k = ρ_k * a_k_new

                    tendencies_θ_liq = -advect(ρawθ_liq_cut, w_cut, grid) + ρaw_cut[1] * (ε_sc * θ_liq_env - δ_sc * θ_liq_cut[1])
                    tendencies_q_tot = -advect(ρawq_tot_cut, w_cut, grid) + ρaw_cut[1] * (ε_sc * q_tot_env - δ_sc * q_tot_cut[1])

                    UpdVar.θ_liq.new[i][k] = ρa_k/ρa_new_k * θ_liq_cut[1] + dt_*tendencies_θ_liq/ρa_new_k
                    UpdVar.q_tot.new[i][k] = ρa_k/ρa_new_k * q_tot_cut[1] + dt_*tendencies_q_tot/ρa_new_k
                else:
                    UpdVar.θ_liq.new[i][k] = GMV.θ_liq.values[k]
                    UpdVar.q_tot.new[i][k] = GMV.q_tot.values[k]

        if self.use_local_micro:
            for i in i_uds:
                for k in grid.over_elems_real(Center()):
                    T, q_liq = eos(tmp['p_0_half'][k],
                                UpdVar.q_tot.new[i][k],
                                UpdVar.θ_liq.new[i][k])
                    UpdVar.T.new[i][k], UpdVar.q_liq.new[i][k] = T, q_liq
                    UpdMicro.compute_update_combined_local_thetal(tmp['p_0_half'], UpdVar.T.new,
                                                                  UpdVar.q_tot.new, UpdVar.q_liq.new,
                                                                  UpdVar.q_rai.new, UpdVar.θ_liq.new,
                                                                  i, k)
                UpdVar.q_rai.new[i][k_1] = 0.0

        return

