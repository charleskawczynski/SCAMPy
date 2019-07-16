import sys
import pylab as plt
from NetCDFIO import NetCDFIO_Stats
import numpy as np
from parameters import *
from TriDiagSolver import solve_tridiag_wrapper, construct_tridiag_diffusion_O1, construct_tridiag_diffusion_O2
from Operators import advect, grad, Laplacian, grad_pos, grad_neg
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann, nice_name
from TimeStepping import TimeStepping
from funcs_thermo import  *
from funcs_turbulence import  *
from funcs_micro import *

def update_env(q, tmp, k, T, θ_liq, q_tot, q_liq, q_rai, alpha):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    tmp['T', i_env][k]      = T
    q['θ_liq', i_env][k]  = θ_liq
    q['q_tot', i_env][k]  = q_tot
    tmp['q_liq', i_env][k]  = q_liq
    q['q_rai', i_env][k] += q_rai
    tmp['B', i_env][k]   = buoyancy_c(tmp['α_0_half'][k], alpha)
    return

def update_cloud_dry(k, T, θ, q_tot, q_liq, q_vap, tmp):
    i_gm, i_env, i_uds, i_sd = tmp.domain_idx()
    if q_liq > 0.0:
        tmp['CF'][k] = 1.
        tmp['θ_cloudy'][k]     = θ
        tmp['t_cloudy'][k]     = T
        tmp['q_tot_cloudy'][k] = q_tot
        tmp['q_vap_cloudy'][k] = q_vap
    else:
        tmp['CF'][k] = 0.
        tmp['θ_dry'][k]     = θ
        tmp['q_tot_dry'][k] = q_tot
    return

def eos_update_SA_mean(grid, q, in_Env, tmp, max_supersaturation):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems_real(Center()):
        p_0_k = tmp['p_0_half'][k]
        T, q_liq  = eos(p_0_k, q['q_tot', i_env][k], q['θ_liq', i_env][k])
        mph = microphysics(T, q_liq, p_0_k, q['q_tot', i_env][k], max_supersaturation, in_Env)
        update_env(q, tmp, k, mph.T, mph.θ_liq, mph.q_tot, mph.q_liq, mph.q_rai, mph.alpha)
        update_cloud_dry(k, mph.T, mph.θ,  mph.q_tot, mph.q_liq, mph.q_vap, tmp)
    return

def satadjust(grid, q, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems(Center()):
        θ_liq = q['θ_liq', i_gm][k]
        q_tot = q['q_tot', i_gm][k]
        p_0 = tmp['p_0_half'][k]
        T, q_liq = eos(p_0, q_tot, θ_liq)
        tmp['q_liq', i_gm][k] = q_liq
        tmp['T', i_gm][k] = T
        q_vap = q_tot - q_liq
        alpha = alpha_c(p_0, T, q_tot, q_vap)
        tmp['B', i_gm][k] = buoyancy_c(tmp['α_0_half'][k], alpha)
    return

def predict(grid, tmp, q, q_tendencies, name, name_predict, Δt):
    for k in grid.over_elems_real(Center()):
        for i in q.over_sub_domains(name):
            tmp[name_predict, i][k] = q[name, i][k] + Δt*q_tendencies[name, i][k]

def compute_covariance_rain(grid, q, tmp, tmp_O2, TS, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]
    for k in grid.over_elems_real(Center()):
        ρa_0 = tmp['ρ_0_half'][k]*ae[k]
        if cv=='tke':            tmp_O2[cv]['rain_src'][k] = 0.0
        if cv=='cv_θ_liq':       tmp_O2[cv]['rain_src'][k]       = ρa_0 * 2. * tmp['cv_θ_liq_rain_dt'][k]       * TS.dti
        if cv=='cv_q_tot':       tmp_O2[cv]['rain_src'][k]       = ρa_0 * 2. * tmp['cv_q_tot_rain_dt'][k]       * TS.dti
        if cv=='cv_θ_liq_q_tot': tmp_O2[cv]['rain_src'][k] = ρa_0 *      tmp['cv_θ_liq_q_tot_rain_dt'][k] * TS.dti
    return


def compute_covariance_dissipation(grid, q, tmp, tmp_O2, tke_diss_coeff, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]
    for k in grid.over_elems_real(Center()):
        l_mix = np.fmax(tmp['l_mix'][k], 1.0)
        tke_env = np.fmax(q['tke', i_env][k], 0.0)

        tmp_O2[cv]['dissipation'][k] = (tmp['ρ_0_half'][k] * ae[k] * q[cv, i_env][k] * pow(tke_env, 0.5)/l_mix * tke_diss_coeff)
    return

def reset_surface_covariance(grid, q, tmp, Case, wstar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    flux1 = Case.Sur.rho_θ_liq_flux
    flux2 = Case.Sur.rho_q_tot_flux
    k_1 = grid.first_interior(Zmin())
    zLL = grid.z_half[k_1]
    alpha0LL  = tmp['α_0_half'][k_1]
    ustar = Case.Sur.ustar
    oblength = Case.Sur.obukhov_length
    q['tke', i_gm][k_1]            = surface_tke(Case.Sur.ustar, wstar, zLL, Case.Sur.obukhov_length)
    q['cv_θ_liq', i_gm][k_1]       = surface_variance(flux1*alpha0LL,flux1*alpha0LL, ustar, zLL, oblength)
    q['cv_q_tot', i_gm][k_1]       = surface_variance(flux2*alpha0LL,flux2*alpha0LL, ustar, zLL, oblength)
    q['cv_θ_liq_q_tot', i_gm][k_1] = surface_variance(flux1*alpha0LL,flux2*alpha0LL, ustar, zLL, oblength)
    return

def update_sol_gm(grid, q_new, q, q_tendencies, TS, tmp, tri_diag):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ρ_0_half = tmp['ρ_0_half']
    ae = q['a', i_env]
    slice_real_n = grid.slice_real(Node())
    slice_all_c = grid.slice_all(Center())

    tri_diag.ρaK[slice_real_n] = [ae.Mid(k)*tmp['K_h'].Mid(k)*ρ_0_half.Mid(k) for k in grid.over_elems_real(Node())]
    construct_tridiag_diffusion_O1(grid, TS.dt, tri_diag, ρ_0_half, ae)
    tri_diag.f[slice_all_c] = [q['q_tot', i_gm][k] + TS.dt*q_tendencies['q_tot', i_gm][k] for k in grid.over_elems(Center())]
    solve_tridiag_wrapper(grid, q_new['q_tot', i_gm], tri_diag)
    tri_diag.f[slice_all_c] = [q['θ_liq', i_gm][k] + TS.dt*q_tendencies['θ_liq', i_gm][k] for k in grid.over_elems(Center())]
    solve_tridiag_wrapper(grid, q_new['θ_liq', i_gm], tri_diag)

    tri_diag.ρaK[slice_real_n] = [ae.Mid(k)*tmp['K_m'].Mid(k)*ρ_0_half.Mid(k) for k in grid.over_elems_real(Node())]
    construct_tridiag_diffusion_O1(grid, TS.dt, tri_diag, ρ_0_half, ae)
    tri_diag.f[slice_all_c] = [q['U', i_gm][k] + TS.dt*q_tendencies['U', i_gm][k] for k in grid.over_elems(Center())]
    solve_tridiag_wrapper(grid, q_new['U', i_gm], tri_diag)
    tri_diag.f[slice_all_c] = [q['V', i_gm][k] + TS.dt*q_tendencies['V', i_gm][k] for k in grid.over_elems(Center())]
    solve_tridiag_wrapper(grid, q_new['V', i_gm], tri_diag)
    return

def compute_zbl_qt_grad(grid, q):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    # computes inversion height as z with max gradient of q_tot
    zbl_q_tot = 0.0
    q_tot_grad = 0.0
    for k in grid.over_elems_real(Center()):
        q_tot_grad_new = grad(q['q_tot', i_gm].Dual(k), grid)
        if np.fabs(q_tot_grad) > q_tot_grad:
            q_tot_grad = np.fabs(q_tot_grad_new)
            zbl_q_tot = grid.z_half[k]
    return zbl_q_tot

def compute_inversion(grid, q, option, tmp, Ri_bulk_crit, temp_C):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    maxgrad = 0.0
    theta_rho_bl = temp_C.first_interior(grid)
    for k in grid.over_elems_real(Center()):
        q_tot = q['q_tot', i_gm][k]
        q_vap = q_tot - tmp['q_liq', i_gm][k]
        temp_C[k] = theta_rho_c(tmp['p_0_half'][k], tmp['T', i_gm][k], q_tot, q_vap)
    if option == 'theta_rho':
        for k in grid.over_elems_real(Center()):
            if temp_C[k] > theta_rho_bl:
                zi = grid.z_half[k]
                break
    elif option == 'thetal_maxgrad':
        for k in grid.over_elems_real(Center()):
            grad_TH = grad(q['θ_liq', i_gm].Dual(k), grid)
            if grad_TH > maxgrad:
                maxgrad = grad_TH
                zi = grid.z[k]
    elif option == 'critical_Ri':
        zi = compute_inversion_height(temp_C, q['U', i_gm], q['V', i_gm], grid, Ri_bulk_crit)
    else:
        print('INVERSION HEIGHT OPTION NOT RECOGNIZED')
    return zi

def compute_mixing_length(grid, q, tmp, obukhov_length, zi, wstar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    tau = compute_mixing_tau(zi, wstar)
    for k in grid.over_elems_real(Center()):
        l1 = tau * np.sqrt(np.fmax(q['tke', i_env][k],0.0))
        z_ = grid.z_half[k]
        if obukhov_length < 0.0: #unstable
            l2 = vkb * z_ * ( (1.0 - 100.0 * z_/obukhov_length)**0.2 )
        elif obukhov_length > 0.0: #stable
            l2 = vkb * z_ /  (1. + 2.7 *z_/obukhov_length)
        else:
            l2 = vkb * z_
        tmp['l_mix'][k] = np.fmax( 1.0/(1.0/np.fmax(l1,1e-10) + 1.0/l2), 1e-3)
    return


def compute_eddy_diffusivities_tke(grid, q, tmp, Case, zi, wstar, prandtl_number, tke_ed_coeff, similarity_diffusivity):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    compute_mixing_length(grid, q, tmp, Case.Sur.obukhov_length, zi, wstar)
    if similarity_diffusivity:
        compute_eddy_diffusivities_similarity_Siebesma2007(grid, Case, tmp, zi, wstar, prandtl_number)
    else:
        for k in grid.over_elems_real(Center()):
            lm = tmp['l_mix'][k]
            K_m_k = tke_ed_coeff * lm * np.sqrt(np.fmax(q['tke', i_env][k],0.0) )
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

def compute_cv_env_tendencies(grid, q_tendencies, tmp_O2, cv):
    i_gm, i_env, i_uds, i_sd = q_tendencies.domain_idx()
    k_1 = grid.first_interior(Zmin())

    for k in grid.over_elems_real(Center()):
        q_tendencies[cv, i_env][k] = tmp_O2[cv]['press'][k] + tmp_O2[cv]['buoy'][k] + tmp_O2[cv]['shear'][k] + tmp_O2[cv]['entr_gain'][k] + tmp_O2[cv]['rain_src'][k]
    q_tendencies[cv, i_env][k_1] = 0.0
    return

def compute_tke_buoy(grid, q, tmp, tmp_O2, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]

    # Note that source terms at the first interior point are not really used because that is where tke boundary condition is
    # enforced (according to MO similarity). Thus here I am being sloppy about lowest grid point
    for k in grid.over_elems_real(Center()):
        q_tot_dry = tmp['q_tot_dry'][k]
        θ_dry = tmp['θ_dry'][k]
        t_cloudy = tmp['t_cloudy'][k]
        q_vap_cloudy = tmp['q_vap_cloudy'][k]
        q_tot_cloudy = tmp['q_tot_cloudy'][k]
        θ_cloudy = tmp['θ_cloudy'][k]
        p_0 = tmp['p_0_half'][k]

        lh = latent_heat(t_cloudy)
        cpm = cpm_c(q_tot_cloudy)
        grad_θ_liq = grad_neg(q['θ_liq', i_env].Cut(k), grid)
        grad_q_tot = grad_neg(q['q_tot', i_env].Cut(k), grid)

        prefactor = Rd * exner_c(p_0)/p_0

        d_alpha_θ_liq_dry = prefactor * (1.0 + (eps_vi - 1.0) * q_tot_dry)
        d_alpha_q_tot_dry = prefactor * θ_dry * (eps_vi - 1.0)
        CF_env = tmp['CF'][k]

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
        tmp_O2[cv]['buoy'][k] = g / tmp['α_0_half'][k] * ae[k] * tmp['ρ_0_half'][k] * (term_1 + term_2)
    return

def apply_gm_bcs(grid, q):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    q['U', i_gm].apply_bc(grid, 0.0)
    q['V', i_gm].apply_bc(grid, 0.0)
    q['θ_liq', i_gm].apply_bc(grid, 0.0)
    q['q_tot', i_gm].apply_bc(grid, 0.0)
    q['q_rai', i_gm].apply_bc(grid, 0.0)
    q['tke', i_gm].apply_bc(grid, 0.0)
    q['cv_q_tot', i_gm].apply_bc(grid, 0.0)
    q['cv_θ_liq', i_gm].apply_bc(grid, 0.0)
    q['cv_θ_liq_q_tot', i_gm].apply_bc(grid, 0.0)

def cleanup_covariance(grid, q):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    tmp_eps = 1e-18
    slice_real_c = grid.slice_real(Center())
    q['tke',            i_gm][slice_real_c] = [0.0 if q['tke', i_gm][k] < tmp_eps else q['tke', i_gm][k] for k in grid.over_elems_real(Center())]
    q['cv_θ_liq',       i_gm][slice_real_c] = [0.0 if q['cv_θ_liq', i_gm][k] < tmp_eps else q['cv_θ_liq', i_gm][k] for k in grid.over_elems_real(Center())]
    q['cv_q_tot',       i_gm][slice_real_c] = [0.0 if q['cv_q_tot', i_gm][k] < tmp_eps else q['cv_q_tot', i_gm][k] for k in grid.over_elems_real(Center())]
    q['cv_θ_liq_q_tot', i_gm][slice_real_c] = [0.0 if np.fabs(q['cv_θ_liq_q_tot', i_gm][k]) < tmp_eps else q['cv_θ_liq_q_tot', i_gm][k] for k in grid.over_elems_real(Center())]
    q['cv_θ_liq',       i_env][slice_real_c] = [0.0 if q['cv_θ_liq', i_env][k] < tmp_eps else q['cv_θ_liq', i_env][k] for k in grid.over_elems_real(Center())]
    q['tke',            i_env][slice_real_c] = [0.0 if q['tke', i_env][k] < tmp_eps else q['tke', i_env][k] for k in grid.over_elems_real(Center())]
    q['cv_q_tot',       i_env][slice_real_c] = [0.0 if q['cv_q_tot', i_env][k] < tmp_eps else q['cv_q_tot', i_env][k] for k in grid.over_elems_real(Center())]
    q['cv_θ_liq_q_tot', i_env][slice_real_c] = [0.0 if np.fabs(q['cv_θ_liq_q_tot', i_env][k]) < tmp_eps else q['cv_θ_liq_q_tot', i_env][k] for k in grid.over_elems_real(Center())]
