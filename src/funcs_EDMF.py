import sys
import pylab as plt
from NetCDFIO import NetCDFIO_Stats
import numpy as np
from parameters import *
from TriDiagSolver import *
from Operators import advect, grad, Laplacian, grad_pos, grad_neg
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann, nice_name
from TimeStepping import TimeStepping
from funcs_thermo import  *
from funcs_turbulence import  *
from funcs_micro import *

def compute_limiters(grid, tmp, q, q_tendencies, name, names_limiter, Δt, bounds_field):
    for k in grid.over_elems_real(q.data_location(name)):
        for i in q.over_sub_domains(name):
            tmp[names_limiter[0], i][k] = (bounds_field[0] - q[name, i][k])/Δt - q_tendencies[name, i][k]
            tmp[names_limiter[1], i][k] = (bounds_field[1] - q[name, i][k])/Δt - q_tendencies[name, i][k]

def add_tendency(grid, q_tendencies, tmp, name_eq, name_tendency):
    for k in grid.over_elems_real(q_tendencies.data_location(name_eq)):
        for i in q_tendencies.over_sub_domains(name_eq):
            q_tendencies[name_eq, i][k] += tmp[name_tendency, i][k]

def compute_src_limited(grid, tmp, name_src_limited, src_name, names_limiter_min, names_limiter_max):
    for k in grid.over_elems_real(tmp.data_location(src_name)):
        for i in tmp.over_sub_domains(src_name):
            for name_min, name_max in zip(names_limiter_min, names_limiter_max):
                tmp[name_src_limited, i][k] = max(min(tmp[src_name, i][k], tmp[name_max, i][k]), tmp[name_min, i][k])

def solve_updraft_velocity_area(grid, q_new, q, q_tendencies, tmp, UpdVar, TS, params):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    k_1 = grid.first_interior(Zmin())
    kb_1 = grid.boundary(Zmin())
    dzi = grid.dzi

    # Solve for area fraction
    for i in i_uds:
        au_lim = UpdVar[i].area_surface_bc * params.max_area_factor
        for k in grid.over_elems_real(Center()):

            a_k = q['a', i][k]
            α_0_kp = tmp['α_0'][k]
            w_k = q['w_half', i][k]

            w_cut = q['w_half', i].Cut(k)
            a_cut = q['a', i].Cut(k)
            ρ_cut = tmp['ρ_0'].Cut(k)
            tendencies = 0.0

            ρaw_cut = ρ_cut*a_cut*w_cut
            adv = - α_0_kp * advect(ρaw_cut, w_cut, grid)
            tendencies+=adv

            ε_term = a_k * w_k * (+ tmp['entr_sc', i][k])
            tendencies+=ε_term
            δ_term = a_k * w_k * (- tmp['detr_sc', i][k])
            tendencies+=δ_term

            a_predict = a_k + TS.Δt_up * tendencies

            q_new['a', i][k] = np.fmin(np.fmax(a_predict, 0.0), au_lim)

        tmp['entr_sc', i][k_1] = 2.0 * dzi
        tmp['detr_sc', i][k_1] = 0.0
        q_new['a', i][k_1] = UpdVar[i].area_surface_bc

    # Solve for updraft velocity
    for i in i_uds:
        q_new['a', i][kb_1] = UpdVar[i].w_surface_bc
        for k in grid.over_elems_real(Center()):
            a_new_k = q_new['a', i][k]
            if a_new_k >= params.minimum_area:

                w_env = q['w_half', i_env][k]
                ρ_k = tmp['ρ_0'][k]
                w_i = q['w_half', i][k]
                a_k = q['a', i][k]
                entr_w = tmp['entr_sc', i][k]
                detr_w = tmp['detr_sc', i][k]
                B_k = tmp['B', i][k]

                a_cut = q['a', i].Cut(k)
                ρ_cut = tmp['ρ_0'].Cut(k)
                w_cut = q['w_half', i].Cut(k)

                ρa_k = ρ_k * a_k
                ρa_new_k = ρ_k * a_new_k
                ρaw_k = ρa_k * w_i
                ρaww_cut = ρ_cut*a_cut*w_cut*w_cut

                adv = -advect(ρaww_cut, w_cut, grid)
                exch = ρaw_k * (- detr_w * w_i + entr_w * w_env)
                buoy = ρa_k * B_k
                press_buoy = - ρa_k * B_k * params.pressure_buoy_coeff
                p_coeff = params.pressure_drag_coeff/params.pressure_plume_spacing
                press_drag = - ρa_k * (p_coeff * (w_i - w_env)**2.0/np.sqrt(np.fmax(a_k, params.minimum_area)))
                nh_press = press_buoy + press_drag

                q_new['w_half', i][k] = ρaw_k/ρa_new_k + TS.Δt_up/ρa_new_k*(adv + exch + buoy + nh_press)

    # Filter results
    for i in i_uds:
        for k in grid.over_elems_real(Center()):
            if q_new['a', i][k] >= params.minimum_area:
                if q_new['w_half', i][k] <= 0.0:
                    q_new['w_half', i][k:] = 0.0
                    q_new['a', i][k:] = 0.0
                    break
            else:
                q_new['w_half', i][k:] = 0.0
                q_new['a', i][k:] = 0.0
                break

    return

def solve_updraft_scalars(grid, q_new, q, q_tendencies, tmp, UpdVar, TS, params):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    dzi = grid.dzi
    k_1 = grid.first_interior(Zmin())

    for i in i_uds:
        q_new['θ_liq', i][k_1] = UpdVar[i].θ_liq_surface_bc
        q_new['q_tot', i][k_1] = UpdVar[i].q_tot_surface_bc

        for k in grid.over_elems_real(Center())[1:]:
            θ_liq_env = q['θ_liq', i_env][k]
            q_tot_env = q['q_tot', i_env][k]

            if q_new['a', i][k] >= params.minimum_area:
                a_k = q['a', i][k]
                a_cut = q['a', i].Cut(k)
                a_k_new = q_new['a', i][k]
                θ_liq_cut = q['θ_liq', i].Cut(k)
                q_tot_cut = q['q_tot', i].Cut(k)
                ρ_k = tmp['ρ_0'][k]
                ρ_cut = tmp['ρ_0'].Cut(k)
                w_cut = q['w_half', i].Cut(k)
                ε_sc = tmp['entr_sc', i][k]
                δ_sc = tmp['detr_sc', i][k]
                ρa_k = ρ_k*a_k

                ρaw_cut = ρ_cut * a_cut * w_cut
                ρawθ_liq_cut = ρaw_cut * θ_liq_cut
                ρawq_tot_cut = ρaw_cut * q_tot_cut
                ρa_new_k = ρ_k * a_k_new

                tendencies_θ_liq = 0.0
                tendencies_q_tot = 0.0

                tendencies_θ_liq += -advect(ρawθ_liq_cut, w_cut, grid)
                tendencies_q_tot += -advect(ρawq_tot_cut, w_cut, grid)

                tendencies_θ_liq += ρaw_cut[1] * (ε_sc * θ_liq_env - δ_sc * θ_liq_cut[1])
                tendencies_q_tot += ρaw_cut[1] * (ε_sc * q_tot_env - δ_sc * q_tot_cut[1])

                q_new['θ_liq', i][k] = ρa_k/ρa_new_k * θ_liq_cut[1] + TS.Δt_up*tendencies_θ_liq/ρa_new_k
                q_new['q_tot', i][k] = ρa_k/ρa_new_k * q_tot_cut[1] + TS.Δt_up*tendencies_q_tot/ρa_new_k
            else:
                q_new['θ_liq', i][k] = q['θ_liq', i_gm][k]
                q_new['q_tot', i][k] = q['q_tot', i_gm][k]
    return

def eos_update_SA_mean(grid, q, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems_real(Center()):
        p_0 = tmp['p_0'][k]
        θ_liq = q['θ_liq', i_env][k]
        q_tot = q['q_tot', i_env][k]
        T, q_liq  = eos(p_0, q_tot, θ_liq)
        tmp['T', i_env][k]      = T
        tmp['q_liq', i_env][k]  = q_liq
        q_vap = q_tot - q_liq
        alpha = alpha_c(p_0, T, q_tot, q_vap)
        tmp['B', i_env][k]   = buoyancy_c(tmp['α_0'][k], alpha)
        θ = theta_c(p_0, T)
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

def satadjust(grid, q, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems(Center()):
        θ_liq = q['θ_liq', i_gm][k]
        q_tot = q['q_tot', i_gm][k]
        p_0 = tmp['p_0'][k]
        T, q_liq = eos(p_0, q_tot, θ_liq)
        tmp['q_liq', i_gm][k] = q_liq
        tmp['T', i_gm][k] = T
        q_vap = q_tot - q_liq
        alpha = alpha_c(p_0, T, q_tot, q_vap)
        tmp['B', i_gm][k] = buoyancy_c(tmp['α_0'][k], alpha)
    return

def predict(grid, tmp, q, q_tendencies, name, name_predict, Δt):
    for k in grid.over_elems_real(q.data_location(name)):
        for i in q.over_sub_domains(name):
            tmp[name_predict, i][k] = q[name, i][k] + Δt*q_tendencies[name, i][k]

def residual(grid, tmp, q_new, q, q_tendencies, name, name_res, Δt):
    for k in grid.over_elems_real(q.data_location(name)):
        for i in q.over_sub_domains(name):
            tmp[name_res, i][k] = (q_new[name, i][k] - q[name, i][k])/Δt - q_tendencies[name, i][k]

def compute_covariance_dissipation(grid, q, tmp, tmp_O2, tke_diss_coeff, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]
    for k in grid.over_elems_real(Center()):
        l_mix = np.fmax(tmp['l_mix'][k], 1.0)
        tke_env = np.fmax(q['tke', i_env][k], 0.0)
        tmp_O2[cv]['dissipation'][k] = (tmp['ρ_0'][k] * ae[k] * q[cv, i_env][k] * pow(tke_env, 0.5)/l_mix * tke_diss_coeff)
    return

def reset_surface_covariance(grid, q, tmp, Case, wstar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    flux1 = Case.Sur.rho_θ_liq_flux
    flux2 = Case.Sur.rho_q_tot_flux
    k_1 = grid.first_interior(Zmin())
    zLL = grid.z_half[k_1]
    alpha0LL  = tmp['α_0'][k_1]
    ustar = Case.Sur.ustar
    oblength = Case.Sur.obukhov_length
    q['tke', i_gm][k_1]            = surface_tke(Case.Sur.ustar, wstar, zLL, Case.Sur.obukhov_length)
    return

def update_sol_gm(grid, q_new, q, q_tendencies, TS, tmp, tri_diag):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ρ_0_half = tmp['ρ_0']
    ae = q['a', i_env]
    slice_real_n = grid.slice_real(Node())
    slice_all_c = grid.slice_all(Center())

    tri_diag.ρaK[slice_real_n] = [ae.Mid(k)*tmp['K_h'].Mid(k)*ρ_0_half.Mid(k) for k in grid.over_elems_real(Node())]
    construct_tridiag_diffusion_O1(grid, TS.Δt, tri_diag, ρ_0_half, ae)
    tri_diag.f[slice_all_c] = [q['q_tot', i_gm][k] + TS.Δt*q_tendencies['q_tot', i_gm][k] for k in grid.over_elems(Center())]
    solve_tridiag_wrapper(grid, q_new['q_tot', i_gm], tri_diag)
    tri_diag.f[slice_all_c] = [q['θ_liq', i_gm][k] + TS.Δt*q_tendencies['θ_liq', i_gm][k] for k in grid.over_elems(Center())]
    solve_tridiag_wrapper(grid, q_new['θ_liq', i_gm], tri_diag)
    return

def compute_inversion(grid, q, option, tmp, Ri_bulk_crit, temp_C):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    maxgrad = 0.0
    domain = grid.over_elems_real(Center())
    theta_rho_bl = temp_C.first_interior(grid)
    for k in domain:
        q_tot = q['q_tot', i_gm][k]
        q_vap = q_tot - tmp['q_liq', i_gm][k]
        temp_C[k] = theta_rho_c(tmp['p_0'][k], tmp['T', i_gm][k], q_tot, q_vap)
    if option == 'theta_rho':
        for k in domain:
            if temp_C[k] > theta_rho_bl:
                zi = grid.z_half[k]
                break
    elif option == 'thetal_maxgrad':
        for k in domain:
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
    for k in grid.over_elems_real(Center()):
        tmp['l_mix'][k] = 100.0
    return

def compute_eddy_diffusivities_tke(grid, q, tmp, Case, zi, wstar, prandtl_number, tke_ed_coeff, similarity_diffusivity):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    compute_mixing_length(grid, q, tmp, Case.Sur.obukhov_length, zi, wstar)
    if similarity_diffusivity:
        ustar = Case.Sur.ustar
        for k in grid.over_elems_real(Center()):
            zzi = grid.z_half[k]/zi
            tmp['K_h'][k] = 0.0
            tmp['K_m'][k] = 0.0
            if zzi <= 1.0 and not (wstar<1e-6):
                tmp['K_h'][k] = vkb * ( (ustar/wstar)**3.0 + 39.0*vkb*zzi)**(1.0/3.0) * zzi * (1.0-zzi) * (1.0-zzi) * wstar * zi
                tmp['K_m'][k] = tmp['K_h'][k] * prandtl_number
    else:
        for k in grid.over_elems_real(Center()):
            lm = tmp['l_mix'][k]
            K_m_k = tke_ed_coeff * lm * np.sqrt(np.fmax(q['tke', i_env][k],0.0) )
            tmp['K_m'][k] = K_m_k
            tmp['K_h'][k] = K_m_k / prandtl_number
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
        p_0 = tmp['p_0'][k]

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
        tmp_O2[cv]['buoy'][k] = g / tmp['α_0'][k] * ae[k] * tmp['ρ_0'][k] * (term_1 + term_2)
    return

def apply_gm_bcs(grid, q):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    q['θ_liq', i_gm].apply_bc(grid, 0.0)
    q['q_tot', i_gm].apply_bc(grid, 0.0)
    q['tke', i_gm].apply_bc(grid, 0.0)

def cleanup_covariance(grid, q):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    tmp_eps = 1e-18
    slice_real_c = grid.slice_real(Center())
    domain = grid.over_elems_real(Center())
    q['tke', i_gm][slice_real_c] = [0.0 if q['tke', i_gm][k] < tmp_eps else q['tke', i_gm][k] for k in domain]
    q['tke', i_env][slice_real_c] = [0.0 if q['tke', i_env][k] < tmp_eps else q['tke', i_env][k] for k in domain]

def compute_grid_means(grid, q, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]
    for k in grid.over_elems_real(Center()):
        tmp['q_liq', i_gm][k] = np.sum([ q['a', i][k] * tmp['q_liq', i][k] for i in i_sd])
        tmp['T', i_gm][k]     = np.sum([ q['a', i][k] * tmp['T', i][k] for i in i_sd])
        tmp['B', i_gm][k]     = np.sum([ q['a', i][k] * tmp['B', i][k] for i in i_sd])
    return

def compute_cv_gm(grid, q, ϕ, ψ, cv, tke_factor, interp_func):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]
    for k in grid.over_elems(Center()):
        Δϕ = interp_func(q[ϕ, i_env], k) - interp_func(q[ϕ, i_gm], k)
        Δψ = interp_func(q[ψ, i_env], k) - interp_func(q[ψ, i_gm], k)
        q[cv, i_gm][k] = tke_factor * ae[k] * Δϕ * Δψ + ae[k] * q[cv, i_env][k]
        for i in i_uds:
            Δϕ = interp_func(q[ϕ, i], k) - interp_func(q[ϕ, i_gm], k)
            Δψ = interp_func(q[ψ, i], k) - interp_func(q[ψ, i_gm], k)
            q[cv, i_gm][k] += tke_factor * q['a', i][k] * Δϕ * Δψ
    return

def compute_covariance_entr(grid, q, tmp, tmp_O2, ϕ, ψ, cv, tke_factor, interp_func):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems_real(Center()):
        tmp_O2[cv]['entr_gain'][k] = 0.0
        for i in i_uds:
            Δϕ = interp_func(q[ϕ, i], k) - interp_func(q[ϕ, i_env], k)
            Δψ = interp_func(q[ψ, i], k) - interp_func(q[ψ, i_env], k)
            tmp_O2[cv]['entr_gain'][k] += tke_factor*q['a', i][k] * np.fabs(q['w_half', i][k]) * tmp['detr_sc', i][k] * Δϕ * Δψ
        tmp_O2[cv]['entr_gain'][k] *= tmp['ρ_0'][k]
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
        ρaK = tmp['ρ_0'][k] * ae[k] * tmp['K_h'][k]
        tmp_O2[cv]['shear'][k] = tke_factor*2.0*ρaK * (grad_ϕ*grad_ψ + grad_u**2.0 + grad_v**2.0)
    return

def compute_covariance_interdomain_src(grid, q, tmp, tmp_O2, ϕ, ψ, cv, tke_factor, interp_func):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems(Center()):
        tmp_O2[cv]['interdomain'][k] = 0.0
        for i in i_uds:
            Δϕ = interp_func(q[ϕ, i], k) - interp_func(q[ϕ, i_env], k)
            Δψ = interp_func(q[ψ, i], k) - interp_func(q[ψ, i_env], k)
            tmp_O2[cv]['interdomain'][k] += tke_factor*q['a', i][k] * (1.0-q['a', i][k]) * Δϕ * Δψ
    return

def compute_covariance_detr(grid, q, tmp, tmp_O2, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems_real(Center()):
        tmp_O2[cv]['detr_loss'][k] = sum([q['a', i][k] * np.fabs(q['w_half', i][k]) * tmp['entr_sc', i][k] for i in i_uds])
        tmp_O2[cv]['detr_loss'][k] *= tmp['ρ_0'][k] * q[cv, i_env][k]
    return

def compute_tke_pressure(grid, q, tmp, tmp_O2, pressure_buoy_coeff, pressure_drag_coeff, pressure_plume_spacing, cv):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems_real(Center()):
        tmp_O2[cv]['press'][k] = 0.0
        for i in i_uds:
            wu_half = q['w_half', i][k]
            we_half = q['w_half', i_env][k]
            a_i = q['a', i][k]
            ρ_0_k = tmp['ρ_0'][k]
            press_buoy = (-1.0 * ρ_0_k * a_i * tmp['B', i][k] * pressure_buoy_coeff)
            press_drag_coeff = -1.0 * ρ_0_k * np.sqrt(a_i) * pressure_drag_coeff/pressure_plume_spacing
            press_drag = press_drag_coeff * (wu_half - we_half)*np.fabs(wu_half - we_half)
            tmp_O2[cv]['press'][k] += (we_half - wu_half) * (press_buoy + press_drag)
    return

def compute_cv_env(grid, q, tmp, tmp_O2, ϕ, ψ, cv, tke_factor, interp_func):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    ae = q['a', i_env]
    for k in grid.over_elems(Center()):
        if ae[k] > 0.0:
            q[cv, i_env][k] = q[cv, i_gm][k]
            for i in i_sd:
                Δϕ = interp_func(q[ϕ, i], k) - interp_func(q[ϕ, i_gm], k)
                Δψ = interp_func(q[ψ, i], k) - interp_func(q[ψ, i_gm], k)
                q[cv, i_env][k] -= tke_factor * q['a', i][k] * Δϕ * Δψ
            q[cv, i_env][k] = q[cv, i_env][k]/ae[k]
        else:
            q[cv, i_env][k] = 0.0
    return

def diagnose_environment(grid, q):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems(Center()):
        a_env = q['a', i_env][k]
        q['q_tot', i_env][k] = (q['q_tot', i_gm][k] - np.sum([q['a', i][k]*q['q_tot', i][k] for i in i_uds]))/a_env
        q['θ_liq', i_env][k] = (q['θ_liq', i_gm][k] - np.sum([q['a', i][k]*q['θ_liq', i][k] for i in i_uds]))/a_env
        # Assuming w_gm = 0!
        q['w_half', i_env][k] = (0.0 - np.sum([q['a', i][k]*q['w_half', i][k] for i in i_uds]))/a_env
    return

def compute_tendencies_gm(grid, q_tendencies, q, Case, TS, tmp, tri_diag):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    k_1 = grid.first_interior(Zmin())
    dzi = grid.dzi
    α_1 = tmp['α_0'][k_1]
    ae_1 = q['a', i_env][k_1]
    slice_all_c = grid.slice_all(Center())

    q_tendencies['q_tot', i_gm][slice_all_c] += [tmp['mf_tend_q_tot'][k] for k in grid.over_elems(Center())]
    q_tendencies['q_tot', i_gm][k_1] += Case.Sur.rho_q_tot_flux * dzi * α_1/ae_1

    q_tendencies['θ_liq', i_gm][slice_all_c] += [tmp['mf_tend_θ_liq'][k] for k in grid.over_elems(Center())]
    q_tendencies['θ_liq', i_gm][k_1] += Case.Sur.rho_θ_liq_flux * dzi * α_1/ae_1
    return

def update_cv_env(grid, q, q_tendencies, tmp, tmp_O2, TS, cv, tri_diag, tke_diss_coeff):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    construct_tridiag_diffusion_O2(grid, q, tmp, TS, tri_diag, tke_diss_coeff)
    k_1 = grid.first_interior(Zmin())

    slice_all_c = grid.slice_all(Center())
    a_e = q['a', i_env]
    tri_diag.f[slice_all_c] = [tmp['ρ_0'][k] * a_e[k] * q[cv, i_env][k] * TS.Δti + q_tendencies[cv, i_env][k] for k in grid.over_elems(Center())]
    tri_diag.f[k_1] = tmp['ρ_0'][k_1] * a_e[k_1] * q[cv, i_env][k_1] * TS.Δti + q[cv, i_env][k_1]
    solve_tridiag_wrapper(grid, q[cv, i_env], tri_diag)

    return

def update_GMV_MF(grid, q, TS, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    slice_all_c = grid.slice_real(Center())
    domain_c = grid.over_elems_real(Center())

    for i in i_uds:
        tmp['mf_tmp', i][slice_all_c] = [((q['w_half', i][k] - q['w_half', i_env][k]) * tmp['ρ_0'][k]
                       * q['a', i][k]) for k in domain_c]

    for k in domain_c:
        tmp['mf_θ_liq'][k] = np.sum([tmp['mf_tmp', i][k] * (q['θ_liq', i][k] - q['θ_liq', i_env][k]) for i in i_uds])
        tmp['mf_q_tot'][k] = np.sum([tmp['mf_tmp', i][k] * (q['q_tot', i][k] - q['q_tot', i_env][k]) for i in i_uds])

    tmp['mf_tend_θ_liq'][slice_all_c] = [-tmp['α_0'][k]*grad(tmp['mf_θ_liq'].Cut(k), grid) for k in domain_c]
    tmp['mf_tend_q_tot'][slice_all_c] = [-tmp['α_0'][k]*grad(tmp['mf_q_tot'].Cut(k), grid) for k in domain_c]
    return

def assign_new_to_values(grid, q_new, q, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    slice_all_c = grid.slice_all(Center())
    for i in i_uds:
        q_new['w_half', i][slice_all_c] = [q['w_half', i][k] for k in grid.over_elems(Center())]
        q_new['q_tot', i][slice_all_c] = [q['q_tot', i][k] for k in grid.over_elems(Center())]
        q_new['θ_liq', i][slice_all_c] = [q['θ_liq', i][k] for k in grid.over_elems(Center())]
    return

def assign_values_to_new(grid, q, q_new, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems(Center()):
        for i in i_uds:
            q['w_half', i][k] = q_new['w_half', i][k]
            q['q_tot', i][k] = q_new['q_tot', i][k]
            q['θ_liq', i][k] = q_new['θ_liq', i][k]
            q['a', i][k] = q_new['a', i][k]
        q['a', i_env][k] = 1.0 - np.sum([q_new['a', i][k] for i in i_uds])
    return

def initialize_updrafts(grid, tmp, q, updraft_fraction):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    k_1 = grid.first_interior(Zmin())
    n_updrafts = len(i_uds)
    for i in i_uds:
        for k in grid.over_elems(Center()):
            q['w_half', i][k] = 0.0
            q['a', i][k] = 0.0
            q['q_tot', i][k] = q['q_tot', i_gm][k]
            tmp['q_liq', i][k] = tmp['q_liq', i_gm][k]
            q['θ_liq', i][k] = q['θ_liq', i_gm][k]
            tmp['T', i][k] = tmp['T', i_gm][k]
            tmp['B', i][k] = 0.0
        q['a', i][k_1] = updraft_fraction/n_updrafts
    for i in i_uds: q['q_tot', i].apply_bc(grid, 0.0)
    for i in i_uds: q['θ_liq', i].apply_bc(grid, 0.0)
    for k in grid.over_elems(Center()):
        q['a', i_env][k] = 1.0 - np.sum([q['a', i][k] for i in i_uds])
    return

def buoyancy(grid, q, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for i in i_uds:
        for k in grid.over_elems_real(Center()):
            if q['a', i][k] > 1e-3:
                q_tot = q['q_tot', i][k]
                q_vap = q_tot - tmp['q_liq', i][k]
                T = tmp['T', i][k]
                α_i = alpha_c(tmp['p_0'][k], T, q_tot, q_vap)
                tmp['B', i][k] = buoyancy_c(tmp['α_0'][k], α_i)
            else:
                tmp['B', i][k] = tmp['B', i_env][k]
    # Subtract grid mean buoyancy
    for k in grid.over_elems_real(Center()):
        tmp['B', i_gm][k] = np.sum([q['a', i][k] * tmp['B', i][k] for i in i_sd])
        for i in i_sd:
            tmp['B', i][k] -= tmp['B', i_gm][k]
    return

def pre_export_data_compute(grid, q, tmp, tmp_O2, Stats, tke_diss_coeff):
    compute_covariance_dissipation(grid, q, tmp, tmp_O2, tke_diss_coeff, 'tke')
    compute_covariance_detr(grid, q, tmp, tmp_O2, 'tke')
