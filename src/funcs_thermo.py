import numpy as np
from parameters import *

def sd_c(p_dry, T):
    return sd_tilde + cpd*np.log(T/T_tilde) - Rd * np.log(p_dry/p_tilde)

def sv_c(p_vap, T):
    return sv_tilde + cpv*np.log(T/T_tilde) - Rv * np.log(p_vap/p_tilde)

def sc_c(L, T):
    return -L/T

def exner_c(p_0, kappa = kappa):
    return (p_0/p_tilde)**kappa

def theta_c(p_0, T):
    return T / exner_c(p_0)

def thetali_c(p_0, T, q_tot, q_liq, q_ice, L):
    # Liquid ice potential temperature consistent with Triopoli and Cotton (1981)
    return theta_c(p_0, T) * np.exp(-latent_heat(T)*(q_liq/(1.0 - q_tot) + q_ice/(1.0 -q_tot))/(T*cpd))

def theta_virt_c( p_0, T, q_tot, q_liq, qr):
    # Virtual potential temperature, mixing ratios are approximated by specific humidities.
    return theta_c(p_0, T) * (1.0 + 0.61 * (qr) - q_liq);

def pd_c(p_0, q_tot, q_vap):
    return p_0*(1.0-q_tot)/(1.0 - q_tot + eps_vi * q_vap)

def pv_c(p_0, q_tot, q_vap):
    return p_0 * eps_vi * q_vap /(1.0 - q_tot + eps_vi * q_vap)

def density_temperature_c(T, q_tot, q_vap):
    return T * (1.0 - q_tot + eps_vi * q_vap)

def theta_rho_c(p_0, T, q_tot, q_vap):
    return density_temperature_c(T,q_tot,q_vap)/exner_c(p_0)

def cpm_c(q_tot):
    return (1.0-q_tot) * cpd + q_tot * cpv

def thetas_entropy_c(s, q_tot):
    return T_tilde*np.exp((s-(1.0-q_tot)*sd_tilde - q_tot*sv_tilde)/cpm_c(q_tot))

def thetas_t_c(p_0, T, q_tot, q_vap, qc, L):
    q_dry = 1.0 - q_tot
    p_dry = pd_c(p_0,q_tot,q_tot-qc)
    p_vap = pv_c(p_0,q_tot,q_tot-qc)
    c_pm = cpm_c(q_tot)
    return T * pow(p_tilde/p_dry,q_dry * Rd/c_pm)*pow(p_tilde/p_vap,q_tot*Rv/c_pm)*np.exp(-L * qc/(c_pm*T))

def entropy_from_thetas_c(theta_s, q_tot):
    return cpm_c(q_tot) * np.log(theta_s/T_tilde) + (1.0 - q_tot)*sd_tilde + q_tot * sv_tilde

def buoyancy_c(alpha_0, alpha):
    return g * (alpha - alpha_0)/alpha_0

def qv_star_c(p_0, q_tot, p_vap):
    return eps_v * (1.0 - q_tot) * p_vap / (p_0 - p_vap)

def alpha_c(p_0, T,  q_tot, q_vap):
    return (Rd * T)/p_0 * (1.0 - q_tot + eps_vi * q_vap)

def t_to_entropy_c(p_0, T,  q_tot, q_liq, q_ice):
    q_vap = q_tot - q_liq - q_ice
    p_vap = pv_c(p_0, q_tot, q_vap)
    p_dry = pd_c(p_0, q_tot, q_vap)
    L = latent_heat(T)
    return sd_c(p_dry,T) * (1.0 - q_tot) + sv_c(p_vap,T) * q_tot + sc_c(L,T)*(q_liq + q_ice)

def t_to_thetali_c(p_0, T,  q_tot, q_liq, q_ice):
    L = latent_heat(T)
    return thetali_c(p_0, T, q_tot, q_liq, q_ice, L)

def pv_star(T):
    #    Magnus formula
    T_C = T - 273.15
    return 6.1094*np.exp((17.625*T_C)/float(T_C+243.04))*100

def qv_star_t(p_0, T):
    p_vap = pv_star(T)
    return eps_v * p_vap / (p_0 + (eps_v-1.0)*p_vap)

def latent_heat(T):
    T_C = T - 273.15
    return (2500.8 - 2.36 * T_C + 0.0016 * T_C *
            T_C - 0.00006 * T_C * T_C * T_C) * 1000.0

def eos_first_guess_thetal(H, p_dry, p_vap, q_tot):
    p_0 = p_dry + p_vap
    return H * exner_c(p_0)

def eos_first_guess_entropy(H, p_dry, p_vap, q_tot):
    q_dry = 1.0 - q_tot
    return (T_tilde *np.exp((H - q_dry * (sd_tilde - Rd * np.log(p_dry/p_tilde))
                               - q_tot * (sv_tilde - Rv * np.log(p_vap/p_tilde)))/((q_dry*cpd + q_tot * cpv))))

def eos(p_0, q_tot, prog):
    q_vap = q_tot
    q_liq = 0.0
    pv_1 = pv_c(p_0, q_tot, q_tot)
    pd_1 = p_0 - pv_1
    T_1 = eos_first_guess_thetal(prog, pd_1, pv_1, q_tot)
    pv_star_1 = pv_star(T_1)
    qv_star_1 = qv_star_c(p_0,q_tot,pv_star_1)
    ql_2=0.0
    # If not saturated
    if(q_tot <= qv_star_1):
        T = T_1
        q_liq = 0.0
    else:
        ql_1 = q_tot - qv_star_1
        prog_1 = t_to_thetali_c(p_0, T_1, q_tot, ql_1, 0.0)
        f_1 = prog - prog_1
        T_2 = T_1 + ql_1 * latent_heat(T_1) /((1.0 - q_tot)*cpd + qv_star_1 * cpv)
        delta_T  = np.fabs(T_2 - T_1)

        while delta_T > 1.0e-3 or ql_2 < 0.0:
            pv_star_2 = pv_star(T_2)
            qv_star_2 = qv_star_c(p_0,q_tot,pv_star_2)
            pv_2 = pv_c(p_0, q_tot, qv_star_2)
            pd_2 = p_0 - pv_2
            ql_2 = q_tot - qv_star_2
            prog_2 =  t_to_thetali_c(p_0,T_2,q_tot, ql_2, 0.0   )
            f_2 = prog - prog_2
            T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1)
            T_1 = T_2
            T_2 = T_n
            f_1 = f_2
            delta_T  = np.fabs(T_2 - T_1)

        T  = T_2
        q_liq = ql_2

    return T, q_liq

def eos_entropy(p_0, q_tot, prog):
    q_vap = q_tot
    q_liq = 0.0
    pv_1 = pv_c(p_0, q_tot, q_tot)
    pd_1 = p_0 - pv_1
    T_1 = eos_first_guess_entropy(prog, pd_1, pv_1, q_tot)
    pv_star_1 = pv_star(T_1)
    qv_star_1 = qv_star_c(p_0,q_tot,pv_star_1)
    ql_2=0.0
    # If not saturated
    if(q_tot <= qv_star_1):
        T = T_1
        q_liq = 0.0
    else:
        ql_1 = q_tot - qv_star_1
        prog_1 = t_to_entropy_c(p_0, T_1, q_tot, ql_1, 0.0)
        f_1 = prog - prog_1
        T_2 = T_1 + ql_1 * latent_heat(T_1) /((1.0 - q_tot)*cpd + qv_star_1 * cpv)
        delta_T  = np.fabs(T_2 - T_1)

        while delta_T > 1.0e-3 or ql_2 < 0.0:
            pv_star_2 = pv_star(T_2)
            qv_star_2 = qv_star_c(p_0,q_tot,pv_star_2)
            pv_2 = pv_c(p_0, q_tot, qv_star_2)
            pd_2 = p_0 - pv_2
            ql_2 = q_tot - qv_star_2
            prog_2 =  t_to_entropy_c(p_0,T_2,q_tot, ql_2, 0.0   )
            f_2 = prog - prog_2
            T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1)
            T_1 = T_2
            T_2 = T_n
            f_1 = f_2
            delta_T  = np.fabs(T_2 - T_1)

        T  = T_2
        q_liq = ql_2

    return T, q_liq

