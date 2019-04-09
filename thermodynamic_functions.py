import numpy as np
from parameters import *

def sd_c(pd, T):
    return sd_tilde + cpd*np.log(T/T_tilde) -Rd*np.log(pd/p_tilde)

def sv_c(pv, T):
    return sv_tilde + cpv*np.log(T/T_tilde) - Rv * np.log(pv/p_tilde)

def sc_c(L, T):
    return -L/T

def exner_c(p0, kappa = kappa):
    return (p0/p_tilde)**kappa

def theta_c(p0, T):
    return T / exner_c(p0)

def thetali_c(p0, T, qt, ql, qi, L):
    # Liquid ice potential temperature consistent with Triopoli and Cotton (1981)
    return theta_c(p0, T) * np.exp(-latent_heat(T)*(ql/(1.0 - qt) + qi/(1.0 -qt))/(T*cpd))

def theta_virt_c( p0, T, qt, ql, qr):
    # Virtual potential temperature, mixing ratios are approximated by specific humidities.
    return theta_c(p0, T) * (1.0 + 0.61 * (qr) - ql);

def pd_c(p0, qt, qv):
    return p0*(1.0-qt)/(1.0 - qt + eps_vi * qv)

def pv_c(p0, qt, qv):
    return p0 * eps_vi * qv /(1.0 - qt + eps_vi * qv)


def density_temperature_c(T, qt, qv):
    return T * (1.0 - qt + eps_vi * qv)

def theta_rho_c(p0, T, qt, qv):
    return density_temperature_c(T,qt,qv)/exner_c(p0)

def cpm_c(qt):
    return (1.0-qt) * cpd + qt * cpv

def thetas_entropy_c(s, qt):
    return T_tilde*np.exp((s-(1.0-qt)*sd_tilde - qt*sv_tilde)/cpm_c(qt))

def thetas_t_c(p0, T, qt, qv, qc, L):
    qd = 1.0 - qt
    pd_ = pd_c(p0,qt,qt-qc)
    pv_ = pv_c(p0,qt,qt-qc)
    cpm_ = cpm_c(qt)
    return T * pow(p_tilde/pd_,qd * Rd/cpm_)*pow(p_tilde/pv_,qt*Rv/cpm_)*np.exp(-L * qc/(cpm_*T))

def entropy_from_thetas_c(thetas, qt):
    return cpm_c(qt) * np.log(thetas/T_tilde) + (1.0 - qt)*sd_tilde + qt * sv_tilde

def buoyancy_c(alpha0, alpha):
    return g * (alpha - alpha0)/alpha0

def qv_star_c(p0, qt, pv):
    return eps_v * (1.0 - qt) * pv / (p0 - pv)

def alpha_c(p0, T,  qt, qv):
    return (Rd * T)/p0 * (1.0 - qt + eps_vi * qv)

def t_to_entropy_c(p0, T,  qt, ql, qi):
    qv = qt - ql - qi
    pv = pv_c(p0, qt, qv)
    pd = pd_c(p0, qt, qv)
    L = latent_heat(T)
    return sd_c(pd,T) * (1.0 - qt) + sv_c(pv,T) * qt + sc_c(L,T)*(ql + qi)

def t_to_thetali_c(p0, T,  qt, ql, qi):
    L = latent_heat(T)
    return thetali_c(p0, T, qt, ql, qi, L)

def pv_star(T):
    #    Magnus formula
    TC = T - 273.15
    return 6.1094*np.exp((17.625*TC)/float(TC+243.04))*100

def qv_star_t(p0, T):
    pv = pv_star(T)
    return eps_v * pv / (p0 + (eps_v-1.0)*pv)

def latent_heat(T):
    TC = T - 273.15
    return (2500.8 - 2.36 * TC + 0.0016 * TC *
            TC - 0.00006 * TC * TC * TC) * 1000.0

def eos_first_guess_thetal(H, pd, pv, qt):
    p0 = pd + pv
    return H * exner_c(p0)

def eos_first_guess_entropy(H, pd, pv, qt):
    qd = 1.0 - qt
    return (T_tilde *np.exp((H - qd*(sd_tilde - Rd *np.log(pd/p_tilde))
                              - qt * (sv_tilde - Rv * np.log(pv/p_tilde)))/((qd*cpd + qt * cpv))))

def eos(t_to_prog, prog_to_t, p0, qt, prog):
    qv = qt
    ql = 0.0
    pv_1 = pv_c(p0,qt,qt )
    pd_1 = p0 - pv_1
    T_1 = prog_to_t(prog, pd_1, pv_1, qt)
    pv_star_1 = pv_star(T_1)
    qv_star_1 = qv_star_c(p0,qt,pv_star_1)

    ql_2=0.0
    # If not saturated
    if(qt <= qv_star_1):
        T = T_1
        ql = 0.0

    else:
        ql_1 = qt - qv_star_1
        prog_1 = t_to_prog(p0, T_1, qt, ql_1, 0.0)
        f_1 = prog - prog_1
        T_2 = T_1 + ql_1 * latent_heat(T_1) /((1.0 - qt)*cpd + qv_star_1 * cpv)
        delta_T  = np.fabs(T_2 - T_1)

        while delta_T > 1.0e-3 or ql_2 < 0.0:
            pv_star_2 = pv_star(T_2)
            qv_star_2 = qv_star_c(p0,qt,pv_star_2)
            pv_2 = pv_c(p0, qt, qv_star_2)
            pd_2 = p0 - pv_2
            ql_2 = qt - qv_star_2
            prog_2 =  t_to_prog(p0,T_2,qt, ql_2, 0.0   )
            f_2 = prog - prog_2
            T_n = T_2 - f_2*(T_2 - T_1)/(f_2 - f_1)
            T_1 = T_2
            T_2 = T_n
            f_1 = f_2
            delta_T  = np.fabs(T_2 - T_1)

        T  = T_2
        ql = ql_2

    return T, ql

