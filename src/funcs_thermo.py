import numpy as np
from RootSolvers import *
from parameters import *
from MoistThermodynamics import *

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

def thetali_c(p_0, T, q_tot, q_liq, q_ice):
    # Liquid ice potential temperature consistent with Triopoli and Cotton (1981)
    q_dry = (1.0 - q_tot)
    return theta_c(p_0, T) * np.exp(-latent_heat(T)*(q_liq/q_dry + q_ice/q_dry)/(T*cpd))

def theta_virt_c( p_0, T, q_tot, q_liq, qr):
    # Virtual potential temperature, mixing ratios are approximated by specific humidities.
    return theta_c(p_0, T) * (1.0 + 0.61 * (qr) - q_liq);

def pd_c(p_0, q_tot, q_vap):
    q_dry = (1.0 - q_tot)
    return p_0*q_dry/(q_dry + eps_vi * q_vap)

def pv_c(p_0, q_tot, q_vap):
    return p_0 * eps_vi * q_vap /(1.0 - q_tot + eps_vi * q_vap)

def density_temperature_c(T, q_tot, q_vap):
    return T * (1.0 - q_tot + eps_vi * q_vap)

def theta_rho_c(p_0, T, q_tot, q_vap):
    return density_temperature_c(T,q_tot,q_vap)/exner_c(p_0)

def cpm_c(q_tot):
    return (1.0-q_tot) * cpd + q_tot * cpv

def buoyancy_c(alpha_0, alpha):
    return g * (alpha - alpha_0)/alpha_0

def qv_star_c(p_0, q_tot, p_vap):
    return eps_v * (1.0 - q_tot) * p_vap / (p_0 - p_vap)

def alpha_c(p_0, T,  q_tot, q_vap):
    return (Rd * T)/p_0 * (1.0 - q_tot + eps_vi * q_vap)

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

def eos(p_0, q_tot, θ_liq, ρ_0):
    ts = LiquidIcePotTempSHumEquil(θ_liq, q_tot, ρ_0, p_0)
    T = air_temperature(ts)
    q_liq = PhasePartition(ts).liq
    return T, q_liq

