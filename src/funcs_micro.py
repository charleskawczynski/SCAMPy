import numpy as np

from funcs_thermo import *
from parameters import *

def r2q(r_, q_tot):
    return r_ * (1. - q_tot)

def q2r(q_, q_tot):
    return q_ / (1. - q_tot)

def rain_source_to_thetal(p_0, T, q_tot, q_liq, q_ice, q_rai):
    θ_liq_ice_old = thetali_c(p_0, T, q_tot, q_liq, q_ice)
    θ_liq_ice_new = thetali_c(p_0, T, q_tot - q_rai, q_liq - q_rai, q_ice)
    return θ_liq_ice_new - θ_liq_ice_old

# instantly convert all cloud water exceeding a threshold to rain water
# the threshold is specified as excess saturation
# rain water is immediately removed from the domain
# Tiedke:   TODO - add reference
def acnv_instant(q_liq, q_tot, sat_treshold, T, p_0):
    p_sat = pv_star(T)
    q_sat = qv_star_c(p_0, q_tot, p_sat)
    return np.fmax(0.0, q_liq - sat_treshold * q_sat)

# time-rate expressions for 1-moment microphysics
# autoconversion:   Kessler 1969, see Table 1 in Wood 2005: https://doi.org/10.1175/JAS3530.1
# accretion, rain evaporation rain terminal velocity:
#    Grabowski and Smolarkiewicz 1996 eqs: 5b-5d
#    https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2

# unfortunately the rate expressions in the paper are for mixing ratios
# need to convert to specific humidities

# TODO - change it to saturation treshold
def acnv_rate(q_liq, q_tot):
    r_liq = q2r(q_liq, q_tot)
    return (1. - q_tot) * 1e-3 * np.fmax(0.0, r_liq - 5e-4)

def accr_rate(q_liq, q_rai, q_tot):
    r_liq = q2r(q_liq, q_tot)
    r_rai = q2r(q_rai, q_tot)
    return (1. - q_tot) * 2.2 * r_liq * r_rai**0.875

def evap_rate(rho, q_vap, q_rai, q_tot, T, p_0):
    p_sat = pv_star(T)
    q_sat = qv_star_c(p_0, q_tot, p_sat)
    r_rai = q2r(q_rai, q_tot)
    r_vap = q2r(q_vap, q_tot)
    r_sat = q2r(q_sat, q_tot)
    C = 1.6 + 124.9 * (1e-3 * rho * r_rai)**0.2046 # ventilation factor
    return (1 - q_tot) * (1. - r_vap/r_sat) * C * (1e-3 * rho * r_rai)**0.525 / rho / (540 + 2.55 * 1e5 / (p_0 * r_sat))
    #      dq/dr     * dr/dt

def terminal_velocity(rho, rho_0, q_rai, q_tot):
    r_rai = q2r(q_rai, q_tot)
    return 14.34 * rho_0**0.5 * rho**-0.3654 * r_rai**0.1346

def microphysics(T, q_liq, p_0, q_tot, max_supersat, in_Env):
    _ret = type('', (), {})()
    _ret.T     = T
    _ret.q_liq = q_liq
    _ret.θ_liq = thetali_c(p_0, T, q_tot, q_liq, 0.0)
    _ret.θ     = theta_c(p_0, T)
    _ret.q_vap = q_tot - q_liq
    _ret.alpha = alpha_c(p_0, T, q_tot, _ret.q_vap)
    _ret.q_rai = 0.0
    _ret.q_tot = q_tot

    if in_Env:
        _ret.q_rai          = acnv_instant(q_liq, q_tot, max_supersat, T, p_0)
        _ret.θ_liq_rain_src = rain_source_to_thetal(p_0, T, q_tot, q_liq, 0.0, _ret.q_rai)

        _ret.q_tot -= _ret.q_rai
        _ret.q_liq -= _ret.q_rai
        _ret.θ_liq += _ret.θ_liq_rain_src

    return _ret
