import numpy as np

from parameters import *

def r2q(r_, q_tot):
    return r_ * (1. - q_tot)

def q2r(q_, q_tot):
    return q_ / (1. - q_tot)

def rain_source_to_thetal(p_0, T, q_tot, q_liq, q_ice, q_rai):
    θ_liq_ice_old = thetali_c(p_0, T, q_tot, q_liq, q_ice)
    θ_liq_ice_new = thetali_c(p_0, T, q_tot - q_rai, q_liq - q_rai, q_ice)
    return θ_liq_ice_new - θ_liq_ice_old

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

def terminal_velocity(rho, rho_0, q_rai, q_tot):
    r_rai = q2r(q_rai, q_tot)
    return 14.34 * rho_0**0.5 * rho**-0.3654 * r_rai**0.1346
