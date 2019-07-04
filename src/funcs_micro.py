import numpy as np

from funcs_thermo import *
from parameters import *

def r2q(r_, qt):
    return r_ * (1. - qt)

def q2r(q_, qt):
    return q_ / (1. - qt)

def rain_source_to_thetal(p0, T, qt, ql, qi, qr):
    thetali_old = t_to_thetali_c(p0, T, qt, ql, qi)
    thetali_new = t_to_thetali_c(p0, T, qt - qr, ql - qr, qi)
    return thetali_new - thetali_old

# instantly convert all cloud water exceeding a threshold to rain water
# the threshold is specified as excess saturation
# rain water is immediately removed from the domain
# Tiedke:   TODO - add reference
def acnv_instant(ql, qt, sat_treshold, T, p0):
    psat = pv_star(T)
    qsat = qv_star_c(p0, qt, psat)
    return np.fmax(0.0, ql - sat_treshold * qsat)

# time-rate expressions for 1-moment microphysics
# autoconversion:   Kessler 1969, see Table 1 in Wood 2005: https://doi.org/10.1175/JAS3530.1
# accretion, rain evaporation rain terminal velocity:
#    Grabowski and Smolarkiewicz 1996 eqs: 5b-5d
#    https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2

# unfortunately the rate expressions in the paper are for mixing ratios
# need to convert to specific humidities

# TODO - change it to saturation treshold
def acnv_rate(ql, qt):
    rl = q2r(ql, qt)
    return (1. - qt) * 1e-3 * np.fmax(0.0, rl - 5e-4)

def accr_rate(ql, qr, qt):
    rl = q2r(ql, qt)
    rr = q2r(qr, qt)
    return (1. - qt) * 2.2 * rl * rr**0.875

def evap_rate(rho, qv, qr, qt, T, p0):
    psat = pv_star(T)
    qsat = qv_star_c(p0, qt, psat)
    rr   = q2r(qr, qt)
    rv   = q2r(qv, qt)
    rsat = q2r(qsat, qt)
    C = 1.6 + 124.9 * (1e-3 * rho * rr)**0.2046 # ventilation factor
    return (1 - qt) * (1. - rv/rsat) * C * (1e-3 * rho * rr)**0.525 / rho / (540 + 2.55 * 1e5 / (p0 * rsat))
    #      dq/dr     * dr/dt

def terminal_velocity(rho, rho0, qr, qt):
    rr = q2r(qr, qt)
    return 14.34 * rho0**0.5 * rho**-0.3654 * rr**0.1346

def microphysics(T, ql, p0, qt, max_supersat, in_Env):
    _ret = type('', (), {})()
    _ret.T     = T
    _ret.ql    = ql
    _ret.thl   = t_to_thetali_c(p0, T, qt, ql, 0.0)
    _ret.th    = theta_c(p0, T)
    _ret.qv    = qt - ql
    _ret.alpha = alpha_c(p0, T, qt, _ret.qv)
    _ret.qr    = 0.0
    _ret.qt    = qt

    if in_Env:
        _ret.qr           = acnv_instant(ql, qt, max_supersat, T, p0)
        _ret.thl_rain_src = rain_source_to_thetal(p0, T, qt, ql, 0.0, _ret.qr)

        _ret.qt  -= _ret.qr
        _ret.ql  -= _ret.qr
        _ret.thl += _ret.thl_rain_src

    return _ret
