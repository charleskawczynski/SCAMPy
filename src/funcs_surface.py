import numpy as np
from funcs_thermo import latent_heat, pd_c, pv_c, sd_c, sv_c, cpm_c, theta_rho_c
from parameters import *

def buoyancy_flux(shf, lhf, T_b, qt_b, alpha0_0):
    cp_ = cpm_c(qt_b)
    lv = latent_heat(T_b)
    return (g * alpha0_0 / cp_ / T_b * (shf + (eps_vi-1.0) * cp_ * T_b * lhf /lv))

def psi_m_unstable(zeta, zeta0):
    x = (1.0 - gamma_m * zeta)**0.25
    x0 = (1.0 - gamma_m * zeta0)**0.25
    psi_m = (2.0 * np.log((1.0 + x)/(1.0 + x0)) + np.log((1.0 + x*x)/(1.0 + x0 * x0))
                         -2.0 * np.arctan(x) + 2.0 * np.arctan(x0))
    return psi_m

def psi_h_unstable(zeta, zeta0):
    y = np.sqrt(1.0 - gamma_h * zeta )
    y0 = np.sqrt(1.0 - gamma_h * zeta0 )
    psi_h = 2.0 * np.log((1.0 + y)/(1.0 + y0))
    return psi_h

def psi_m_stable(zeta, zeta0):
    psi_m = -beta_m * (zeta - zeta0)
    return  psi_m

def psi_h_stable(zeta, zeta0):
    psi_h = -beta_h * (zeta - zeta0)
    return  psi_h

def entropy_flux(tflux,qtflux, p0_1, T_1, qt_1):
        cp_1 = cpm_c(qt_1)
        pd_1 = pd_c(p0_1, qt_1, qt_1)
        pv_1 = pv_c(p0_1, qt_1, qt_1)
        sd_1 = sd_c(pd_1, T_1)
        sv_1 = sv_c(pv_1, T_1)
        return cp_1*tflux/T_1 + qtflux*(sv_1-sd_1)

def compute_ustar(windspeed, buoyancy_flux, z0, z1) :
    logz = np.log(z1 / z0)
    #use neutral condition as first guess
    ustar0 = windspeed * vkb / logz
    ustar = ustar0
    if (np.abs(buoyancy_flux) > 1.0e-20):
        lmo = -ustar0 * ustar0 * ustar0 / (buoyancy_flux * vkb)
        zeta = z1 / lmo
        zeta0 = z0 / lmo
        if (zeta >= 0.0):
            f0 = windspeed - ustar0 / vkb * (logz - psi_m_stable(zeta, zeta0))
            ustar1 = windspeed * vkb / (logz - psi_m_stable(zeta, zeta0))
            lmo = -ustar1 * ustar1 * ustar1 / (buoyancy_flux * vkb)
            zeta = z1 / lmo
            zeta0 = z0 / lmo
            f1 = windspeed - ustar1 / vkb * (logz - psi_m_stable(zeta, zeta0))
            ustar = ustar1
            delta_ustar = ustar1 -ustar0
            while np.abs(delta_ustar) > 1e-3:
                ustar_new = ustar1 - f1 * delta_ustar / (f1-f0)
                f0 = f1
                ustar0 = ustar1
                ustar1 = ustar_new
                lmo = -ustar1 * ustar1 * ustar1 / (buoyancy_flux * vkb)
                zeta = z1 / lmo
                zeta0 = z0 / lmo
                f1 = windspeed - ustar1 / vkb * (logz - psi_m_stable(zeta, zeta0))
                delta_ustar = ustar1 -ustar0
        else: # b_flux nonzero, zeta  is negative
            f0 = windspeed - ustar0 / vkb * (logz - psi_m_unstable(zeta, zeta0))
            ustar1 = windspeed * vkb / (logz - psi_m_unstable(zeta, zeta0))
            lmo = -ustar1 * ustar1 * ustar1 / (buoyancy_flux * vkb)
            zeta = z1 / lmo
            zeta0 = z0 / lmo
            f1 = windspeed - ustar1 / vkb * (logz - psi_m_unstable(zeta, zeta0))
            ustar = ustar1
            delta_ustar = ustar1 - ustar0
            while np.abs(delta_ustar) > 1e-3:
                ustar_new = ustar1 - f1 * delta_ustar / (f1 - f0)
                f0 = f1
                ustar0 = ustar1
                ustar1 = ustar_new
                lmo = -ustar1 * ustar1 * ustar1 / (buoyancy_flux * vkb)
                zeta = z1 / lmo
                zeta0 = z0 / lmo
                f1 = windspeed - ustar1 / vkb * (logz - psi_m_unstable(zeta, zeta0))
                delta_ustar = ustar1 - ustar
    return ustar

def exchange_coefficients_byun(Ri, zb, z0):
    logz = np.log(zb/z0)
    zfactor = zb/(zb-z0)*logz
    sb = Ri/Pr0
    if Ri > 0.0:
        zeta = zfactor/(2.0*beta_h*(beta_m*Ri -1.0))*((1.0-2.0*beta_h*Ri)-np.sqrt(1.0+4.0*(beta_h - beta_m)*sb))
        lmo = zb/zeta
        zeta0 = z0/lmo
        psi_m = psi_m_stable(zeta, zeta0)
        psi_h = psi_h_stable(zeta,zeta0)
    else:
        qb = 1.0/9.0 * (1.0 /(gamma_m * gamma_m) + 3.0 * gamma_h/gamma_m * sb * sb)
        pb = 1.0/54.0 * (-2.0/(gamma_m*gamma_m*gamma_m) + 9.0/gamma_m * (-gamma_h/gamma_m + 3.0)*sb * sb)
        crit = qb * qb *qb - pb * pb
        if crit < 0.0:
            tb = np.cbrt(np.sqrt(-crit) + np.fabs(pb))
            zeta = zfactor * (1.0/(3.0*gamma_m)-(tb + qb/tb))
        else:
            angle = np.arccos(pb/np.sqrt(qb * qb * qb))
            zeta = zfactor * (-2.0 * np.sqrt(qb) * np.cos(angle/3.0)+1.0/(3.0*gamma_m))
        lmo = zb/zeta
        zeta0 = z0/lmo
        psi_m = psi_m_unstable(zeta, zeta0)
        psi_h = psi_h_unstable(zeta,zeta0)

    cu = vkb/(logz-psi_m)
    cth = vkb/(logz-psi_h)/Pr0
    cm = cu * cu
    ch = cu * cth
    return cm, ch, lmo
