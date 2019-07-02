import numpy as np
import copy
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from parameters import *
from thermodynamic_functions import *

# Entrainment Rates
def entr_detr_dry(entr_in):
    eps = 1.0 # to avoid division by zero when z = 0 or z_i
    # Following Soares 2004
    _ret = type('', (), {})()
    _ret.entr_sc = 0.5*(1.0/entr_in.z + 1.0/np.fmax(entr_in.zi - entr_in.z, 10.0)) #vkb/(z + 1.0e-3)
    _ret.detr_sc = 0.0
    return  _ret

def entr_detr_inverse_z(entr_in):
    _ret = type('', (), {})()
    _ret.entr_sc = vkb/entr_in.z
    _ret.detr_sc= 0.0
    return _ret

def entr_detr_inverse_w(entr_in):
    eps_w = 1.0/(np.fmax(np.fabs(entr_in.w),1.0)* 500)
    _ret = type('', (), {})()
    if entr_in.af>0.0:
        partiation_func  = entr_detr_buoyancy_sorting(entr_in)
        _ret.entr_sc = partiation_func*eps_w/2.0
        _ret.detr_sc = (1.0-partiation_func/2.0)*eps_w
    else:
        _ret.entr_sc = 0.0
        _ret.detr_sc = 0.0
    return _ret

def entr_detr_buoyancy_sorting(entr_in):
        sqpi_inv = 1.0/np.sqrt(pi)
        sqrt2 = np.sqrt(2.0)
        partiation_func = 0.0
        inner_partiation_func = 0.0
        abscissas, weights = np.polynomial.hermite.hermgauss(entr_in.quadrature_order)

        if entr_in.env_QTvar != 0.0 and entr_in.env_Hvar != 0.0:
            sd_q = np.sqrt(entr_in.env_QTvar)
            sd_h = np.sqrt(entr_in.env_Hvar)
            corr = np.fmax(np.fmin(entr_in.env_HQTcov/np.fmax(sd_h*sd_q, 1e-13),1.0),-1.0)

            # limit sd_q to prevent negative qt_hat
            sd_q_lim = (1e-10 - entr_in.qt_env)/(sqrt2 * abscissas[0])
            sd_q = np.fmin(sd_q, sd_q_lim)
            sigma_h_star = np.sqrt(np.fmax(1.0-corr*corr,0.0)) * sd_h

            for m_q in range(entr_in.quadrature_order):
                qt_hat    = (entr_in.qt_env + sqrt2 * sd_q * abscissas[m_q] + entr_in.qt_up)/2.0
                mu_h_star = entr_in.H_env + sqrt2 * corr * sd_h * abscissas[m_q]
                inner_partiation_func = 0.0
                for m_h in range(entr_in.quadrature_order):
                    h_hat = (sqrt2 * sigma_h_star * abscissas[m_h] + mu_h_star + entr_in.H_up)/2.0
                    # condensation
                    T, ql  = eos(entr_in.p0, qt_hat, h_hat)
                    # calcualte buoyancy
                    qv_ = qt_hat - ql
                    alpha_mix = alpha_c(entr_in.p0, T, qt_hat, qv_)
                    bmix = buoyancy_c(entr_in.alpha0, alpha_mix) - entr_in.b_mean

                    # sum only the points with positive buoyancy to get the buoyant fraction
                    if bmix >0.0:
                        inner_partiation_func  += weights[m_h] * sqpi_inv
                partiation_func  += inner_partiation_func * weights[m_q] * sqpi_inv
        else:
            h_hat = ( entr_in.H_env + entr_in.H_up)/2.0
            qt_hat = ( entr_in.qt_env + entr_in.qt_up)/2.0

            # condensation
            T, ql  = eos(entr_in.p0, qt_hat, h_hat)
            # calcualte buoyancy
            alpha_mix = alpha_c(entr_in.p0, T, qt_hat, qt_hat - ql)
            bmix = buoyancy_c(entr_in.alpha0, alpha_mix) - entr_in.b_mean

        return partiation_func

def entr_detr_tke2(entr_in):
    # in cloud portion from Soares 2004
    if entr_in.z >= entr_in.zi :
        _ret.detr_sc= 3.0e-3
    else:
        _ret.detr_sc = 0.0

    _ret.entr_sc = (0.05 * np.sqrt(entr_in.tke) / np.fmax(entr_in.w, 0.01) / np.fmax(entr_in.af, 0.001) / np.fmax(entr_in.z, 1.0))
    return  _ret

# yair - this is a new entr-detr function that takes entr as proportional to tke/w and detr ~ b/w2
def entr_detr_tke(entr_in):
    _ret.detr_sc = np.fabs(entr_in.b)/ np.fmax(entr_in.w * entr_in.w, 1e-3)
    _ret.entr_sc = np.sqrt(entr_in.tke) / np.fmax(entr_in.w, 0.01) / np.fmax(np.sqrt(entr_in.af), 0.001) / 50000.0
    return  _ret


def entr_detr_b_w2(entr_in):
    # in cloud portion from Soares 2004
    _ret = type('', (), {})()
    if entr_in.z >= entr_in.zi:
        _ret.detr_sc= 4.0e-3 + 0.12 *np.fabs(np.fmin(entr_in.b,0.0)) / np.fmax(entr_in.w * entr_in.w, 1e-2)
    else:
        _ret.detr_sc = 0.0
    _ret.entr_sc = 0.12 * np.fmax(entr_in.b,0.0) / np.fmax(entr_in.w * entr_in.w, 1e-2)
    return  _ret

def entr_detr_suselj(entr_in):
    entr_dry = 2.5e-3
    l0 = (entr_in.zbl - entr_in.zi)/10.0
    if entr_in.z >= entr_in.zi :
        _ret.detr_sc= 4.0e-3 +  0.12* np.fabs(np.fmin(entr_in.b,0.0)) / np.fmax(entr_in.w * entr_in.w, 1e-2)
        _ret.entr_sc = 0.1 / entr_in.dz * entr_in.poisson
    else:
        _ret.detr_sc = 0.0
        _ret.entr_sc = 0.0 #entr_dry # Very low entrainment rate needed for Dycoms to work
    return  _ret

def entr_detr_none(entr_in):
    _ret.entr_sc = 0.0
    _ret.detr_sc = 0.0
    return  _ret

# convective velocity scale
def get_wstar(bflux, zi):
    return np.cbrt(np.fmax(bflux * zi, 0.0))

# BL height
def get_inversion(theta_rho, u, v, grid, Ri_bulk_crit):
    theta_rho_b = theta_rho.surface_bl(grid)
    h = 0.0
    Ri_bulk=0.0
    Ri_bulk_low = 0.0
    kmin = grid.first_interior(Zmin())
    k = kmin
    z = grid.z_half
    # test if we need to look at the free convective limit
    if (u[kmin] * u[kmin] + v[kmin] * v[kmin]) <= 0.01:
        for k in grid.over_elems_real(Center()):
            if theta_rho[k] > theta_rho_b:
                break
        h = (z[k] - z[k-1])/(theta_rho[k] - theta_rho[k-1]) * (theta_rho_b - theta_rho[k-1]) + z[k-1]
    else:
        for k in grid.over_elems_real(Center()):
            Ri_bulk_low = Ri_bulk
            Ri_bulk = g * (theta_rho[k] - theta_rho_b) * z[k]/theta_rho_b / (u[k] * u[k] + v[k] * v[k])
            if Ri_bulk > Ri_bulk_crit:
                break
        h = (z[k] - z[k-1])/(Ri_bulk - Ri_bulk_low) * (Ri_bulk_crit - Ri_bulk_low) + z[k-1]

    return h

# Teixiera convective tau
def get_mixing_tau(zi, wstar):
    # return 0.5 * zi / wstar
    #return zi / (np.fmax(wstar, 1e-5))
    return zi / (wstar + 0.001)

# MO scaling of near surface tke and scalar variance
def get_surface_tke(ustar, wstar, zLL, oblength):
    if oblength < 0.0:
        return ((3.75 + np.cbrt(zLL/oblength * zLL/oblength)) * ustar * ustar + 0.2 * wstar * wstar)
    else:
        return (3.75 * ustar * ustar)

def get_surface_variance(flux1, flux2, ustar, zLL, oblength):
    c_star1 = -flux1/ustar
    c_star2 = -flux2/ustar
    if oblength < 0.0:
        return 4.0 * c_star1 * c_star2 * pow(1.0 - 8.3 * zLL/oblength, -2.0/3.0)
    else:
        return 4.0 * c_star1 * c_star2

def set_cloudbase_flag(ql, current_flag):
    if ql > 1.0e-8:
        new_flag = True
    else:
        new_flag = current_flag
    return  new_flag

