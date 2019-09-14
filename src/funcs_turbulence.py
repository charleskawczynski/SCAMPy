import numpy as np
import copy
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid
from parameters import *

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
def compute_convective_velocity(bflux, zi):
    return np.cbrt(np.fmax(bflux * zi, 0.0))

# BL height
def compute_inversion_height(theta_rho, u, v, grid, Ri_bulk_crit):
    theta_rho_b = theta_rho.first_interior(grid)
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
def compute_mixing_tau(zi, wstar):
    # return 0.5 * zi / wstar
    #return zi / (np.fmax(wstar, 1e-5))
    return zi / (wstar + 0.001)

# MO scaling of near surface tke and scalar variance
def surface_tke(ustar, wstar, zLL, oblength):
    if oblength < 0.0:
        return ((3.75 + np.cbrt(zLL/oblength * zLL/oblength)) * ustar * ustar + 0.2 * wstar * wstar)
    else:
        return (3.75 * ustar * ustar)

def surface_variance(flux1, flux2, ustar, zLL, oblength):
    c_star1 = -flux1/ustar
    c_star2 = -flux2/ustar
    if oblength < 0.0:
        return 4.0 * c_star1 * c_star2 * pow(1.0 - 8.3 * zLL/oblength, -2.0/3.0)
    else:
        return 4.0 * c_star1 * c_star2

def set_cloudbase_flag(q_liq, current_flag):
    if q_liq > 1.0e-8:
        new_flag = True
    else:
        new_flag = current_flag
    return  new_flag

