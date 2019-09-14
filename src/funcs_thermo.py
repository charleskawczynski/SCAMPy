import numpy as np
from RootSolvers import *
from parameters import *
from PlanetParameters import *
from MoistThermodynamics import *

def qv_star_c(p_0, q_tot, p_vap):
    return eps_v * (1.0 - q_tot) * p_vap / (p_0 - p_vap)

def pv_star(T):
    #    Magnus formula
    T_C = T - 273.15
    return 6.1094*np.exp((17.625*T_C)/float(T_C+243.04))*100

def qv_star_t(p_0, T):
    p_vap = pv_star(T)
    return eps_v * p_vap / (p_0 + (eps_v-1.0)*p_vap)

