import numpy as np
from RootSolvers import *
from parameters import *
from PlanetParameters import *
from MoistThermodynamics import *

def qv_star_t(p_0, T):
    p_vap = saturation_vapor_pressure_raw(T, Liquid())
    return eps_v * p_vap / (p_0 + (eps_v-1.0)*p_vap)

