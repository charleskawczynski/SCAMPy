import numpy as np
from MoistThermodynamics import *

def convert_forcing_thetal(p_0, q_tot, q_vap, T, q_tot_tendency, T_tendency):
    return T_tendency/exner_raw(p_0)
