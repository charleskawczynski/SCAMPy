import numpy as np
from funcs_thermo import pv_c, pd_c, sv_c, sd_c, cpm_c, exner_c

def convert_forcing_thetal(p_0, q_tot, q_vap, T, q_tot_tendency, T_tendency):
    return T_tendency/exner_c(p_0)
