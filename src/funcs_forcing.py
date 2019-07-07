import numpy as np
from funcs_thermo import pv_c, pd_c, sv_c, sd_c, cpm_c, exner_c

def convert_forcing_entropy(p_0, q_tot, q_vap, T, q_tot_tendency, T_tendency):
    p_vap = pv_c(p_0, q_tot, q_vap)
    p_dry = pd_c(p_0, q_tot, q_vap)
    return cpm_c(q_tot) * T_tendency/T + (sv_c(p_vap,T)-sd_c(p_dry,T)) * q_tot_tendency

def convert_forcing_thetal(p_0, q_tot, q_vap, T, q_tot_tendency, T_tendency):
    return T_tendency/exner_c(p_0)
