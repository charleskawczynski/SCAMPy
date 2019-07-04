import numpy as np
from funcs_thermo import pv_c, pd_c, sv_c, sd_c, cpm_c, exner_c

def convert_forcing_entropy(p0, qt, qv, T, qt_tendency, T_tendency):
    pv = pv_c(p0, qt, qv)
    pd = pd_c(p0, qt, qv)
    return cpm_c(qt) * T_tendency/T + (sv_c(pv,T)-sd_c(pd,T)) * qt_tendency

def convert_forcing_thetal(p0, qt, qv, T, qt_tendency, T_tendency):
    return T_tendency/exner_c(p0)