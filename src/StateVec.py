import numpy as np
import copy

"""
var_tuple = (('rho_0', 1), ('w', 3), ('a', 3), ('alpha_0', 1))
        unkowns = (
        ('a', N_subdomains),
        ('w', N_subdomains),
        ('q_tot', N_subdomains),
        ('θ_liq', N_subdomains),
        ('tke', N_subdomains),
        ('cv_q_tot', N_subdomains),
        ('cv_θ_liq', N_subdomains),
        ('cv_θ_liq_q_tot', N_subdomains),
        )
        temp_vars = (('rho_0', 1),
                     ('alpha_0', 1),
                     ('p_0', 1),
                     ('w', 3),
                     ('a', 3),
                     )
        unknowns = (('rho_0', 1), ('w', 3), ('a', 3), ('alpha_0', 1))
        self.state_vec = StateVec
        self.state_vec = StateVec

"""

def get_var_mapper(var_tuple):
    var_names = [v for (v, nsd) in var_tuple]
    end_index = list(np.cumsum([nsd for (v, nsd) in var_tuple]))
    start_index = [0]+[x for x in end_index][0:-1]
    vals = [list(range(a,b)) for a,b in zip(start_index, end_index)]
    var_mapper = {k : v for k,v in zip(var_names, vals)}
    return var_names, var_mapper

class StateVec:
    def __init__(self, var_tuple, grid):
        self.n_subdomains = max([nsd for v, nsd in var_tuple])
        self.n_vars = sum([nsd for v, nsd in var_tuple])
        self.var_names, self.var_names = get_var_mapper(var_tuple)
        self.all_vars = [0.0 for v in range(0, self.n_vars)]
        self.fields = [copy.deepcopy(all_vars) for k in grid.over_points_half()]
        return

