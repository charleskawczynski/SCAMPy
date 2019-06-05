import numpy as np
import copy
from Grid import Grid

import numpy as np

class Cut:
    def __init__(self, k):
        self.k = k
        return

class Dual:
    def __init__(self, k):
        self.k = k
        return

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
        self.var_names, self.var_mapper = get_var_mapper(var_tuple)
        self.fields = np.zeros((self.n_vars, grid.nz))
        return

    def __getitem__(self, tup):
        if len(tup)==2:
            name, k = tup
            i = 0
        elif len(tup)==3:
            name, k, i = tup
        else:
            raise ValueError("__getitem__ called with wrong dimensions in StateVec.py")
        if not (isinstance(k, Cut) or isinstance(k, Dual)):
            return self.fields[self.var_mapper[name][i], k]
        else:
            if isinstance(k, Cut):
                return [self.fields[self.var_mapper[name][i], j] for j in range(k.k-1,k.k+2)]
            elif isinstance(k, Dual):
                return [(self.fields[self.var_mapper[name][i], j]+self.fields[self.var_mapper[name][i], j+1])/2 for j in range(k.k-1,k.k+1)]

    def __setitem__(self, tup, value):
        if len(tup)==2:
            name, k = tup
            i = 0
        elif len(tup)==3:
            name, k, i = tup
        else:
            raise ValueError("__getitem__ called with wrong dimensions in StateVec.py")
        self.fields[self.var_mapper[name][i], k] = value

    def __str__(self):
        s = ''
        s+= '\n------------------ StateVec'
        s+= '\nn_subdomains = '+str(self.n_subdomains)
        s+= '\nvar_names    = '+str(self.var_names)
        s+= '\nvar_mapper   = '+str(self.var_mapper)
        s+= '\nfields = \n'+str(self.fields)
        return s

    def over_sub_domains(self, i=None):
        if i==None:
            return list(range(self.n_subdomains))
        elif isinstance(i, int):
            return [j for j in range(self.n_subdomains) if not j==i]
        elif isinstance(i, str):
            return list(range(len(self.var_mapper[i])))
        elif isinstance(i, tuple):
            return [j for j in range(self.n_subdomains) if not j in i]
        else:
            raise TypeError("Bad index in over_sub_domains in StateVec.py")

    def var_names_except(self, names=()):
        return [name for name in self.var_names if not name in names]

"""
# Example:
z_min        = 0.0
z_max        = 1.0
n_elems_real = 10
n_ghost      = 1
grid = Grid(z_min, z_max, n_elems_real, n_ghost)
unknowns = (('ρ_0', 1), ('w', 3), ('a', 3), ('alpha_0', 1))
state_vec = StateVec(unknowns, grid)
print(state_vec)

state_vec['ρ_0', 1] = 1.0
state_vec['ρ_0', 3] = 4.0
print(state_vec)
state_vec['w', 2, 2] = 3.0
print(state_vec)
print(state_vec.over_sub_domains())
print(state_vec.over_sub_domains(1))
print(state_vec.over_sub_domains('ρ_0'))
print(state_vec.var_names_except('ρ_0'))
print(state_vec['ρ_0', Cut(2)])
"""



