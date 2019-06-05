import numpy as np
import copy

import numpy as np

class Grid:
    def __init__(self, z_min, z_max, n_elems_real, n_ghost):
        self.dz = (z_max-z_min)/n_elems_real
        self.dzi = 1.0/self.dz
        self.gw = n_ghost
        self.nz = n_elems_real
        self.nzg = self.nz + 2 * self.gw
        self.z_half = np.empty((self.nzg),dtype=np.double,order='c')
        self.z      = np.empty((self.nzg),dtype=np.double,order='c')
        count = 0
        for i in range(-self.gw,self.nz+self.gw,1):
            self.z[count] = (i + 1) * self.dz
            self.z_half[count] = (i+0.5)*self.dz
            count += 1
        return

    def k_first_full(self):
        return 0

    def k_first_half(self):
        return 0

    def k_last_full(self):
        return self.nzg

    def k_last_half(self):
        return self.nzg

    def k_surface(self): # Index for fields at full location
        return self.gw-1

    def k_surface_bl(self): # Index for fields at half location
        return self.gw

    def k_top_atmos(self): # Index for fields at full location
        return self.nz+self.gw-1

    def k_top_atmos_bl(self): # Index for fields at half location
        return self.nz+self.gw-1 # = nzg - 1

    def k_surface_ghost_full(self):
        return self.k_surface()-1

    def k_surface_ghost_half(self):
        return self.k_surface_bl()-1

    def k_top_atmos_ghost_full(self):
        return self.k_top_atmos()+1

    def k_top_atmos_ghost_half(self):
        return self.k_top_atmos_bl()+1

    def k_full_real(self):
        return slice(self.k_surface(),self.k_top_atmos()+1,1)

    def k_half_real(self):
        return slice(self.k_surface_bl(),self.k_top_atmos_bl()+1,1)

    def z_surface(self):
        return self.z[self.k_surface()]

    def z_surface_bl(self):
        return self.z_half[self.k_surface_bl()]

    def z_full_real(self):
        return [self.z[i] for i in self.over_points_full_real()]

    def z_half_real(self):
        return [self.z_half[i] for i in self.over_points_half_real()]

    def over_points_full(self):
        return range(self.k_surface_ghost_full(), self.k_top_atmos_ghost_full()+1)

    def over_points_half(self):
        return range(self.k_surface_ghost_half(), self.k_top_atmos_ghost_half()+1)

    def over_points_full_ghost_surface(self):
        return list(range(self.k_first_full(), self.k_surface()+1))

    def over_points_full_ghost_top(self):
        return list(range(self.k_top_atmos_ghost_full(), self.k_last_full()))

    def over_points_half_ghost_surface(self):
        return list(range(self.k_first_half(), self.k_surface_bl()))

    def over_points_half_ghost_top(self):
        return list(range(self.k_top_atmos_ghost_half(), self.k_last_half()))

    def over_points_half_ghost(self):
        return list(range(self.k_surface_ghost_half(), self.k_top_atmos_ghost_half()))

    def over_points_full_real(self):
        return range(self.k_surface(), self.k_top_atmos()+1)

    def over_points_half_real(self):
        return range(self.k_surface_bl(), self.k_top_atmos_bl()+1)



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
        self.var_names, self.var_mapper = get_var_mapper(var_tuple)
        self.fields = np.zeros((self.n_vars, grid.nz))
        return

    def __getitem__(self, name, k, i=1):
        return self.fields[self.var_mapper[name][i], k]

    def __setitem__(self, name, k, i=1, value):
        self.fields[self.var_mapper[name][i], k] = value

    def __str__(self):
        s = ''
        s+= '\n------------------ StateVec'
        s+= '\nn_subdomains = '+str(self.n_subdomains)
        s+= '\nvar_names    = '+str(self.var_names)
        s+= '\nvar_mapper   = '+str(self.var_mapper)
        s+= '\nfields = \n'+str(self.fields)
        return s

z_min        = 0.0
z_max        = 1.0
n_elems_real = 10
n_ghost      = 1
grid = Grid(z_min, z_max, n_elems_real, n_ghost)
unknowns = (('rho_0', 1), ('w', 3), ('a', 3), ('alpha_0', 1))
state_vec = StateVec(unknowns, grid)
print(state_vec)

state_vec['rho_0', 1, 1] = 2.0
print(state_vec)
state_vec['w', 2, 2] = 3.0
print(state_vec)





