import sys
import numpy as np
import copy
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid
import matplotlib.pyplot as plt
from Field import Field, Full, Half, Dirichlet, Neumann
import pandas as pd
from VarMapper import *
from DomainIdx import *

# markershapes = ['r-o', 'b-o', 'k-^', 'g-^', 'r-o', 'b-o', 'k-^', 'g-^', 'r-o', 'b-o', 'k-^', 'g-^', 'r-o', 'b-o', 'k-^', 'g-^', 'r-o', 'b-o', 'k-^', 'g-^']
markershapes = ['b-', 'b-o', 'k-^', 'g-^', 'r-o', 'b-o', 'k-^', 'g-^', 'r-o', 'b-o', 'k-^', 'g-^', 'r-o', 'b-o', 'k-^', 'g-^', 'r-o', 'b-o', 'k-^', 'g-^']

class UseDat:
    def __init__(self):
        return

def friendly_name(s):
    s = s.replace('ρ', 'rho')
    s = s.replace('α', 'alpha')
    s = s.replace('θ', 'theta')
    return s

class StateVec:
    def __init__(self, var_tuple, grid, dd):

        self.n_subdomains = dd.sum()
        self.var_mapper, self.dss_per_var, self.var_names = get_var_mapper(var_tuple, dd)
        idx = DomainIdx(dd)
        self.idx = idx
        self.sd_unmapped = get_sd_unmapped(var_tuple, idx, dd)

        self.idx_ss_per_var = {name : DomainIdx(dd, self.dss_per_var[name]) for name in self.var_names}
        self.a_map = {name : get_sv_a_map(idx, self.idx_ss_per_var[name]) for name in self.var_names}

        self.var_names = [sys.intern(x) for x in self.var_names]
        self.locs = {name : loc for name, dss, loc, bc in var_tuple}
        self.nsd = {name : dss.sum(dd) for name, dss, loc, bc in var_tuple}
        self.bcs = {name : bc for name, dss, loc, bc in var_tuple}
        self.fields = [Field.field(grid, self.locs[v], self.bcs[v]) for v in self.var_mapper for i in range(self.nsd[v])]
        return

    def var_suffix(self, name, i = None):
        if i==None:
            i = self.idx.gridmean()
        return self.idx.var_suffix(self.idx, name, i)

    def var_string(self, name, i = None):
        if i==None:
            i = self.idx.gridmean()
        return name+self.var_suffix(name, i)

    def __getitem__(self, tup):
        if isinstance(tup, tuple):
            name, i = tup
            i_sv = get_i_state_vec(self.var_mapper, self.a_map[name], name, i)
        else:
            name = tup
            i = self.idx.gridmean()
            i_sv = get_i_state_vec(self.var_mapper, self.a_map[name], name, i)
        return self.fields[i_sv]

    def __str__(self):
        s = ''
        s+= '\n------------------ StateVec'
        s+= '\nn_subdomains = '+str(self.n_subdomains)
        s+= '\nvar_names    = '+str(self.var_names)
        s+= '\nvar_mapper   = '+str(self.var_mapper)
        s+= '\nfields = \n'+'\n'.join([str(x) for x in self.fields])
        return s

    def assign(self, grid, name, value):
        if isinstance(name, tuple):
            for k in grid.over_elems(Center()):
                for v in name:
                    for i in self.over_sub_domains(v):
                        self[v, i][k] = value
        elif isinstance(name, str):
            for k in grid.over_elems(Center()):
                for i in self.over_sub_domains(name):
                    self[name, i][k] = value

    def data_location(self, name):
        return self.fields[self.var_mapper[name][0]].loc

    def over_sub_domains(self, name):
        return [x for x in self.sd_unmapped[name] if not x==0] # sd_unmapped is between 1 and length of all domains

    def surface(self, grid, var_name, i=None):
        if i==None:
            i = self.idx.gridmean()
        k = grid.first_interior(Zmin())
        return self[var_name, i].Dual(k)[0]

    def var_names_except(self, names=()):
        return [name for name in self.var_names if not name in names]

    def plot_state(self, grid, directory, filename, name_idx = None, i_sd = None, include_ghost = True):
        if i_sd==None:
            i_sd = self.idx.gridmean()
        domain_range = grid.over_elems(Center()) if include_ghost else grid.over_elems_real(Center())

        x = [grid.z_half[k] for k in domain_range]
        if name_idx == None:
            r = 0
            for name_idx in self.var_names:
                y = [self[name_idx, i_sd][k] for k in domain_range]
                plt.plot(y, x, markershapes[r], label=friendly_name(name_idx))
                plt.hold(True)
                r+=1
            plt.title('state vector vs z')
            plt.xlabel('state vector')
            plt.ylabel('z')
        else:
            x_name = filename
            y = [self[name_idx, i_sd][k] for k in domain_range]
            plt.plot(y, x, markershapes[i_sd])

            plt.title(x_name + ' vs z')
            plt.xlabel(x_name)
            plt.ylabel('z')
        plt.savefig(directory + filename)
        plt.close()

    def export_state(self, grid, directory, filename, ExportType = UseDat()):
        domain = grid.over_elems(Center())
        vn = self.var_names
        if len(vn)==7:
            vn = vn[0:5]
        elif len(vn)>7:
            vn = vn[0:12]

        headers = [self.var_string(name, i) for name in vn for i in self.over_sub_domains(name)]
        n_vars = len(headers)
        include_z = False
        if include_z:
            headers = ['z']+headers
        n_elem = len(domain)
        data = np.array([self[name, i][k] for k in domain for name in vn for i in self.over_sub_domains(name)])
        data = data.reshape(n_elem, n_vars)
        z = grid.z_half[domain]
        z = np.expand_dims(z, axis=1)
        data_all = data
        if include_z:
            data_all = np.hstack((z, data_all))

        file = directory+filename+'.csv'
        df = pd.DataFrame(data=data_all.astype(float))
        df.to_csv(file, sep=' ', header=headers, float_format='%6.8f', index=False)

        file = directory+filename+'_aligned.csv'
        space_buffer = 2
        max_len = 15
        fmt = '%'+str(max_len)+'.8f'
        headers = [str(' '*(max_len-len(x))+x).replace('"','') for x in headers]
        df = pd.DataFrame(data=data_all.astype(float))
        df.to_csv(file, sep=' ', header=headers, float_format=fmt, index=False)

        with open(file, 'r', encoding='utf-8') as f:
            data = f.read().replace('"', '')
        with open(file, 'w+', encoding='utf-8') as f:
            f.write(data)

        file = directory+'z_only.csv'
        df = pd.DataFrame(data=z.astype(float))
        df.to_csv(file, sep=' ', header=['z'], float_format='%6.8f', index=False)
        return


# Example:
"""
z_min        = 0.0
z_max        = 1.0
n_elems_real = 10
n_ghost      = 1
grid = Grid(z_min, z_max, n_elems_real, n_ghost)
unknowns = (('rho_0'  , Center() , Neumann(), 1),
            ('w'      , Node()   , Neumann(), 3),
            ('a'      , Center() , Neumann(), 3),
            ('alpha_0', Center() , Neumann(), 1))
state_vec = StateVec(unknowns, grid)
print(state_vec)
state_vec['rho_0'][1] = 1.0
state_vec['rho_0'][3] = 4.0
print(state_vec)
state_vec['w', 2][2] = 3.0
print(state_vec)
print('grid.z      = ', grid.z)
print('grid.z_half = ', grid.z_half)
print(state_vec.over_sub_domains())
print(state_vec.over_sub_domains(1))
print(state_vec.over_sub_domains('rho_0'))
print(state_vec.var_names_except('rho_0'))

for k in grid.over_elems(Center()):
    state_vec['rho_0'][k] = 2.0
    print('k = ', k)

for k in grid.over_elems_real(Center()):
    state_vec['rho_0'][k] = 3.0
    print('k = ', k)

state_vec.plot_state(grid, './', 'test')
"""



