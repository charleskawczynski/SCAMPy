import sys
import numpy as np
import copy
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid
import matplotlib.pyplot as plt
from Field import Field, Full, Half, Dirichlet, Neumann
import numpy as np
import pandas as pd

import numpy as np

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

def get_var_mapper(var_tuple):
    var_names = [var_name for (var_name, loc, bc, nsd) in var_tuple]

    var_names_test = sorted(var_names)
    var_names_unique = sorted(list(set(var_names_test)))
    assert len(var_names_test)==len(var_names_unique)
    assert all([x==y for x,y in zip(var_names_test, var_names_unique)])

    end_index = list(np.cumsum([nsd for (var_name, loc, bc, nsd) in var_tuple]))
    start_index = [0]+[x for x in end_index][0:-1]
    vals = [list(range(a,b)) for a,b in zip(start_index, end_index)]
    var_mapper = {k : v for k,v in zip(var_names, vals)}
    return var_names, var_mapper

class StateVec:
    def __init__(self, var_tuple, grid):
        self.n_subdomains = max([nsd for var_name, loc, bc, nsd in var_tuple])

        self.i_gm = self.n_subdomains-1
        self.i_env = self.n_subdomains-2
        self.i_uds = [i for i in range(self.n_subdomains) if not any([i==j for j in [self.i_env, self.i_gm]])]
        self.i_sd = self.i_uds+[self.i_env]

        self.n_vars = sum([nsd for var_name, loc, bc, nsd in var_tuple])
        self.var_names, self.var_mapper = get_var_mapper(var_tuple)
        self.var_names = [sys.intern(x) for x in self.var_names]
        n = len(list(grid.over_elems(Center())))
        self.locs = {var_name : loc for var_name, loc, bc, nsd in var_tuple}
        self.nsd = {var_name : nsd for var_name, loc, bc, nsd in var_tuple}
        self.bcs = {var_name : bc for var_name, loc, bc, nsd in var_tuple}
        self.fields = [Field.field(grid, self.locs[v], self.bcs[v]) for v in self.var_mapper for i in range(self.nsd[v])]
        return

    def idx_name(self, i):
        if i==self.i_gm:
            return "gm"
        elif i==self.i_env:
            return "en"
        elif i in self.i_uds:
            if len(self.i_uds)==1:
                return "ud"
            else:
                return 'ud_'+str(i)
        else:
            raise ValueError('Bad index in idx_name in StateVec.py')

    def var_suffix(self, name, i):
        if self.nsd[name] == 1:
            return '_gm'
        else:
            return '_'+self.idx_name(i)

    def var_string(self, name, i):
        if self.nsd[name] == 1:
            return name+'_gm'
        else:
            return name+'_'+self.idx_name(i)

    def get_gm(self, i):
        return self.i_gm

    def __getitem__(self, tup):
        if isinstance(tup, tuple):
            # name, i = tup
            return self.fields[self.var_mapper[tup[0]][tup[1]]]
        else:
            # name = tup
            # i = 0
            return self.fields[self.var_mapper[tup][0]]

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


    def domain_idx(self):
        return self.i_gm, self.i_env, self.i_uds, self.i_sd

    def subdomains(self, var_name):
        if self.nsd[var_name]==1:
            return (0,)
        else:
            return self.i_sd+[self.i_gm]

    def data_location(self, name):
        return self.fields[self.var_mapper[name][0]].loc

    def slice_updrafts(self):
        return slice(self.i_uds[0], self.i_uds[:-1])

    def slice_sub_domains(self): # restricts index order
        return slice(self.i_sd[0], self.i_sd[:-1])

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

    def surface(self, grid, var_name, i_sd=0):
        k = grid.first_interior(Zmin())
        return self[var_name, i_sd].Dual(k)[0]

    def var_names_except(self, names=()):
        return [name for name in self.var_names if not name in names]

    def plot_state(self, grid, directory, filename, name_idx = None, i_sd = 0, include_ghost = True):
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
        vn = vn[0:5] if len(vn)==7 else vn[0:12]

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
        spacing = 10
        fmt = '%'+str(spacing+space_buffer)+'.8f'
        headers = [str(' '*(spacing-len(x)+space_buffer)+x).replace('"','') for x in headers]
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



