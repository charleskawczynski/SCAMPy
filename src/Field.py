import os
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
import numpy as np

class Neumann:
    def __init__(self):
        return

class Dirichlet:
    def __init__(self):
        return

class Field:
    def __init__(self, n, loc):
        self.loc = loc
        self.values = np.zeros((n,), dtype=np.double, order='c')
        return

    @classmethod
    def field(cls, grid, loc):
        if isinstance(loc, Center):
            return Half(grid)
        elif isinstance(loc, Node):
            return Full(grid)
        else:
            print('loc = ', loc)
            raise TypeError('Bad location in field in Field.py')

    def __setitem__(self, key, value):
        self.values[key] = value

    def __len__(self):
        return len(self.values)

    def export_data(self, grid, file_name):
        if isinstance(self.loc, Node):
            a = np.transpose(grid.z)
        else:
            a = np.transpose(grid.z_half)
        b = np.transpose(self.values)
        A = np.vstack((a, b))
        A = np.transpose(A)
        fmt = ",".join(["%10.6e"] * (2))
        ext = file_name.split('.')[-1]
        var_name = file_name.replace('.'+ext, '')
        var_name = var_name.replace(ext, '')
        var_name = var_name.split('.')[-1]
        var_name = var_name.split(os.sep)[-1]
        np.savetxt(file_name, A, fmt=fmt, header="z,"+var_name, comments='')
        return

    def apply_bc(self, grid, bc, value):
        if isinstance(bc, Dirichlet):
            self.apply_Dirichlet(grid, value)
        elif isinstance(bc, Neumann):
            self.apply_Neumann(grid, value)
        else:
            raise TypeError('Bad bc in apply_bc in Field.py')

class Full(Field):
    def __init__(self, grid):
        super(Full, self).__init__(grid.nzg, Node())
        return

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            return self.values[key]
        elif isinstance(key, Cut):
            return [self.values[k] for k in [key.k-1, key.k, key.k+1]]
        elif isinstance(key, Dual):
            return [self.values[k] for k in [key.k-1, key.k]]
        elif isinstance(key, Mid):
            return 0.5*(self.values[key.k]+self.values[key.k+1])
        else:
            print('key = '+str(key))
            raise ValueError('Bad key in full Field.py')

    def surface(self, grid):
        return self.values[grid.surface()]

    def surface_bl(self, grid):
        k_s = grid.surface()
        return (self.values[k_s]+self.values[k_s+1])/2.0

    def extrap(self, grid):
        for k in reversed(grid.over_elems_ghost(Node(), Zmin())):
            self.values[k] = 2.0*self.values[k+1] - self.values[k+2]
        for k in grid.over_elems_ghost(Node(), Zmax()):
            self.values[k] = 2.0*self.values[k-1] - self.values[k-2]

    def apply_Dirichlet(self, grid, value):
        self.values[0:grid.boundary(Zmin())+1]  = value
        self.values[grid.boundary(Zmax()):] = value

    def apply_Neumann(self, grid, value):
        n_hat_zmin = grid.n_hat(Zmin())
        n_hat_zmax = grid.n_hat(Zmax())
        k_1 = grid.boundary(Zmin())
        k_2 = grid.boundary(Zmax())
        self.values[0:k_1]  = 2.0*self.values[k_1] - self.values[k_1-n_hat_zmin] + 2.0*grid.dz*value*n_hat_zmin
        self.values[k_2+1:] = 2.0*self.values[k_2] - self.values[k_2-n_hat_zmax] + 2.0*grid.dz*value*n_hat_zmax

class Half(Field):
    def __init__(self, grid):
        super(Half, self).__init__(grid.nzg, Center())
        return

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            return self.values[key]
        elif isinstance(key, Cut):
            return [self.values[k] for k in [key.k-1, key.k, key.k+1]]
        elif isinstance(key, Dual):
            return [self.values[k] for k in [key.k-1, key.k]]
        elif isinstance(key, Mid):
            return 0.5*(self.values[key.k]+self.values[key.k+1])
        else:
            print('key = '+str(key))
            raise ValueError('Bad key in half Field.py')

    def surface(self, grid):
        k_i = grid.first_interior(Zmin())
        return (self.values[k_i]+self.values[k_i-1])/2.0

    def surface_bl(self, grid):
        k_i = grid.first_interior(Zmin())
        return self.values[k_i]

    def extrap(self, grid):
        for k in reversed(grid.over_elems_ghost(Center(), Zmin())):
            self.values[k] = 2.0*self.values[k+1] - self.values[k+2]
        for k in grid.over_elems_ghost(Center(), Zmax()):
            self.values[k] = 2.0*self.values[k-1] - self.values[k-2]

    def apply_Dirichlet(self, grid, value):
        k_1 = grid.first_interior(Zmin())
        k_2 = grid.first_interior(Zmax())
        self.values[0:k_1] = 2*value - self.values[k_1]
        self.values[k_2+1:]  = 2*value - self.values[k_2]

    def apply_Neumann(self, grid, value):
        n_hat_zmin = grid.n_hat(Zmin())
        n_hat_zmax = grid.n_hat(Zmax())
        k_1 = grid.first_interior(Zmin())
        k_2 = grid.first_interior(Zmax())
        self.values[0:k_1]  = self.values[k_1] - n_hat_zmin*grid.dz*value
        self.values[k_2+1:] = self.values[k_2] - n_hat_zmax*grid.dz*value
