import os
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
import numpy as np

def nice_name(s):
    s = s.replace('α', 'alpha')
    s = s.replace('ρ', 'rho')
    return s

class Neumann:
    def __init__(self):
        return

class Dirichlet:
    def __init__(self):
        return

class Field:
    def __init__(self, n, loc, bc = None):
        self.loc = loc
        self.bc = bc
        self.values = np.zeros((n,), dtype=np.double, order='c')
        return

    @classmethod
    def field(cls, grid, loc, bc = None):
        if isinstance(loc, Center):
            return Half(grid, bc)
        elif isinstance(loc, Node):
            return Full(grid, bc)
        else:
            print('loc = ', loc)
            raise TypeError('Bad location in field in Field.py')

    def __setitem__(self, key, value):
        self.values[key] = value

    def __len__(self):
        return len(self.values)

    def __str__(self):
        s = ''
        s+=str(self.loc)
        s+=str(self.values)
        return s

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

    def apply_bc(self, grid, value):
        if isinstance(self.bc, Dirichlet):
            self.apply_Dirichlet(grid, value)
        elif isinstance(self.bc, Neumann):
            self.apply_Neumann(grid, value)
        else:
            raise TypeError('Bad bc in apply_bc in Field.py')

class Full(Field):
    def __init__(self, grid, bc = None):
        super(Full, self).__init__(grid.nzg, Node(), bc)
        return

    def __getitem__(self, key):
        return self.values[key]

    def Mid(self, k):
        return 0.5*(self.values[k]+self.values[k-1])

    def Identity(self, k):
        return self.values[k]

    def Dual(self, key):
        return np.array([self.values[k] for k in [key-1, key]])

    def Cut(self, key):
        return np.array([self.values[k] for k in [key-1, key, key+1]])

    def DualCut(self, key):
        return np.array([0.5*(self.values[k]+self.values[k+1]) for k in [key-2, key-1, key]])

    def surface(self, grid):
        return self.values[grid.boundary(Zmin())]

    def first_interior(self, grid):
        k_b = grid.boundary(Zmin())
        return (self.values[k_b]+self.values[k_b+1])/2.0

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
    def __init__(self, grid, bc = None):
        super(Half, self).__init__(grid.nzg, Center(), bc)
        return

    def __getitem__(self, key):
        return self.values[key]

    def Mid(self, k):
        return 0.5*(self.values[k]+self.values[k+1])

    def Identity(self, k):
        return self.values[k]

    def Dual(self, key):
        return np.array([self.values[k] for k in [key, key+1]])

    def Cut(self, key):
        return np.array([self.values[k] for k in [key-1, key, key+1]])

    def DualCut(self, key):
        return np.array([0.5*(self.values[k]+self.values[k+1]) for k in [key-1, key, key+1]])

    def surface(self, grid):
        k_i = grid.first_interior(Zmin())
        return (self.values[k_i]+self.values[k_i-1])/2.0

    def first_interior(self, grid):
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
