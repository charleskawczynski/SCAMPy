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
    def half(cls, grid):
        return Field(grid.nzg, Center())

    @classmethod
    def full(cls, grid):
        return Field(grid.nzg, Node())

    @classmethod
    def field(cls, grid, loc):
        if isinstance(loc, Center):
            return Field.half(grid)
        elif isinstance(loc, Node):
            return Field.full(grid)
        else:
            print('loc = ', loc)
            raise TypeError('Bad location in field in Field.py')

    def __getitem__(self, key):
        if not (isinstance(key, Dual) or isinstance(key, Cut) or isinstance(key, Mid)):
            return self.values[key]
        elif isinstance(key, Cut):
            return [self.values[k] for k in [key.k-1, key.k, key.k+1]]
        else:
            if isinstance(self.loc, Node):
                if isinstance(key, Dual):
                    return [self.values[k] for k in [key.k-1, key.k]]
                elif isinstance(key, Mid):
                    return 0.5*(self.values[key.k]+self.values[key.k+1])
                else:
                    print('key = '+str(key))
                    raise ValueError('Bad key in full Field.py')
            elif isinstance(self.loc, Center):
                if isinstance(key, Dual):
                    return [self.values[k] for k in [key.k-1, key.k]]
                elif isinstance(key, Mid):
                    return 0.5*(self.values[key.k]+self.values[key.k+1])
                else:
                    print('key = '+str(key))
                    raise ValueError('Bad key in half Field.py')
            else:
                raise TypeError('Bad data location in __getitem__ in Field.py')


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

    def surface(self, grid):
        if isinstance(self.loc, Node):
            return self.values[grid.surface()]
        else:
            return (self.values[grid.first_interior(Zmin())]+self.values[grid.first_interior(Zmin())-1])/2.0

    def surface_bl(self, grid):
        if isinstance(self.loc, Node):
            return (self.values[grid.surface()]+self.values[grid.surface()+1])/2.0
        else:
            return self.values[grid.first_interior(Zmin())]

    def extrap(self, grid):
        if isinstance(self.loc, Node):
            for k in reversed(grid.over_elems_ghost(Node(), Zmin())):
                self.values[k] = 2.0*self.values[k+1] - self.values[k+2]
            for k in grid.over_elems_ghost(Node(), Zmax()):
                self.values[k] = 2.0*self.values[k-1] - self.values[k-2]
        else:
            for k in reversed(grid.over_elems_ghost(Center(), Zmin())):
                self.values[k] = 2.0*self.values[k+1] - self.values[k+2]
            for k in grid.over_elems_ghost(Center(), Zmax()):
                self.values[k] = 2.0*self.values[k-1] - self.values[k-2]

    def apply_Neumann(self, grid, value):
        n_hat_zmin = grid.n_hat(Zmin())
        n_hat_zmax = grid.n_hat(Zmax())
        if isinstance(self.loc, Node):
            self.values[0:grid.boundary(Zmin())]  = 2.0*self.values[grid.boundary(Zmin())] - self.values[grid.boundary(Zmin())-n_hat_zmin] + 2.0*grid.dz*value*n_hat_zmin
            self.values[grid.boundary(Zmax())+1:] = 2.0*self.values[grid.boundary(Zmax())] - self.values[grid.boundary(Zmax())-n_hat_zmax] + 2.0*grid.dz*value*n_hat_zmax
        else:
            self.values[0:grid.first_interior(Zmin())]  = self.values[grid.first_interior(Zmin())] - n_hat_zmin*grid.dz*value
            self.values[grid.first_interior(Zmax())+1:] = self.values[grid.first_interior(Zmax())] - n_hat_zmax*grid.dz*value

    def apply_Dirichlet(self, grid, value):
        if isinstance(self.loc, Node):
            self.values[0:grid.boundary(Zmin())+1]  = value
            self.values[grid.boundary(Zmax()):] = value
        else:
            self.values[0:grid.first_interior(Zmin())] = 2*value - self.values[grid.first_interior(Zmin())]
            self.values[grid.first_interior(Zmax())+1:]  = 2*value - self.values[grid.first_interior(Zmax())]


    def apply_bc(self, grid, bc, value):
        if isinstance(bc, Dirichlet):
            self.apply_Dirichlet(grid, value)
        elif isinstance(bc, Neumann):
            self.apply_Neumann(grid, value)
        else:
            raise TypeError('Bad bc in apply_bc in Field.py')


