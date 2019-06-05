import os
import numpy as np

class Cut:
    def __init__(self, k):
        self.k = k
        return

class Dual:
    def __init__(self, k):
        self.k = k
        return

class Field:
    def __init__(self, n, full_data):
        self.full_data = full_data
        self.values = np.zeros((n,), dtype=np.double, order='c')
        return

    @classmethod
    def half(cls, grid):
        return Field(grid.nzg, False)

    @classmethod
    def full(cls, grid):
        return Field(grid.nzg, True)

    @classmethod
    def field(cls, grid, loc):
        if loc == 'half':
            return Field.half(grid)
        elif loc == 'full':
            return Field.full(grid)
        else:
            print('Invalid location setting for variable! Must be half or full')

    def __getitem__(self, key):
        if not (isinstance(key, Dual) or isinstance(key, Cut)):
            return self.values[key]
        elif isinstance(key, Cut):
            return [self.values[k] for k in [key.k-1, key.k, key.k+1]]
        else:
            if self.full_data:
                if isinstance(key, Dual):
                    return [self.values[k] for k in [key.k-1, key.k]]
                else:
                    print('key = '+str(key))
                    raise ValueError('Bad key in full Field.py')
            else:
                if isinstance(key, Dual):
                    return [self.values[k] for k in [key.k-1, key.k]]
                else:
                    print('key = '+str(key))
                    raise ValueError('Bad key in half Field.py')


    def __setitem__(self, key, value):
        self.values[key] = value

    def __len__(self):
        return len(self.values)

    def export_data(self, grid, file_name):
        if self.full_data:
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
        if self.full_data:
            return self.values[grid.surface()]
        else:
            return (self.values[grid.k_surface_bl()]+self.values[grid.k_surface_bl()-1])/2.0

    def surface_bl(self, grid):
        if self.full_data:
            return (self.values[grid.surface()]+self.values[grid.surface()+1])/2.0
        else:
            return self.values[grid.k_surface_bl()]

    def extrap(self, grid):
        if self.full_data:
            for k in reversed(grid.over_points_full_ghost_surface()):
                self.values[k] = 2.0*self.values[k+1] - self.values[k+2]
            for k in grid.over_points_full_ghost_top():
                self.values[k] = 2.0*self.values[k-1] - self.values[k-2]
        else:
            for k in reversed(grid.over_points_half_ghost_surface()):
                self.values[k] = 2.0*self.values[k+1] - self.values[k+2]
            for k in grid.over_points_half_ghost_top():
                self.values[k] = 2.0*self.values[k-1] - self.values[k-2]


    def apply_Neumann(self, grid, value):
        if self.full_data:
            self.values[0:grid.k_surface()]             = 2.0*self.values[grid.k_surface()  ] - self.values[grid.k_surface()  +1] + 2.0*grid.dz*value*(+1)
            self.values[grid.k_top_atmos_ghost_full():] = 2.0*self.values[grid.k_top_atmos()] - self.values[grid.k_top_atmos()-1] + 2.0*grid.dz*value*(-1)
        else:
            self.values[0:grid.k_surface_bl()]          = self.values[grid.k_surface_bl()  ] + grid.dz*value
            self.values[grid.k_top_atmos_ghost_half():] = self.values[grid.k_top_atmos_bl()] - grid.dz*value

    def apply_Dirichlet(self, grid, value):
        if self.full_data:
            self.values[0:grid.k_surface_ghost_full()]  = value
            self.values[grid.k_top_atmos_ghost_full():] = value
        else:
            self.values[0:grid.k_surface_ghost_half()]  = 2*value - self.values[grid.k_surface_bl()  ]
            self.values[grid.k_top_atmos_ghost_half():] = 2*value - self.values[grid.k_top_atmos_bl()]
