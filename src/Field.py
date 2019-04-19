import numpy as np

class Field:
    def __init__(self, n, full_data):
        self.full_data = full_data
        self.values = np.zeros((n,), dtype=np.double, order='c')
        return

    @classmethod
    def full(cls, grid):
        return Field(grid.nzg, True)

    @classmethod
    def half(cls, grid):
        return Field(grid.nzg, False)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __len__(self):
        return len(self.values)

    def surface(self, grid):
        if self.full_data:
            return self.values[grid.surface()]
        else:
            return (self.values[grid.surface_bl()]+self.values[grid.surface_bl()-1])/2.0

    def surface_bl(self, grid):
        if self.full_data:
            return (self.values[grid.surface()]+self.values[grid.surface()+1])/2.0
        else:
            return self.values[grid.surface_bl()]

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
