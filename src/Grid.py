import numpy as np

class Zmin:
    def __init__(self):
        return

class Zmax:
    def __init__(self):
        return

class Center:
    def __init__(self):
        return

class Node:
    def __init__(self):
        return

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
        for i in range(-self.gw, self.nz+self.gw,1):
            self.z[count] = (i + 1) * self.dz
            self.z_half[count] = (i+0.5)*self.dz
            count += 1
        return

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

    def slice_real(self, loc):
        if isinstance(loc, Center):
            return slice(self.k_surface_bl(),self.k_top_atmos_bl()+1,1)
        elif isinstance(loc, Node):
            return slice(self.k_surface(),self.k_top_atmos()+1,1)
        else:
            raise TypeError("Bad location in slice_real in Grid.py")

    def over_points_full(self):
        return range(self.k_surface_ghost_full(), self.k_top_atmos_ghost_full()+1)

    def over_points_half(self):
        return range(self.k_surface_ghost_half(), self.k_top_atmos_ghost_half()+1)

    def over_points_full_ghost_surface(self):
        return list(range(0, self.k_surface()+1))

    def over_points_full_ghost_top(self):
        return list(range(self.k_top_atmos_ghost_full(), self.nzg))

    def over_points_half_ghost_surface(self):
        return list(range(0, self.k_surface_bl()))

    def over_points_half_ghost_top(self):
        return list(range(self.k_top_atmos_ghost_half(), self.nzg))

    def over_points_half_ghost(self):
        return list(range(self.k_surface_ghost_half(), self.k_top_atmos_ghost_half()))

    def over_points_full_real(self):
        return range(self.k_surface(), self.k_top_atmos()+1)

    def over_points_half_real(self):
        return range(self.k_surface_bl(), self.k_top_atmos_bl()+1)

