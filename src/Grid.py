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
        for i in range(-self.gw, self.nz+self.gw,1):
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

    def z_full(self):
        return [self.z[i] for i in self.over_points_full()]

    def z_half(self):
        return [self.z_half[i] for i in self.over_points_half()]

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

