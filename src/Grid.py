import numpy as np

class Grid:
    def __init__(self, namelist):
        #Get the grid spacing
        self.dz = namelist['grid']['dz']
        #Set the inverse grid spacing
        self.dzi = 1.0/self.dz
        #Get the grid dimensions and ghost points
        self.gw = namelist['grid']['gw']
        self.nz = namelist['grid']['nz']
        self.nzg = self.nz + 2 * self.gw
        self.z_half = np.empty((self.nzg),dtype=np.double,order='c')
        self.z      = np.empty((self.nzg),dtype=np.double,order='c')
        count = 0
        for i in range(-self.gw,self.nz+self.gw,1):
            self.z[count] = (i + 1) * self.dz
            self.z_half[count] = (i+0.5)*self.dz
            count += 1
        return

    def k_surface(self): # Index for fields at full location
        return self.gw-1

    def k_surface_bl(self): # Index for fields at half location
        return self.gw

    def k_top_atmos(self): # Index for fields at full location
        return self.nz+self.gw

    def k_top_atmos_bl(self): # Index for fields at half location
        return self.nz+self.gw

    def k_full_real(self):
        return slice(self.k_surface(),self.k_top_atmos(),1)

    def k_half_real(self):
        return slice(self.k_surface_bl(),self.k_top_atmos_bl(),1)

    def z_surface(self):
        return self.z[self.k_surface()]

    def z_surface_bl(self):
        return self.z_half[self.k_surface_bl()]

    def z_full_real(self):
        return [self.z[i] for i in self.over_points_full_real()]

    def z_half_real(self):
        return [self.z_half[i] for i in self.over_points_half_real()]

    def over_points_full(self):
        return range(self.nzg)

    def over_points_half(self):
        return range(self.nzg)

    def over_points_full_real(self):
        return range(self.k_surface(), self.k_top_atmos())

    def over_points_half_real(self):
        return range(self.k_surface_bl(), self.k_top_atmos_bl())

