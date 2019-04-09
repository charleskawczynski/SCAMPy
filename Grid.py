import numpy as np
import time
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
        self.z_half = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')
        self.z = np.empty((self.nz+2*self.gw),dtype=np.double,order='c')
        count = 0
        for i in range(-self.gw,self.nz+self.gw,1):
            self.z[count] = (i + 1) * self.dz
            self.z_half[count] = (i+0.5)*self.dz
            count += 1
        return

    def over_points_full(self):
        return range(-self.gw, self.nz+self.gw,1)

    def over_points_full_real(self):
        return range(-self.gw,self.nz+self.gw,1)

    def over_points_half(self):
        return range(self.nzg)

    def over_points_half_real(self):
        return range(self.gw, self.nzg-self.gw)
