import numpy as np

class Cut:
    def __init__(self, k):
        self.k = k
        return

class Dual:
    def __init__(self, k):
        self.k = k
        return

class Mid:
    def __init__(self, k):
        self.k = k
        return

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
        self.n_ghost = n_ghost
        self.z_min = z_min
        self.z_max = z_max
        self.nz = n_elems_real
        self.nzg = self.nz + 2 * self.gw
        self.z_half = np.empty((self.nzg), dtype=np.double, order='c')
        self.z      = np.empty((self.nzg), dtype=np.double, order='c')
        count = 0
        for i in range(-self.gw, self.nz+self.gw,1):
            self.z[count] = (i + 1) * self.dz
            self.z_half[count] = (i+0.5)*self.dz
            count += 1
        return

    def n_hat(self, b):
        if isinstance(b, Zmin):
            return -1
        elif isinstance(b, Zmax):
            return 1
        else:
            raise TypeError("Bad boundary in n_hat in Grid.py")

    def binary(self, b):
        if isinstance(b, Zmin):
            return 0
        elif isinstance(b, Zmax):
            return 1
        else:
            raise TypeError("Bad boundary in n_hat in Grid.py")

    def first_interior(self, b):
        if isinstance(b, Zmin):
            return self.gw
        elif isinstance(b, Zmax):
            return self.nz+self.gw-1
        else:
            raise TypeError("Bad boundary in n_hat in Grid.py")

    def boundary(self, b): # Index for fields at full location only (maybe add location to be sure it's not misused?)
        return self.first_interior(b)-1+self.binary(b)

    def slice_real(self, loc):
        if isinstance(loc, Center):
            return slice(self.first_interior(Zmin()),self.first_interior(Zmax())+1,1)
        elif isinstance(loc, Node):
            return slice(self.boundary(Zmin()),self.boundary(Zmax())+1,1)
        else:
            raise TypeError("Bad location in slice_real in Grid.py")

    def slice_all(self, loc):
        if isinstance(loc, Center):
            return slice(0, self.first_interior(Zmax())+2, 1)
        elif isinstance(loc, Node):
            return slice(0, self.first_interior(Zmax())+2, 1)
        else:
            raise TypeError("Bad location in slice_all in Grid.py")

    def slice_ghost(self, loc, b):
        if isinstance(loc, Center):
            if isinstance(b, Zmin):
                return slice(0, self.first_interior(b)-1,1)
            elif isinstance(b, Zmax):
                return slice(self.first_interior(b)+1, self.nzg,1)
            else:
                raise TypeError("Bad boundary 1 in slice_ghost in Grid.py")
        elif isinstance(loc, Node):
            if isinstance(b, Zmin):
                return slice(0, self.boundary(b)-1,1)
            elif isinstance(b, Zmax):
                return slice(self.boundary(b)+1, self.nzg,1)
            else:
                raise TypeError("Bad boundary 2 in slice_ghost in Grid.py")
        else:
            raise TypeError("Bad location in slice_ghost in Grid.py")

    def over_elems_ghost(self, loc, b):
        if isinstance(loc, Center):
            if isinstance(b, Zmin):
                return list(range(0, self.first_interior(b)))
            elif isinstance(b, Zmax):
                return list(range(self.first_interior(b)+1, self.nzg))
            else:
                print('loc, b = ', loc, b)
                raise TypeError("Bad boundary 1 in over_elems_ghost in Grid.py")
        elif isinstance(loc, Node):
            if isinstance(b, Zmin):
                return list(range(0, self.boundary(b)+1))
            elif isinstance(b, Zmax):
                return list(range(self.boundary(b)+1, self.nzg))
            else:
                print('loc, b = ', loc, b)
                raise TypeError("Bad boundary 2 in over_elems_ghost in Grid.py")
        else:
            print('loc, b = ', loc, b)
            raise TypeError("Bad location in over_elems_ghost in Grid.py")

    def over_elems(self, loc):
        if isinstance(loc, Center):
            return range(0, self.first_interior(Zmax())+1+1)
        elif isinstance(loc, Node):
            return range(0, self.boundary(Zmax())+1+1)
        else:
            raise TypeError("Bad location in over_elems in Grid.py")

    def over_elems_real(self, loc):
        if isinstance(loc, Center):
            return range(self.first_interior(Zmin()), self.first_interior(Zmax())+1)
        elif isinstance(loc, Node):
            return range(self.boundary(Zmin()), self.boundary(Zmax())+1)
        else:
            raise TypeError("Bad location in over_elems_real in Grid.py")

