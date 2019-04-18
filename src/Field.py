import numpy as np

class Field:
    def __init__(self, n, full_data):
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
            return self.values[grid.surface()]
        else:
            return (self.values[grid.surface()]+self.values[grid.surface()+1])/2.0


