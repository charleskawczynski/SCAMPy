import numpy as np

class Field:
    def __init__(self, n):
        self.values = np.zeros((n,), dtype=np.double, order='c')
        return

    @classmethod
    def full(cls, Gr):
        return Field(Gr.nzg)

    @classmethod
    def half(cls, Gr):
        return Field(Gr.nzg)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value


