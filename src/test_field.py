import os
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
import numpy as np
import cProfile


def test():
    grid = Grid(0.0, 1.0, 1000, 2)
    f = Half(grid)
    for i in range(1,1000):
        for k in grid.over_elems_real(Center()):
            temp = f.Mid(k)
            f[k]+=1.0


if __name__ == "__main__":
    cProfile.run('test()')

