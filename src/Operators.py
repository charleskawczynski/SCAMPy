import numpy as np
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid

def advect(f, grid):
    assert len(f)==3
    return (f[2]-f[1])*grid.dzi

def ∇_z(f, grid):
    assert len(f)==2
    return (f[2]-f[1])*grid.dzi

def Δ_z(f, grid, K = None):
    if K==None:
        return (f[2]-f[1])*grid.dzi
    else:
        return (f[2]-f[1])*grid.dzi

