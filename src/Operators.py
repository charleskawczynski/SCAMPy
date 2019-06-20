import numpy as np
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut

def advect(f, grid):
    assert len(f)==3
    if f[2]>2:
      return (f[2]-f[1])*grid.dzi
    else:
      return (f[3]-f[2])*grid.dzi

def grad(f, grid):
    if len(f)==2:
      return (f[2]-f[1])*grid.dzi
    elif len(f)==3:
      return (f[3]-f[1])*0.5*grid.dzi
    else:
      raise ValueError('Bad length in Operators.py')

def Laplacian(f, grid, K = None):
    assert len(f)==3
    if K==None:
      return (f[3]+f[1]-2.0*f[2])*grid.dzi2
    else:
      assert len(f)==2
      return (K[2]*(f[3]-f[2])-K[1]*(f[2]-f[1]))*grid.dzi2

