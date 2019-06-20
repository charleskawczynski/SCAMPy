import numpy as np
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut

def advect(f, w, grid):
    assert len(f)==3
    assert len(w)==3
    if w[1]<0.0:
      return (f[2]-f[1])*grid.dzi
    else:
      return (f[1]-f[0])*grid.dzi

def grad(f, grid):
    if len(f)==2:
      return (f[1]-f[0])*grid.dzi
    elif len(f)==3:
      return (f[2]-f[0])*0.5*grid.dzi
    else:
      raise ValueError('Bad length in Operators.py')

def Laplacian(f, grid, K = None):
    assert len(f)==3
    if K==None:
      return (f[2]+f[0]-2.0*f[1])*grid.dzi2
    else:
      assert len(f)==2
      return (K[1]*(f[2]-f[1])-K[0]*(f[1]-f[0]))*grid.dzi2

