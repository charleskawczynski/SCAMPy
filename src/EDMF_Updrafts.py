import numpy as np
from parameters import *
from funcs_thermo import  *
from funcs_micro import  *
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann, nice_name
from funcs_EDMF import *
from NetCDFIO import NetCDFIO_Stats
import pylab as plt

def compute_cloud_base_top_cover(grid, q, tmp, UpdVar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    # k_2 = grid.first_interior(Zmax())
    for i in i_uds:
        UpdVar.cloud_base[i] = grid.z_half[grid.nzg-grid.gw-1]
        # UpdVar.cloud_base[i] = grid.z_half[k_2]
        UpdVar.cloud_top[i] = 0.0
        UpdVar.cloud_cover[i] = 0.0
        for k in grid.over_elems_real(Center()):
            if tmp['q_liq', i][k] > 1e-8 and q['a_tmp', i][k] > 1e-3:
                UpdVar.cloud_base[i] = np.fmin(UpdVar.cloud_base[i], grid.z_half[k])
                UpdVar.cloud_top[i] = np.fmax(UpdVar.cloud_top[i], grid.z_half[k])
                UpdVar.cloud_cover[i] = np.fmax(UpdVar.cloud_cover[i], q['a_tmp', i][k])
    return

class UpdraftVariables:
    def __init__(self, nu, namelist, paramlist, grid):
        self.n_updrafts = nu

        self.cloud_base  = np.zeros((nu,), dtype=np.double, order='c')
        self.cloud_top   = np.zeros((nu,), dtype=np.double, order='c')
        self.cloud_cover = np.zeros((nu,), dtype=np.double, order='c')
        return

    def initialize_io(self, Stats):
        Stats.add_ts('updraft_cloud_cover')
        Stats.add_ts('updraft_cloud_base')
        Stats.add_ts('updraft_cloud_top')
        return

    def export_data(self, grid, q, tmp, Stats):
        compute_cloud_base_top_cover(grid, q, tmp, self)
        Stats.write_ts('updraft_cloud_cover', np.sum(self.cloud_cover))
        Stats.write_ts('updraft_cloud_base' , np.amin(self.cloud_base))
        Stats.write_ts('updraft_cloud_top'  , np.amax(self.cloud_top))
        return

