import numpy as np
from parameters import *
from funcs_thermo import  *
from funcs_micro import  *
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann, nice_name
from funcs_EDMF import *
from NetCDFIO import NetCDFIO_Stats
from funcs_utility import *
import pylab as plt

def compute_cloud_base_top_cover(grid, q, tmp, UpdVar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    k_2 = grid.first_interior(Zmax())
    for i in i_uds:
        UpdVar[i].cloud_base = grid.z_half[k_2]
        UpdVar[i].cloud_top = 0.0
        UpdVar[i].cloud_cover = 0.0
        for k in grid.over_elems_real(Center()):
            a_ik = q['a_tmp', i][k]
            z_k = grid.z_half[k]
            if tmp['q_liq', i][k] > 1e-8 and a_ik > 1e-3:
                UpdVar[i].cloud_base  = np.fmin(UpdVar[i].cloud_base, z_k)
                UpdVar[i].cloud_top   = np.fmax(UpdVar[i].cloud_top, z_k)
                UpdVar[i].cloud_cover = np.fmax(UpdVar[i].cloud_cover, a_ik)
    return

def export_data_updrafts(grid, UpdVar, q, tmp, Stats):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    compute_cloud_base_top_cover(grid, q, tmp, UpdVar)
    Stats.write_ts('updraft_cloud_cover', np.sum([UpdVar[i].cloud_cover for i in i_uds]))
    Stats.write_ts('updraft_cloud_base' , np.amin([UpdVar[i].cloud_base for i in i_uds]))
    Stats.write_ts('updraft_cloud_top'  , np.amax([UpdVar[i].cloud_top for i in i_uds]))
    return

def initialize_io_updrafts(UpdVar, Stats):
    Stats.add_ts('updraft_cloud_cover')
    Stats.add_ts('updraft_cloud_base')
    Stats.add_ts('updraft_cloud_top')
    return

class UpdraftVariables:
    def __init__(self, i, surface_area, n_updrafts):
        self.cloud_base  = 0.0
        self.cloud_top   = 0.0
        self.cloud_cover = 0.0
        self.area_surface_bc  = 0.0
        self.w_surface_bc     = 0.0
        self.θ_liq_surface_bc = 0.0
        self.q_tot_surface_bc = 0.0
        self.surface_scalar_coeff = 0.0
        a_ = surface_area/n_updrafts
        self.surface_scalar_coeff = percentile_bounds_mean_norm(1.0-surface_area + i    *a_,
                                                                1.0-surface_area + (i+1)*a_ , 1000)
        return
