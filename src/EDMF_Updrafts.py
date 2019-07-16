import numpy as np
from parameters import *
from funcs_thermo import  *
from funcs_micro import  *
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann, nice_name
from funcs_EDMF import *
from NetCDFIO import NetCDFIO_Stats
import pylab as plt

def compute_sources(grid, q, tmp, max_supersaturation):
    i_gm, i_env, i_uds, i_sd = tmp.domain_idx()
    for i in i_uds:
        for k in grid.over_elems(Center()):
            q_tot = q['q_tot_tmp', i][k]
            q_tot = tmp['q_liq', i][k]
            T = tmp['T', i][k]
            p_0 = tmp['p_0_half'][k]
            tmp_qr = acnv_instant(q_tot, q_tot, max_supersaturation, T, p_0)
            tmp['prec_src_θ_liq', i][k] = rain_source_to_thetal(p_0, T, q_tot, q_tot, 0.0, tmp_qr)
            tmp['prec_src_q_tot', i][k] = -tmp_qr
    for k in grid.over_elems(Center()):
        tmp['prec_src_θ_liq', i_gm][k] = np.sum([tmp['prec_src_θ_liq', i][k] * q['a_tmp', i][k] for i in i_uds])
        tmp['prec_src_q_tot', i_gm][k] = np.sum([tmp['prec_src_q_tot', i][k] * q['a_tmp', i][k] for i in i_uds])
    return

def update_updraftvars(grid, q, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for i in i_uds:
        for k in grid.over_elems(Center()):
            s = tmp['prec_src_q_tot', i][k]
            q['q_tot_tmp', i][k] += s
            tmp['q_liq', i][k] += s
            q['q_rai_tmp', i][k] -= s
            q['θ_liq_tmp', i][k] += tmp['prec_src_θ_liq', i][k]
    return

def buoyancy(grid, q, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for i in i_uds:
        for k in grid.over_elems_real(Center()):
            if q['a_tmp', i][k] > 1e-3:
                q_tot = q['q_tot_tmp', i][k]
                q_vap = q_tot - tmp['q_liq', i][k]
                T = tmp['T', i][k]
                α_i = alpha_c(tmp['p_0_half'][k], T, q_tot, q_vap)
                tmp['B', i][k] = buoyancy_c(tmp['α_0_half'][k], α_i)
            else:
                tmp['B', i][k] = tmp['B', i_env][k]
    # Subtract grid mean buoyancy
    for k in grid.over_elems_real(Center()):
        tmp['B', i_gm][k] = q['a', i_env][k] * tmp['B', i_env][k]
        for i in i_uds:
            tmp['B', i_gm][k] += q['a_tmp', i][k] * tmp['B', i][k]
        for i in i_uds:
            tmp['B', i][k] -= tmp['B', i_gm][k]
        tmp['B', i_env][k] -= tmp['B', i_gm][k]
    return

def compute_cloud_base_top_cover(grid, q, tmp, UpdVar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for i in i_uds:
        UpdVar.cloud_base[i] = grid.z_half[grid.nzg-grid.gw-1]
        UpdVar.cloud_top[i] = 0.0
        UpdVar.cloud_cover[i] = 0.0
        for k in grid.over_elems_real(Center()):
            if tmp['q_liq', i][k] > 1e-8 and q['a_tmp', i][k] > 1e-3:
                UpdVar.cloud_base[i] = np.fmin(UpdVar.cloud_base[i], grid.z_half[k])
                UpdVar.cloud_top[i] = np.fmax(UpdVar.cloud_top[i], grid.z_half[k])
                UpdVar.cloud_cover[i] = np.fmax(UpdVar.cloud_cover[i], q['a_tmp', i][k])
    return

def assign_values_to_new(grid, q, q_new, tmp):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    # slice_all_c = grid.slice_all(Center())
    for i in i_uds:
        for k in grid.over_elems(Center()):
            q['w_tmp', i][k] = q_new['w', i][k]
            q['a_tmp', i][k] = q_new['a', i][k]
            q['q_tot_tmp', i][k] = q_new['q_tot', i][k]
            q['q_rai_tmp', i][k] = q_new['q_rai', i][k]
            q['θ_liq_tmp', i][k] = q_new['θ_liq', i][k]
    return

def initialize(grid, tmp, q, UpdVar, updraft_fraction):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    k_1 = grid.first_interior(Zmin())
    n_updrafts = len(i_uds)
    for i in i_uds:
        for k in grid.over_elems(Center()):
            q['w_tmp', i][k] = 0.0
            q['a_tmp', i][k] = 0.0
            q['q_tot_tmp', i][k] = q['q_tot', i_gm][k]
            tmp['q_liq', i][k] = tmp['q_liq', i_gm][k]
            q['q_rai_tmp', i][k] = q['q_rai', i_gm][k]
            q['θ_liq_tmp', i][k] = q['θ_liq', i_gm][k]
            tmp['T', i][k] = tmp['T', i_gm][k]
            tmp['B', i][k] = 0.0
        q['a_tmp', i][k_1] = updraft_fraction/n_updrafts
    for i in i_uds: q['q_tot_tmp', i].apply_bc(grid, 0.0)
    for i in i_uds: q['q_rai_tmp', i].apply_bc(grid, 0.0)
    for i in i_uds: q['θ_liq_tmp', i].apply_bc(grid, 0.0)
    for k in grid.over_elems(Center()):
        for i in i_uds:
            q['a', i][k] = q['a_tmp', i][k]
        q['a', i_env][k] = 1.0 - sum([q['a_tmp', i][k] for i in i_uds])
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

