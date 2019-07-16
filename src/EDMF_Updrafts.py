import numpy as np
from parameters import *
from funcs_thermo import  *
from funcs_micro import  *
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann, nice_name
from funcs_EDMF import *
from NetCDFIO import NetCDFIO_Stats
import pylab as plt

def compute_sources(grid, q, tmp, UpdVar, max_supersaturation):
    i_gm, i_env, i_uds, i_sd = tmp.domain_idx()
    for i in i_uds:
        for k in grid.over_elems(Center()):
            q_tot = UpdVar.q_tot.values[i][k]
            q_tot = tmp['q_liq', i][k]
            T = tmp['T', i][k]
            p_0 = tmp['p_0_half'][k]
            tmp_qr = acnv_instant(q_tot, q_tot, max_supersaturation, T, p_0)
            tmp['prec_src_θ_liq', i][k] = rain_source_to_thetal(p_0, T, q_tot, q_tot, 0.0, tmp_qr)
            tmp['prec_src_q_tot', i][k] = -tmp_qr
    for k in grid.over_elems(Center()):
        tmp['prec_src_θ_liq', i_gm][k] = np.sum([tmp['prec_src_θ_liq', i][k] * UpdVar.Area.values[i][k] for i in i_uds])
        tmp['prec_src_q_tot', i_gm][k] = np.sum([tmp['prec_src_q_tot', i][k] * UpdVar.Area.values[i][k] for i in i_uds])
    return

def update_updraftvars(grid, q, tmp, UpdVar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for i in i_uds:
        for k in grid.over_elems(Center()):
            s = tmp['prec_src_q_tot', i][k]
            UpdVar.q_tot.values[i][k] += s
            tmp['q_liq', i][k] += s
            UpdVar.q_rai.values[i][k] -= s
            UpdVar.θ_liq.values[i][k] += tmp['prec_src_θ_liq', i][k]
    return

def compute_update_combined_local_thetal(tmp, T, q_tot, q_liq, q_rai, θ_liq, i, k, max_supersaturation):
    p_0_k = tmp['p_0_half'][k]
    q_tot_k = q_tot[i][k]
    q_liq_k = q_liq[i][k]
    T_k = T[i][k]
    tmp_qr = acnv_instant(q_liq_k, q_tot_k, max_supersaturation, T_k, p_0_k)
    s = -tmp_qr
    tmp['prec_src_q_tot', i][k] = s
    tmp['prec_src_θ_liq', i][k] = rain_source_to_thetal(p_0_k, T_k, q_tot_k, q_liq_k, 0.0, tmp_qr)
    q_tot[i][k] += s
    q_liq[i][k] += s
    q_rai[i][k] -= s
    θ_liq[i][k] += tmp['prec_src_θ_liq', i][k]
    return

def buoyancy(grid, q, tmp, UpdVar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for i in i_uds:
        for k in grid.over_elems_real(Center()):
            if UpdVar.Area.values[i][k] > 1e-3:
                q_tot = UpdVar.q_tot.values[i][k]
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
            tmp['B', i_gm][k] += UpdVar.Area.values[i][k] * tmp['B', i][k]
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
            if tmp['q_liq', i][k] > 1e-8 and UpdVar.Area.values[i][k] > 1e-3:
                UpdVar.cloud_base[i] = np.fmin(UpdVar.cloud_base[i], grid.z_half[k])
                UpdVar.cloud_top[i] = np.fmax(UpdVar.cloud_top[i], grid.z_half[k])
                UpdVar.cloud_cover[i] = np.fmax(UpdVar.cloud_cover[i], UpdVar.Area.values[i][k])
    return

def assign_values_to_new(grid, q, tmp, UpdVar):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for i in i_uds:
        for k in grid.over_elems(Center()):
            UpdVar.W.values[i][k] = UpdVar.W.new[i][k]
            UpdVar.Area.values[i][k] = UpdVar.Area.new[i][k]
            UpdVar.q_tot.values[i][k] = UpdVar.q_tot.new[i][k]
            tmp['q_liq', i][k] = UpdVar.q_liq.new[i][k]
            UpdVar.q_rai.values[i][k] = UpdVar.q_rai.new[i][k]
            UpdVar.θ_liq.values[i][k] = UpdVar.θ_liq.new[i][k]
            tmp['T', i][k] = UpdVar.T.new[i][k]
            tmp['B', i][k] = UpdVar.B.new[i][k]
    return

def initialize(grid, tmp, q, UpdVar, updraft_fraction):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    k_1 = grid.first_interior(Zmin())
    n_updrafts = len(i_uds)
    for i in i_uds:
        for k in grid.over_elems(Center()):
            UpdVar.W.values[i][k] = 0.0
            UpdVar.Area.values[i][k] = 0.0
            UpdVar.q_tot.values[i][k] = q['q_tot', i_gm][k]
            tmp['q_liq', i][k] = tmp['q_liq', i_gm][k]
            UpdVar.q_rai.values[i][k] = q['q_rai', i_gm][k]
            UpdVar.θ_liq.values[i][k] = q['θ_liq', i_gm][k]
            tmp['T', i][k] = tmp['T', i_gm][k]
            tmp['B', i][k] = 0.0
        UpdVar.Area.values[i][k_1] = updraft_fraction/n_updrafts
    UpdVar.q_tot.set_bcs(grid)
    UpdVar.q_rai.set_bcs(grid)
    UpdVar.θ_liq.set_bcs(grid)
    for k in grid.over_elems(Center()):
        for i in i_uds:
            q['a', i][k] = UpdVar.Area.values[i][k]
        q['a', i_env][k] = 1.0 - sum([UpdVar.Area.values[i][k] for i in i_uds])
    return


class UpdraftVariable:
    def __init__(self, grid, nu, loc, bc):
        self.values     = [Field.field(grid, loc, bc) for i in range(nu)]
        self.new        = [Field.field(grid, loc, bc) for i in range(nu)]

    def set_bcs(self, grid):
        n_updrafts = np.shape(self.values)[0]
        for i in range(n_updrafts):
            self.values[i].apply_bc(grid, 0.0)
        return

class UpdraftVariables:
    def __init__(self, nu, namelist, paramlist, grid):
        self.n_updrafts = nu

        self.W     = UpdraftVariable(grid, nu, Node()  , Dirichlet())
        self.Area  = UpdraftVariable(grid, nu, Center(), Neumann())
        self.q_tot = UpdraftVariable(grid, nu, Center(), Neumann())
        self.q_liq = UpdraftVariable(grid, nu, Center(), Neumann())
        self.q_rai = UpdraftVariable(grid, nu, Center(), Neumann())

        self.T     = UpdraftVariable(grid, nu, Center(), Neumann())
        self.B     = UpdraftVariable(grid, nu, Center(), Neumann())
        self.θ_liq = UpdraftVariable(grid, nu, Center(), Neumann())

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

