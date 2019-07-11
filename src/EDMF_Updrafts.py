import numpy as np
from parameters import *
from funcs_thermo import  *
from funcs_micro import  *
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann, nice_name
from Variables import GridMeanVariables
from NetCDFIO import NetCDFIO_Stats
from EDMF_Environment import EnvironmentVariables
import pylab as plt


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

        self.updraft_fraction = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']

        self.cloud_base  = np.zeros((nu,), dtype=np.double, order='c')
        self.cloud_top   = np.zeros((nu,), dtype=np.double, order='c')
        self.cloud_cover = np.zeros((nu,), dtype=np.double, order='c')
        return

    def initialize(self, grid, GMV, tmp, q):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        k_1 = grid.first_interior(Zmin())
        for i in i_uds:
            for k in grid.over_elems(Center()):
                self.W.values[i][k] = 0.0
                self.Area.values[i][k] = 0.0
                self.q_tot.values[i][k] = GMV.q_tot.values[k]
                self.q_liq.values[i][k] = GMV.q_liq.values[k]
                self.q_rai.values[i][k] = GMV.q_rai.values[k]
                self.θ_liq.values[i][k] = GMV.θ_liq.values[k]
                self.T.values[i][k] = GMV.T.values[k]
                self.B.values[i][k] = 0.0
            self.Area.values[i][k_1] = self.updraft_fraction/self.n_updrafts
        self.q_tot.set_bcs(grid)
        self.q_rai.set_bcs(grid)
        self.θ_liq.set_bcs(grid)
        for k in grid.over_elems(Center()):
            for i in i_uds:
                q['a', i][k] = self.Area.values[i][k]
            q['a', i_env][k] = 1.0 - sum([self.Area.values[i][k] for i in i_uds])

        return

    def initialize_io(self, Stats):
        Stats.add_profile('updraft_area')
        Stats.add_profile('updraft_w')
        Stats.add_profile('updraft_q_tot')
        Stats.add_profile('updraft_q_liq')
        Stats.add_profile('updraft_q_rai')
        Stats.add_profile('updraft_θ_liq')
        Stats.add_profile('updraft_temperature')
        Stats.add_profile('updraft_buoyancy')
        Stats.add_ts('updraft_cloud_cover')
        Stats.add_ts('updraft_cloud_base')
        Stats.add_ts('updraft_cloud_top')
        return

    def assign_new_to_values(self, grid):
        for i in range(self.n_updrafts):
            for k in grid.over_elems(Center()):
                self.W.new[i][k] = self.W.values[i][k]
                self.Area.new[i][k] = self.Area.values[i][k]
                self.q_tot.new[i][k] = self.q_tot.values[i][k]
                self.q_liq.new[i][k] = self.q_liq.values[i][k]
                self.q_rai.new[i][k] = self.q_rai.values[i][k]
                self.θ_liq.new[i][k] = self.θ_liq.values[i][k]
                self.T.new[i][k] = self.T.values[i][k]
                self.B.new[i][k] = self.B.values[i][k]
        return

    def set_values_with_new(self, grid):
        for i in range(self.n_updrafts):
            for k in grid.over_elems(Center()):
                self.W.values[i][k] = self.W.new[i][k]
                self.Area.values[i][k] = self.Area.new[i][k]
                self.q_tot.values[i][k] = self.q_tot.new[i][k]
                self.q_liq.values[i][k] = self.q_liq.new[i][k]
                self.q_rai.values[i][k] = self.q_rai.new[i][k]
                self.θ_liq.values[i][k] = self.θ_liq.new[i][k]
                self.T.values[i][k] = self.T.new[i][k]
                self.B.values[i][k] = self.B.new[i][k]
        return

    def io(self, grid, Stats):
        self.get_cloud_base_top_cover(grid)
        Stats.write_ts('updraft_cloud_cover', np.sum(self.cloud_cover))
        Stats.write_ts('updraft_cloud_base' , np.amin(self.cloud_base))
        Stats.write_ts('updraft_cloud_top'  , np.amax(self.cloud_top))
        return

    def get_cloud_base_top_cover(self, grid):
        for i in range(self.n_updrafts):
            self.cloud_base[i] = grid.z_half[grid.nzg-grid.gw-1]
            self.cloud_top[i] = 0.0
            self.cloud_cover[i] = 0.0
            for k in grid.over_elems_real(Center()):
                if self.q_liq.values[i][k] > 1e-8 and self.Area.values[i][k] > 1e-3:
                    self.cloud_base[i] = np.fmin(self.cloud_base[i], grid.z_half[k])
                    self.cloud_top[i] = np.fmax(self.cloud_top[i], grid.z_half[k])
                    self.cloud_cover[i] = np.fmax(self.cloud_cover[i], self.Area.values[i][k])
        return

class UpdraftThermodynamics:
    def __init__(self, n_updrafts, grid, UpdVar):
        self.n_updrafts = n_updrafts
        return

    def satadjust(self, grid, UpdVar, tmp):
        i_gm, i_env, i_uds, i_sd = tmp.domain_idx()
        for i in i_uds:
            for k in grid.over_elems(Center()):
                T, q_liq = eos(tmp['p_0_half'][k], UpdVar.q_tot.values[i][k], UpdVar.θ_liq.values[i][k])
                UpdVar.q_liq.values[i][k] = q_liq
                UpdVar.T.values[i][k] = T
        return

    def buoyancy(self, grid, q, tmp, UpdVar, EnvVar, GMV):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        for i in i_uds:
            for k in grid.over_elems_real(Center()):
                if UpdVar.Area.values[i][k] > 1e-3:
                    q_tot = UpdVar.q_tot.values[i][k]
                    q_vap = q_tot - UpdVar.q_liq.values[i][k]
                    T = UpdVar.T.values[i][k]
                    α_i = alpha_c(tmp['p_0_half'][k], T, q_tot, q_vap)
                    UpdVar.B.values[i][k] = buoyancy_c(tmp['α_0_half'][k], α_i)
                else:
                    UpdVar.B.values[i][k] = EnvVar.B.values[k]
        # Subtract grid mean buoyancy
        for k in grid.over_elems_real(Center()):
            GMV.B.values[k] = q['a', i_env][k] * EnvVar.B.values[k]
            for i in i_uds:
                GMV.B.values[k] += UpdVar.Area.values[i][k] * UpdVar.B.values[i][k]
            for i in i_uds:
                UpdVar.B.values[i][k] -= GMV.B.values[k]
            EnvVar.B.values[k] -= GMV.B.values[k]
        return

class UpdraftMicrophysics:
    def __init__(self, paramlist, n_updrafts, grid):
        self.n_updrafts = n_updrafts
        self.max_supersaturation = paramlist['turbulence']['updraft_microphysics']['max_supersaturation']
        self.prec_src_θ_liq = [Half(grid) for i in range(n_updrafts)]
        self.prec_src_q_tot = [Half(grid) for i in range(n_updrafts)]
        self.prec_src_θ_liq_tot = Half(grid)
        self.prec_src_q_tot_tot = Half(grid)
        return

    def compute_sources(self, grid, UpdVar, tmp):
        i_gm, i_env, i_uds, i_sd = tmp.domain_idx()
        for i in i_uds:
            for k in grid.over_elems(Center()):
                q_tot = UpdVar.q_tot.values[i][k]
                q_tot = UpdVar.q_liq.values[i][k]
                T = UpdVar.T.values[i][k]
                p_0 = tmp['p_0_half'][k]
                tmp_qr = acnv_instant(q_tot, q_tot, self.max_supersaturation, T, p_0)
                self.prec_src_q_tot[i][k] = -tmp_qr
                self.prec_src_θ_liq[i][k] = rain_source_to_thetal(p_0, T, q_tot, q_tot, 0.0, tmp_qr)
        for k in grid.over_elems(Center()):
            self.prec_src_θ_liq_tot[k] = np.sum([self.prec_src_θ_liq[i][k] * UpdVar.Area.values[i][k] for i in i_uds])
            self.prec_src_q_tot_tot[k] = np.sum([self.prec_src_q_tot[i][k] * UpdVar.Area.values[i][k] for i in i_uds])

        return

    def update_updraftvars(self, grid, UpdVar):
        for i in range(self.n_updrafts):
            for k in grid.over_elems(Center()):
                s = self.prec_src_q_tot[i][k]
                UpdVar.q_tot.values[i][k] += s
                UpdVar.q_liq.values[i][k] += s
                UpdVar.q_rai.values[i][k] -= s
                UpdVar.θ_liq.values[i][k] += self.prec_src_θ_liq[i][k]
        return

    def compute_update_combined_local_thetal(self, p_0, T, q_tot, q_liq, q_rai, θ_liq, i, k):

        p_0_k = p_0[k]
        q_tot_k = q_tot[i][k]
        q_liq_k = q_liq[i][k]
        T_k = T[i][k]
        tmp_qr = acnv_instant(q_liq_k, q_tot_k, self.max_supersaturation, T_k, p_0_k)
        s = -tmp_qr
        self.prec_src_q_tot[i][k] = s
        self.prec_src_θ_liq[i][k] = rain_source_to_thetal(p_0_k, T_k, q_tot_k, q_liq_k, 0.0, tmp_qr)
        q_tot[i][k] += s
        q_liq[i][k] += s
        q_rai[i][k] -= s
        θ_liq[i][k] += self.prec_src_θ_liq[i][k]

        return
