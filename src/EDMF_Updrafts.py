import numpy as np
from parameters import *
from funcs_thermo import  *
from funcs_micro import  *
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
from Variables import GridMeanVariables
from NetCDFIO import NetCDFIO_Stats
from EDMF_Environment import EnvironmentVariables
import pylab as plt


class UpdraftVariable:
    def __init__(self, grid, nu, loc, bc):
        self.values     = [Field.field(grid, loc, bc) for i in range(nu)]
        self.old        = [Field.field(grid, loc, bc) for i in range(nu)]
        self.new        = [Field.field(grid, loc, bc) for i in range(nu)]
        self.tendencies = [Field.field(grid, loc, bc) for i in range(nu)]

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
        self.θ_liq = UpdraftVariable(grid, nu, Center(), Neumann())
        self.T     = UpdraftVariable(grid, nu, Center(), Neumann())
        self.B     = UpdraftVariable(grid, nu, Center(), Neumann())
        self.H     = UpdraftVariable(grid, nu, Center(), Neumann())

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
                self.H.values[i][k] = GMV.H.values[k]
                self.T.values[i][k] = GMV.T.values[k]
                self.B.values[i][k] = 0.0
            self.Area.values[i][k_1] = self.updraft_fraction/self.n_updrafts
        self.q_tot.set_bcs(grid)
        self.q_rai.set_bcs(grid)
        self.H.set_bcs(grid)
        for k in grid.over_elems(Center()):
            for i in i_uds:
                q['a', i][k] = self.Area.values[i][k]
            q['a', i_env][k] = 1.0 - sum([self.Area.values[i][k] for i in i_uds])

        return

    def initialize_io(self, Stats):
        Stats.add_profile('updraft_area')
        Stats.add_profile('updraft_w')
        Stats.add_profile('updraft_qt')
        Stats.add_profile('updraft_ql')
        Stats.add_profile('updraft_qr')
        Stats.add_profile('updraft_thetal')
        Stats.add_profile('updraft_temperature')
        Stats.add_profile('updraft_buoyancy')
        Stats.add_ts('updraft_cloud_cover')
        Stats.add_ts('updraft_cloud_base')
        Stats.add_ts('updraft_cloud_top')
        return

    def set_new_with_values(self, grid):
        for i in range(self.n_updrafts):
            for k in grid.over_elems(Center()):
                self.W.new[i][k] = self.W.values[i][k]
                self.Area.new[i][k] = self.Area.values[i][k]
                self.q_tot.new[i][k] = self.q_tot.values[i][k]
                self.q_liq.new[i][k] = self.q_liq.values[i][k]
                self.q_rai.new[i][k] = self.q_rai.values[i][k]
                self.H.new[i][k] = self.H.values[i][k]
                self.θ_liq.new[i][k] = self.θ_liq.values[i][k]
                self.T.new[i][k] = self.T.values[i][k]
                self.B.new[i][k] = self.B.values[i][k]
        return

    def set_old_with_values(self, grid):
        for i in range(self.n_updrafts):
            for k in grid.over_elems(Center()):
                self.W.old[i][k] = self.W.values[i][k]
                self.Area.old[i][k] = self.Area.values[i][k]
                self.q_tot.old[i][k] = self.q_tot.values[i][k]
                self.q_liq.old[i][k] = self.q_liq.values[i][k]
                self.q_rai.old[i][k] = self.q_rai.values[i][k]
                self.H.old[i][k] = self.H.values[i][k]
                self.θ_liq.old[i][k] = self.θ_liq.values[i][k]
                self.T.old[i][k] = self.T.values[i][k]
                self.B.old[i][k] = self.B.values[i][k]
        return

    def set_values_with_new(self, grid):
        for i in range(self.n_updrafts):
            for k in grid.over_elems(Center()):
                self.W.values[i][k] = self.W.new[i][k]
                self.Area.values[i][k] = self.Area.new[i][k]
                self.q_tot.values[i][k] = self.q_tot.new[i][k]
                self.q_liq.values[i][k] = self.q_liq.new[i][k]
                self.q_rai.values[i][k] = self.q_rai.new[i][k]
                self.H.values[i][k] = self.H.new[i][k]
                self.θ_liq.values[i][k] = self.θ_liq.new[i][k]
                self.T.values[i][k] = self.T.new[i][k]
                self.B.values[i][k] = self.B.new[i][k]
        return

    def io(self, grid, Stats):
        self.get_cloud_base_top_cover(grid)
        Stats.write_ts('updraft_cloud_cover', np.sum(self.cloud_cover))
        Stats.write_ts('updraft_cloud_base', np.amin(self.cloud_base))
        Stats.write_ts('updraft_cloud_top', np.amax(self.cloud_top))
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
                T, ql = eos(tmp['p_0_half'][k], UpdVar.q_tot.values[i][k], UpdVar.H.values[i][k])
                UpdVar.q_liq.values[i][k] = ql
                UpdVar.T.values[i][k] = T
        return

    def buoyancy(self, grid, q, tmp, UpdVar, EnvVar, GMV, extrap):
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
        self.prec_source_h = [Half(grid) for i in range(n_updrafts)]
        self.prec_source_qt = [Half(grid) for i in range(n_updrafts)]
        self.prec_source_h_tot  = Half(grid)
        self.prec_source_qt_tot = Half(grid)
        return

    def compute_sources(self, grid, UpdVar, tmp):
        i_gm, i_env, i_uds, i_sd = tmp.domain_idx()
        for i in i_uds:
            for k in grid.over_elems(Center()):
                tmp_qr = acnv_instant(UpdVar.q_liq.values[i][k], UpdVar.q_tot.values[i][k], self.max_supersaturation,\
                                      UpdVar.T.values[i][k], tmp['p_0_half'][k])
                self.prec_source_qt[i][k] = -tmp_qr
                self.prec_source_h[i][k]  = rain_source_to_thetal(tmp['p_0_half'][k], UpdVar.T.values[i][k],\
                                             UpdVar.q_tot.values[i][k], UpdVar.q_liq.values[i][k], 0.0, tmp_qr)
        for k in grid.over_elems(Center()):
            self.prec_source_h_tot[k]  = np.sum([self.prec_source_h[i][k] * UpdVar.Area.values[i][k] for i in i_uds])
            self.prec_source_qt_tot[k] = np.sum([self.prec_source_qt[i][k]* UpdVar.Area.values[i][k] for i in i_uds])

        return

    def update_updraftvars(self, grid, UpdVar):
        for i in range(self.n_updrafts):
            for k in grid.over_elems(Center()):
                s = self.prec_source_qt[i][k]
                UpdVar.q_tot.values[i][k] += s
                UpdVar.q_liq.values[i][k] += s
                UpdVar.q_rai.values[i][k] -= s
                UpdVar.H.values[i][k] += self.prec_source_h[i][k]
        return

    def compute_update_combined_local_thetal(self, p0, T, qt, ql, qr, h, i, k):

        tmp_qr = acnv_instant(ql[i][k], qt[i][k], self.max_supersaturation, T[i][k], p0[k])
        self.prec_source_qt[i][k] = -tmp_qr
        self.prec_source_h[i][k]  = rain_source_to_thetal(p0[k], T[i][k], qt[i][k], ql[i][k], 0.0, tmp_qr)
        s = self.prec_source_qt[i][k]
        qt[i][k] += s
        ql[i][k] += s
        qr[i][k] -= s
        h[i][k]  += self.prec_source_h[i][k]

        return
