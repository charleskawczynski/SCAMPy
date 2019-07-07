import numpy as np
import sys
from parameters import *
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann, nice_name
from TimeStepping import TimeStepping
from Variables import VariableDiagnostic, GridMeanVariables
from funcs_thermo import  *
from funcs_micro import *

class EnvironmentVariable:
    def __init__(self, grid, loc, bc):
        self.values = Field.field(grid, loc, bc)

    def set_bcs(self, grid):
        self.values.apply_bc(grid, 0.0)
        return

class EnvironmentVariable_2m:
    def __init__(self, grid, loc, bc):
        self.values      = Field.field(grid, loc, bc)
        self.dissipation = Field.field(grid, loc, bc)
        self.entr_gain   = Field.field(grid, loc, bc)
        self.detr_loss   = Field.field(grid, loc, bc)
        self.buoy        = Field.field(grid, loc, bc)
        self.press       = Field.field(grid, loc, bc)
        self.shear       = Field.field(grid, loc, bc)
        self.interdomain = Field.field(grid, loc, bc)
        self.rain_src    = Field.field(grid, loc, bc)

    def set_bcs(self, grid):
        self.values.apply_bc(grid, 0.0)
        return

class EnvironmentVariables:
    def __init__(self,  namelist, grid):
        self.W              = EnvironmentVariable(grid   , Node()   , Dirichlet())
        self.q_tot          = EnvironmentVariable(grid   , Center() , Neumann())
        self.q_liq          = EnvironmentVariable(grid   , Center() , Neumann())
        self.q_rai          = EnvironmentVariable(grid   , Center() , Neumann())
        self.θ_liq          = EnvironmentVariable(grid   , Center() , Neumann())
        self.T              = EnvironmentVariable(grid   , Center() , Neumann())
        self.B              = EnvironmentVariable(grid   , Center() , Neumann())
        self.CF             = EnvironmentVariable(grid   , Center() , Neumann())
        self.H              = EnvironmentVariable(grid   , Center() , Neumann())
        self.tke            = EnvironmentVariable_2m(grid, Center() , Neumann())
        self.cv_q_tot       = EnvironmentVariable_2m(grid, Center() , Neumann())
        self.cv_θ_liq       = EnvironmentVariable_2m(grid, Center() , Neumann())
        self.cv_θ_liq_q_tot = EnvironmentVariable_2m(grid, Center() , Neumann())
        return

    def initialize_io(self, Stats):
        Stats.add_profile('env_w')
        Stats.add_profile('env_qt')
        Stats.add_profile('env_ql')
        Stats.add_profile('env_qr')
        Stats.add_profile('env_thetal')
        Stats.add_profile('env_temperature')
        Stats.add_profile('env_tke')
        Stats.add_profile('env_Hvar')
        Stats.add_profile('env_QTvar')
        Stats.add_profile('env_HQTcov')
        return

    def io(self, grid, Stats):
        Stats.write_profile_new('env_w'           , grid, self.W.values)
        Stats.write_profile_new('env_qt'          , grid, self.q_tot.values)
        Stats.write_profile_new('env_ql'          , grid, self.q_liq.values)
        Stats.write_profile_new('env_qr'          , grid, self.q_rai.values)
        Stats.write_profile_new('env_thetal'      , grid, self.H.values)
        Stats.write_profile_new('env_temperature' , grid, self.T.values)
        Stats.write_profile_new('env_tke'         , grid, self.tke.values)
        Stats.write_profile_new('env_Hvar'        , grid, self.cv_θ_liq.values)
        Stats.write_profile_new('env_QTvar'       , grid, self.cv_q_tot.values)
        Stats.write_profile_new('env_HQTcov'      , grid, self.cv_θ_liq_q_tot.values)
        return

class EnvironmentThermodynamics:
    def __init__(self, namelist, paramlist, grid, EnvVar):
        self.qt_dry         = Half(grid)
        self.th_dry         = Half(grid)
        self.t_cloudy       = Half(grid)
        self.qv_cloudy      = Half(grid)
        self.qt_cloudy      = Half(grid)
        self.th_cloudy      = Half(grid)
        self.cv_θ_liq_rain_dt   = Half(grid)
        self.cv_q_tot_rain_dt  = Half(grid)
        self.cv_θ_liq_q_tot_rain_dt = Half(grid)
        self.max_supersaturation = paramlist['turbulence']['updraft_microphysics']['max_supersaturation']
        return

    def update_EnvVar(self, tmp, k, EnvVar, T, H, qt, ql, qr, alpha):
        EnvVar.T.values[k]   = T
        EnvVar.θ_liq.values[k] = H
        EnvVar.H.values[k]   = H
        EnvVar.q_tot.values[k]  = qt
        EnvVar.q_liq.values[k]  = ql
        EnvVar.q_rai.values[k] += qr
        EnvVar.B.values[k]   = buoyancy_c(tmp['α_0_half'][k], alpha)
        return

    def update_cloud_dry(self, k, EnvVar, T, th, qt, ql, qv):
        if ql > 0.0:
            EnvVar.CF.values[k] = 1.
            self.th_cloudy[k]   = th
            self.t_cloudy[k]    = T
            self.qt_cloudy[k]   = qt
            self.qv_cloudy[k]   = qv
        else:
            EnvVar.CF.values[k] = 0.
            self.th_dry[k]      = th
            self.qt_dry[k]      = qt
        return

    def eos_update_SA_mean(self, grid, EnvVar, in_Env, tmp):
        for k in grid.over_elems_real(Center()):
            p_0_k = tmp['p_0_half'][k]
            T, ql  = eos(p_0_k, EnvVar.q_tot.values[k], EnvVar.H.values[k])
            mph = microphysics(T, ql, p_0_k, EnvVar.q_tot.values[k], self.max_supersaturation, in_Env)
            self.update_EnvVar(tmp,   k, EnvVar, mph.T, mph.thl, mph.qt, mph.ql, mph.qr, mph.alpha)
            self.update_cloud_dry(k, EnvVar, mph.T, mph.th,  mph.qt, mph.ql, mph.qv)
        return
