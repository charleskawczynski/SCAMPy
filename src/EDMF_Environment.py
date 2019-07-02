import numpy as np
import sys
from parameters import *
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
from TimeStepping import TimeStepping
from ReferenceState import ReferenceState
from Variables import VariableDiagnostic, GridMeanVariables
from thermodynamic_functions import  *
from microphysics_functions import *

class EnvironmentVariable:
    def __init__(self, Gr, loc, bc, name, units):
        self.values = Field.field(Gr, loc, bc)
        self.name = name
        self.units = units

    def set_bcs(self, Gr):
        self.values.apply_bc(Gr, 0.0)
        return

class EnvironmentVariable_2m:
    def __init__(self, Gr, loc, bc, name, units):
        self.values      = Field.field(Gr, loc, bc)
        self.dissipation = Field.field(Gr, loc, bc)
        self.entr_gain   = Field.field(Gr, loc, bc)
        self.detr_loss   = Field.field(Gr, loc, bc)
        self.buoy        = Field.field(Gr, loc, bc)
        self.press       = Field.field(Gr, loc, bc)
        self.shear       = Field.field(Gr, loc, bc)
        self.interdomain = Field.field(Gr, loc, bc)
        self.rain_src    = Field.field(Gr, loc, bc)
        self.name = name
        self.units = units

    def set_bcs(self, Gr):
        self.values.apply_bc(Gr, 0.0)
        return

class EnvironmentVariables:
    def __init__(self,  namelist, Gr  ):
        self.grid = Gr
        self.EnvThermo_scheme = str(namelist['thermodynamics']['saturation'])
        self.W              = EnvironmentVariable(Gr, Node(), Dirichlet(), 'w','m/s' )
        self.q_tot          = EnvironmentVariable(Gr, Center(), Neumann(), 'qt','kg/kg' )
        self.q_liq          = EnvironmentVariable(Gr, Center(), Neumann(), 'ql','kg/kg' )
        self.q_rai          = EnvironmentVariable(Gr, Center(), Neumann(), 'qr','kg/kg' )
        self.θ_liq          = EnvironmentVariable(Gr, Center(), Neumann(), 'thetal', 'K')
        self.T              = EnvironmentVariable(Gr, Center(), Neumann(), 'temperature','K' )
        self.B              = EnvironmentVariable(Gr, Center(), Neumann(), 'buoyancy','m^2/s^3' )
        self.CF             = EnvironmentVariable(Gr, Center(), Neumann(),'cloud_fraction', '-')
        self.H              = EnvironmentVariable(Gr, Center(), Neumann(), 'thetal','K' )
        self.tke            = EnvironmentVariable_2m(Gr, Center(), Neumann(), 'tke','m^2/s^2' )
        self.cv_q_tot       = EnvironmentVariable_2m(Gr, Center(), Neumann(), 'qt_var','kg^2/kg^2' )
        self.cv_θ_liq       = EnvironmentVariable_2m(Gr, Center(), Neumann(), 'thetal_var', 'K^2')
        self.cv_θ_liq_q_tot = EnvironmentVariable_2m(Gr, Center(), Neumann(), 'thetal_qt_covar', 'K(kg/kg)' )
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

    def io(self, Stats):
        Stats.write_profile_new('env_w'           , self.grid, self.W.values)
        Stats.write_profile_new('env_qt'          , self.grid, self.q_tot.values)
        Stats.write_profile_new('env_ql'          , self.grid, self.q_liq.values)
        Stats.write_profile_new('env_qr'          , self.grid, self.q_rai.values)
        Stats.write_profile_new('env_thetal'      , self.grid, self.H.values)
        Stats.write_profile_new('env_temperature' , self.grid, self.T.values)
        Stats.write_profile_new('env_tke'         , self.grid, self.tke.values)
        Stats.write_profile_new('env_Hvar'        , self.grid, self.cv_θ_liq.values)
        Stats.write_profile_new('env_QTvar'       , self.grid, self.cv_q_tot.values)
        Stats.write_profile_new('env_HQTcov'      , self.grid, self.cv_θ_liq_q_tot.values)
        return

class EnvironmentThermodynamics:
    def __init__(self, namelist, paramlist, Gr, Ref, EnvVar):
        self.grid = Gr
        self.Ref = Ref
        self.t_to_prog_fp = t_to_thetali_c
        self.prog_to_t_fp = eos_first_guess_thetal
        self.qt_dry         = Half(Gr)
        self.th_dry         = Half(Gr)
        self.t_cloudy       = Half(Gr)
        self.qv_cloudy      = Half(Gr)
        self.qt_cloudy      = Half(Gr)
        self.th_cloudy      = Half(Gr)
        self.Hvar_rain_dt   = Half(Gr)
        self.QTvar_rain_dt  = Half(Gr)
        self.HQTcov_rain_dt = Half(Gr)
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

    def eos_update_SA_mean(self, EnvVar, in_Env, tmp):
        for k in self.grid.over_elems_real(Center()):
            p_0_k = tmp['p_0_half'][k]
            T, ql  = eos(self.t_to_prog_fp, self.prog_to_t_fp, p_0_k, EnvVar.q_tot.values[k], EnvVar.H.values[k])
            mph = microphysics(T, ql, p_0_k, EnvVar.q_tot.values[k], self.max_supersaturation, in_Env)
            self.update_EnvVar(tmp,   k, EnvVar, mph.T, mph.thl, mph.qt, mph.ql, mph.qr, mph.alpha)
            self.update_cloud_dry(k, EnvVar, mph.T, mph.th,  mph.qt, mph.ql, mph.qv)
        return

    def satadjust(self, EnvVar, in_Env, tmp):
        self.eos_update_SA_mean(EnvVar, in_Env, tmp)
        return
