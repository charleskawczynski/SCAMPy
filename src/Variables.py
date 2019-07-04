import sys
import numpy as np
import pylab as plt
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
from TimeStepping import TimeStepping
from NetCDFIO import NetCDFIO_Stats
from ReferenceState import ReferenceState

from thermodynamic_functions import eos, t_to_entropy_c, t_to_thetali_c, \
    alpha_c, buoyancy_c

class VariablePrognostic:
    def __init__(self, grid, loc, bc):
        self.values     = Field.field(grid, loc, bc)
        self.new        = Field.field(grid, loc, bc)
        self.tendencies = Field.field(grid, loc, bc)
        return

    def zero_tendencies(self, grid):
        for k in grid.over_elems(Center()):
            self.tendencies[k] = 0.0
        return

    def set_bcs(self, grid):
        self.values.apply_bc(grid, 0.0)
        self.new.apply_bc(grid, 0.0)
        return

class VariableDiagnostic:

    def __init__(self, grid, loc, bc):
        self.values = Field.field(grid, loc, bc)
        return
    def set_bcs(self, grid):
        self.values.apply_bc(grid, 0.0)
        return

class GridMeanVariables:
    def __init__(self, namelist, grid, Ref):
        self.Ref = Ref

        self.U              = VariablePrognostic(grid, Center(), Neumann())
        self.V              = VariablePrognostic(grid, Center(), Neumann())
        self.W              = VariablePrognostic(grid, Node()  , Neumann())
        self.q_tot          = VariablePrognostic(grid, Center(), Neumann())
        self.q_rai          = VariablePrognostic(grid, Center(), Neumann())
        self.H              = VariablePrognostic(grid, Center(), Neumann())
        self.q_liq          = VariableDiagnostic(grid, Center(), Neumann())
        self.T              = VariableDiagnostic(grid, Center(), Neumann())
        self.B              = VariableDiagnostic(grid, Center(), Neumann())
        self.θ_liq          = VariableDiagnostic(grid, Center(), Neumann())
        self.tke            = VariableDiagnostic(grid, Center(), Neumann())
        self.cv_q_tot       = VariableDiagnostic(grid, Center(), Neumann())
        self.cv_θ_liq       = VariableDiagnostic(grid, Center(), Neumann())
        self.cv_θ_liq_q_tot = VariableDiagnostic(grid, Center(), Neumann())

        return

    def zero_tendencies(self, grid):
        self.U.zero_tendencies(grid)
        self.V.zero_tendencies(grid)
        self.q_tot.zero_tendencies(grid)
        self.q_rai.zero_tendencies(grid)
        self.H.zero_tendencies(grid)
        return

    def update(self, grid, TS):
        for k in grid.over_elems_real(Center()):
            self.U.values[k]  +=  self.U.tendencies[k] * TS.dt
            self.V.values[k]  +=  self.V.tendencies[k] * TS.dt
            self.H.values[k]  +=  self.H.tendencies[k] * TS.dt
            self.q_tot.values[k] +=  self.q_tot.tendencies[k] * TS.dt
            self.q_rai.values[k] +=  self.q_rai.tendencies[k] * TS.dt

        self.U.set_bcs(grid)
        self.V.set_bcs(grid)
        self.H.set_bcs(grid)
        self.q_tot.set_bcs(grid)
        self.q_rai.set_bcs(grid)
        self.tke.set_bcs(grid)
        self.cv_q_tot.set_bcs(grid)
        self.cv_θ_liq.set_bcs(grid)
        self.cv_θ_liq_q_tot.set_bcs(grid)
        self.zero_tendencies(grid)
        return

    def initialize_io(self, Stats):
        Stats.add_profile('u_mean')
        Stats.add_profile('v_mean')
        Stats.add_profile('qt_mean')
        Stats.add_profile('qr_mean')
        Stats.add_profile('thetal_mean')
        Stats.add_profile('temperature_mean')
        Stats.add_profile('buoyancy_mean')
        Stats.add_profile('ql_mean')
        Stats.add_profile('tke_mean')
        Stats.add_profile('Hvar_mean')
        Stats.add_profile('QTvar_mean')
        Stats.add_profile('HQTcov_mean')
        Stats.add_ts('lwp')
        return

    def io(self, grid, Stats, tmp):
        lwp = 0.0
        Stats.write_profile_new('u_mean'           , grid, self.U.values)
        Stats.write_profile_new('v_mean'           , grid, self.V.values)
        Stats.write_profile_new('qt_mean'          , grid, self.q_tot.values)
        Stats.write_profile_new('ql_mean'          , grid, self.q_liq.values)
        Stats.write_profile_new('qr_mean'          , grid, self.q_rai.values)
        Stats.write_profile_new('temperature_mean' , grid, self.T.values)
        Stats.write_profile_new('buoyancy_mean'    , grid, self.B.values)
        Stats.write_profile_new('thetal_mean'      , grid, self.H.values)
        Stats.write_profile_new('tke_mean'         , grid, self.tke.values)
        Stats.write_profile_new('Hvar_mean'        , grid, self.cv_θ_liq.values)
        Stats.write_profile_new('QTvar_mean'       , grid, self.cv_q_tot.values)
        Stats.write_profile_new('HQTcov_mean'      , grid, self.cv_θ_liq_q_tot.values)

        for k in grid.over_elems_real(Center()):
            lwp += tmp['ρ_0_half'][k]*self.q_liq.values[k]*grid.dz
        Stats.write_ts('lwp', lwp)

        return

    def satadjust(self, grid, tmp):
        for k in grid.over_elems(Center()):
            h = self.H.values[k]
            qt = self.q_tot.values[k]
            p0 = tmp['p_0_half'][k]
            T, ql = eos(p0, qt, h)
            self.q_liq.values[k] = ql
            self.T.values[k] = T
            qv = qt - ql
            self.θ_liq.values[k] = t_to_thetali_c(p0, T, qt, ql, 0.0)
            alpha = alpha_c(p0, T, qt, qv)
            self.B.values[k] = buoyancy_c(tmp['α_0_half'][k], alpha)
        return
