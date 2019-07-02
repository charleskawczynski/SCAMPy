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
    def __init__(self, grid, loc, bc, name):
        self.values     = Field.field(grid, loc, bc)
        self.new        = Field.field(grid, loc, bc)
        self.tendencies = Field.field(grid, loc, bc)
        self.name = name
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

    def __init__(self, grid, loc, bc, name):
        self.values = Field.field(grid, loc, bc)
        self.name = name
        return
    def set_bcs(self, grid):
        self.values.apply_bc(grid, 0.0)
        return

class GridMeanVariables:
    def __init__(self, namelist, grid, Ref):
        self.grid = grid
        self.Ref = Ref

        self.U              = VariablePrognostic(grid, Center(), Neumann(), 'U')
        self.V              = VariablePrognostic(grid, Center(), Neumann(), 'V')
        self.W              = VariablePrognostic(grid, Node()  , Neumann(), 'W')
        self.q_tot          = VariablePrognostic(grid, Center(), Neumann(), 'q_tot')
        self.q_rai          = VariablePrognostic(grid, Center(), Neumann(), 'q_rai')
        self.H              = VariablePrognostic(grid, Center(), Neumann(), 'H')
        self.q_liq          = VariableDiagnostic(grid, Center(), Neumann(), 'q_liq')
        self.T              = VariableDiagnostic(grid, Center(), Neumann(), 'T')
        self.B              = VariableDiagnostic(grid, Center(), Neumann(), 'B')
        self.θ_liq          = VariableDiagnostic(grid, Center(), Neumann(), 'θ_liq')
        self.tke            = VariableDiagnostic(grid, Center(), Neumann(), 'tke')
        self.cv_q_tot       = VariableDiagnostic(grid, Center(), Neumann(), 'cv_q_tot')
        self.cv_θ_liq       = VariableDiagnostic(grid, Center(), Neumann(), 'cv_θ_liq')
        self.cv_θ_liq_q_tot = VariableDiagnostic(grid, Center(), Neumann(), 'cv_θ_liq_q_tot')

        return

    def zero_tendencies(self):
        self.U.zero_tendencies(self.grid)
        self.V.zero_tendencies(self.grid)
        self.q_tot.zero_tendencies(self.grid)
        self.q_rai.zero_tendencies(self.grid)
        self.H.zero_tendencies(self.grid)
        return

    def update(self, TS):
        for k in self.grid.over_elems_real(Center()):
            self.U.values[k]  +=  self.U.tendencies[k] * TS.dt
            self.V.values[k]  +=  self.V.tendencies[k] * TS.dt
            self.H.values[k]  +=  self.H.tendencies[k] * TS.dt
            self.q_tot.values[k] +=  self.q_tot.tendencies[k] * TS.dt
            self.q_rai.values[k] +=  self.q_rai.tendencies[k] * TS.dt

        self.U.set_bcs(self.grid)
        self.V.set_bcs(self.grid)
        self.H.set_bcs(self.grid)
        self.q_tot.set_bcs(self.grid)
        self.q_rai.set_bcs(self.grid)
        self.tke.set_bcs(self.grid)
        self.cv_q_tot.set_bcs(self.grid)
        self.cv_θ_liq.set_bcs(self.grid)
        self.cv_θ_liq_q_tot.set_bcs(self.grid)
        self.zero_tendencies()
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

    def io(self, Stats, tmp):
        lwp = 0.0
        Stats.write_profile_new('u_mean'           , self.grid, self.U.values)
        Stats.write_profile_new('v_mean'           , self.grid, self.V.values)
        Stats.write_profile_new('qt_mean'          , self.grid, self.q_tot.values)
        Stats.write_profile_new('ql_mean'          , self.grid, self.q_liq.values)
        Stats.write_profile_new('qr_mean'          , self.grid, self.q_rai.values)
        Stats.write_profile_new('temperature_mean' , self.grid, self.T.values)
        Stats.write_profile_new('buoyancy_mean'    , self.grid, self.B.values)
        Stats.write_profile_new('thetal_mean'      , self.grid, self.H.values)
        Stats.write_profile_new('tke_mean'         , self.grid, self.tke.values)
        Stats.write_profile_new('Hvar_mean'        , self.grid, self.cv_θ_liq.values)
        Stats.write_profile_new('QTvar_mean'       , self.grid, self.cv_q_tot.values)
        Stats.write_profile_new('HQTcov_mean'      , self.grid, self.cv_θ_liq_q_tot.values)

        for k in self.grid.over_elems_real(Center()):
            lwp += tmp['ρ_0_half'][k]*self.q_liq.values[k]*self.grid.dz
        Stats.write_ts('lwp', lwp)

        return

    def satadjust(self, tmp):
        for k in self.grid.over_elems(Center()):
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
