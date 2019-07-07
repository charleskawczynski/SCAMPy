import sys
import numpy as np
import pylab as plt
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
from TimeStepping import TimeStepping
from NetCDFIO import NetCDFIO_Stats

from funcs_thermo import eos, t_to_entropy_c, t_to_thetali_c, \
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
    def __init__(self, namelist, grid):
        self.U              = VariablePrognostic(grid, Center(), Neumann())
        self.V              = VariablePrognostic(grid, Center(), Neumann())
        self.W              = VariablePrognostic(grid, Node()  , Neumann())
        self.q_tot          = VariablePrognostic(grid, Center(), Neumann())
        self.q_rai          = VariablePrognostic(grid, Center(), Neumann())
        self.θ_liq          = VariablePrognostic(grid, Center(), Neumann())
        self.q_liq          = VariableDiagnostic(grid, Center(), Neumann())
        self.T              = VariableDiagnostic(grid, Center(), Neumann())
        self.B              = VariableDiagnostic(grid, Center(), Neumann())
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
        self.θ_liq.zero_tendencies(grid)
        return

    def update(self, grid, TS):
        for k in grid.over_elems_real(Center()):
            self.U.values[k]  +=  self.U.tendencies[k] * TS.dt
            self.V.values[k]  +=  self.V.tendencies[k] * TS.dt
            self.θ_liq.values[k]  +=  self.θ_liq.tendencies[k] * TS.dt
            self.q_tot.values[k] +=  self.q_tot.tendencies[k] * TS.dt
            self.q_rai.values[k] +=  self.q_rai.tendencies[k] * TS.dt

        self.U.set_bcs(grid)
        self.V.set_bcs(grid)
        self.θ_liq.set_bcs(grid)
        self.q_tot.set_bcs(grid)
        self.q_rai.set_bcs(grid)
        self.tke.set_bcs(grid)
        self.cv_q_tot.set_bcs(grid)
        self.cv_θ_liq.set_bcs(grid)
        self.cv_θ_liq_q_tot.set_bcs(grid)
        self.zero_tendencies(grid)
        return

    def initialize_io(self, Stats):
        Stats.add_profile('U_mean')
        Stats.add_profile('V_mean')
        Stats.add_profile('q_tot_mean')
        Stats.add_profile('q_liq_mean')
        Stats.add_profile('q_rai_mean')
        Stats.add_profile('T_mean')
        Stats.add_profile('B_mean')
        Stats.add_profile('θ_liq_mean')
        Stats.add_profile('tke_mean')
        Stats.add_profile('cv_θ_liq_mean')
        Stats.add_profile('cv_q_tot_mean')
        Stats.add_profile('cv_θ_liq_q_tot_mean')
        Stats.add_ts('lwp')
        return

    def io(self, grid, Stats, tmp):
        lwp = 0.0
        Stats.write_profile_new('U_mean'              , grid, self.U.values)
        Stats.write_profile_new('V_mean'              , grid, self.V.values)
        Stats.write_profile_new('q_tot_mean'          , grid, self.q_tot.values)
        Stats.write_profile_new('q_liq_mean'          , grid, self.q_liq.values)
        Stats.write_profile_new('q_rai_mean'          , grid, self.q_rai.values)
        Stats.write_profile_new('T_mean'              , grid, self.T.values)
        Stats.write_profile_new('B_mean'              , grid, self.B.values)
        Stats.write_profile_new('θ_liq_mean'          , grid, self.θ_liq.values)
        Stats.write_profile_new('tke_mean'            , grid, self.tke.values)
        Stats.write_profile_new('cv_θ_liq_mean'       , grid, self.cv_θ_liq.values)
        Stats.write_profile_new('cv_q_tot_mean'       , grid, self.cv_q_tot.values)
        Stats.write_profile_new('cv_θ_liq_q_tot_mean' , grid, self.cv_θ_liq_q_tot.values)

        for k in grid.over_elems_real(Center()):
            lwp += tmp['ρ_0_half'][k]*self.q_liq.values[k]*grid.dz
        Stats.write_ts('lwp', lwp)

        return

    def satadjust(self, grid, tmp):
        for k in grid.over_elems(Center()):
            θ_liq = self.θ_liq.values[k]
            q_tot = self.q_tot.values[k]
            p_0 = tmp['p_0_half'][k]
            T, q_liq = eos(p_0, q_tot, θ_liq)
            self.q_liq.values[k] = q_liq
            self.T.values[k] = T
            q_vap = q_tot - q_liq
            alpha = alpha_c(p_0, T, q_tot, q_vap)
            self.B.values[k] = buoyancy_c(tmp['α_0_half'][k], alpha)
        return
