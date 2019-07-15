import sys
import numpy as np
import pylab as plt
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
from TimeStepping import TimeStepping
from NetCDFIO import NetCDFIO_Stats

from funcs_thermo import eos, t_to_entropy_c, t_to_thetali_c, \
    alpha_c, buoyancy_c

def satadjust(grid, q, tmp, GMV):
    i_gm, i_env, i_uds, i_sd = q.domain_idx()
    for k in grid.over_elems(Center()):
        θ_liq = q['θ_liq', i_gm][k]
        q_tot = q['q_tot', i_gm][k]
        p_0 = tmp['p_0_half'][k]
        T, q_liq = eos(p_0, q_tot, θ_liq)
        tmp['q_liq', i_gm][k] = q_liq
        tmp['T', i_gm][k] = T
        q_vap = q_tot - q_liq
        alpha = alpha_c(p_0, T, q_tot, q_vap)
        tmp['B', i_gm][k] = buoyancy_c(tmp['α_0_half'][k], alpha)
    return

class VariablePrognostic:
    def __init__(self, grid, loc, bc):
        self.values     = Field.field(grid, loc, bc)
        self.new        = Field.field(grid, loc, bc)
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

    def initialize_io(self, Stats):
        Stats.add_ts('lwp')
        return

    def export_data(self, grid, Stats, tmp):
        lwp = 0.0
        for k in grid.over_elems_real(Center()):
            lwp += tmp['ρ_0_half'][k]*self.q_liq.values[k]*grid.dz
        Stats.write_ts('lwp', lwp)

        return
