import sys
import numpy as np
import pylab as plt
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
from TimeStepping import TimeStepping
from NetCDFIO import NetCDFIO_Stats
from ReferenceState import ReferenceState

from thermodynamic_functions import eos, t_to_entropy_c, t_to_thetali_c, \
    eos_first_guess_thetal, eos_first_guess_entropy, alpha_c, buoyancy_c

class VariablePrognostic:
    def __init__(self, Gr, loc, bc, name, units):
        self.values     = Field.field(Gr, loc, bc)
        self.new        = Field.field(Gr, loc, bc)
        self.tendencies = Field.field(Gr, loc, bc)
        self.name = name
        self.units = units
        return

    def zero_tendencies(self, Gr):
        for k in Gr.over_elems(Center()):
            self.tendencies[k] = 0.0
        return

    def set_bcs(self, Gr):
        self.values.apply_bc(Gr, 0.0)
        self.new.apply_bc(Gr, 0.0)
        return

class VariableDiagnostic:

    def __init__(self, Gr, loc, bc, name, units):
        self.values = Field.field(Gr, loc, bc)
        self.name = name
        self.units = units
        return
    def set_bcs(self, Gr):
        self.values.apply_bc(Gr, 0.0)
        return

class GridMeanVariables:
    def __init__(self, namelist, Gr, Ref):
        self.grid = Gr
        self.Ref = Ref

        self.U = VariablePrognostic(Gr, Center(), Neumann(),'u', 'm/s' )
        self.V = VariablePrognostic(Gr, Center(), Neumann(), 'v', 'm/s' )
        self.W = VariablePrognostic(Gr, Node(), Neumann(), 'v', 'm/s' )
        self.QT = VariablePrognostic(Gr, Center(), Neumann(), 'qt', 'kg/kg')
        self.QR = VariablePrognostic(Gr, Center(), Neumann(), 'qr', 'kg/kg')

        self.H = VariablePrognostic(Gr, Center(), Neumann(),'thetal', 'K')
        self.t_to_prog_fp = t_to_thetali_c
        self.prog_to_t_fp = eos_first_guess_thetal

        # Diagnostic Variables--same class as the prognostic variables, but we append to diagnostics list
        # self.diagnostics_list  = []
        self.QL  = VariableDiagnostic(Gr, Center(), Neumann(), 'ql', 'kg/kg')
        self.T   = VariableDiagnostic(Gr, Center(), Neumann(), 'temperature', 'K')
        self.B   = VariableDiagnostic(Gr, Center(), Neumann(), 'buoyancy', 'm^2/s^3')
        self.THL = VariableDiagnostic(Gr, Center(), Neumann(), 'thetal','K')

        try:
            self.EnvThermo_scheme = str(namelist['thermodynamics']['saturation'])
        except:
            self.EnvThermo_scheme = 'sa_mean'

        self.TKE = VariableDiagnostic(Gr, Center(), Neumann(), 'tke','m^2/s^2' )

        self.QTvar = VariableDiagnostic(Gr, Center(), Neumann(), 'qt_var','kg^2/kg^2' )
        self.Hvar = VariableDiagnostic(Gr, Center(), Neumann() ,'thetal_var', 'K^2')
        self.HQTcov = VariableDiagnostic(Gr, Center(), Neumann() ,'thetal_qt_covar', 'K(kg/kg)' )

        if self.EnvThermo_scheme == 'sommeria_deardorff':
            self.THVvar = VariableDiagnostic(Gr, Center(), Neumann(), 'thatav_var','K^2' )

        return

    def zero_tendencies(self):
        self.U.zero_tendencies(self.grid)
        self.V.zero_tendencies(self.grid)
        self.QT.zero_tendencies(self.grid)
        self.QR.zero_tendencies(self.grid)
        self.H.zero_tendencies(self.grid)
        return

    def update(self, TS):
        for k in self.grid.over_elems_real(Center()):
            self.U.values[k]  +=  self.U.tendencies[k] * TS.dt
            self.V.values[k]  +=  self.V.tendencies[k] * TS.dt
            self.H.values[k]  +=  self.H.tendencies[k] * TS.dt
            self.QT.values[k] +=  self.QT.tendencies[k] * TS.dt
            self.QR.values[k] +=  self.QR.tendencies[k] * TS.dt

        self.U.set_bcs(self.grid)
        self.V.set_bcs(self.grid)
        self.H.set_bcs(self.grid)
        self.QT.set_bcs(self.grid)
        self.QR.set_bcs(self.grid)
        self.TKE.set_bcs(self.grid)
        self.QTvar.set_bcs(self.grid)
        self.Hvar.set_bcs(self.grid)
        self.HQTcov.set_bcs(self.grid)

        if self.EnvThermo_scheme == 'sommeria_deardorff':
            self.THVvar.set_bcs(self.grid)

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
        Stats.write_profile_new('qt_mean'          , self.grid, self.QT.values)
        Stats.write_profile_new('ql_mean'          , self.grid, self.QL.values)
        Stats.write_profile_new('qr_mean'          , self.grid, self.QR.values)
        Stats.write_profile_new('temperature_mean' , self.grid, self.T.values)
        Stats.write_profile_new('buoyancy_mean'    , self.grid, self.B.values)
        Stats.write_profile_new('thetal_mean'      , self.grid, self.H.values)
        Stats.write_profile_new('tke_mean'         , self.grid, self.TKE.values)
        Stats.write_profile_new('Hvar_mean'        , self.grid, self.Hvar.values)
        Stats.write_profile_new('QTvar_mean'       , self.grid, self.QTvar.values)
        Stats.write_profile_new('HQTcov_mean'      , self.grid, self.HQTcov.values)

        for k in self.grid.over_elems_real(Center()):
            lwp += tmp['ρ_0_half'][k]*self.QL.values[k]*self.grid.dz
        Stats.write_ts('lwp', lwp)

        return

    def satadjust(self, tmp):
        for k in self.grid.over_elems(Center()):
            h = self.H.values[k]
            qt = self.QT.values[k]
            p0 = tmp['p_0_half'][k]
            T, ql = eos(self.t_to_prog_fp, self.prog_to_t_fp, p0, qt, h )
            self.QL.values[k] = ql
            self.T.values[k] = T
            qv = qt - ql
            self.THL.values[k] = t_to_thetali_c(p0, T, qt, ql, 0.0)
            alpha = alpha_c(p0, T, qt, qv)
            self.B.values[k] = buoyancy_c(tmp['α_0_half'][k], alpha)
        return
