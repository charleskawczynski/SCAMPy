import sys
import numpy as np
import pylab as plt
from Grid import Grid
from Field import Field
from TimeStepping import TimeStepping
from NetCDFIO import NetCDFIO_Stats
from ReferenceState import ReferenceState

from thermodynamic_functions import eos, t_to_entropy_c, t_to_thetali_c, \
    eos_first_guess_thetal, eos_first_guess_entropy, alpha_c, buoyancy_c

class VariablePrognostic:
    def __init__(self, Gr, loc, kind, bc, name, units):
        # Value at the current and next timestep, used for calculating turbulence tendencies
        self.values     = Field.field(Gr, loc)
        self.new        = Field.field(Gr, loc)
        self.mf_update  = Field.field(Gr, loc)
        self.tendencies = Field.field(Gr, loc)
        # Placement on staggered grid
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.bc = bc
        self.kind = kind
        self.name = name
        self.units = units
        return

    def zero_tendencies(self, Gr):
        for k in range(Gr.nzg):
            self.tendencies[k] = 0.0
        return

    def set_bcs(self,Gr):
        start_low = Gr.gw - 1
        start_high = Gr.nzg - Gr.gw - 1

        if self.bc == 'sym':
            for k in range(Gr.gw):
                self.values[start_high + k +1] = self.values[start_high  - k]
                self.values[start_low - k] = self.values[start_low + 1 + k]

                self.mf_update[start_high + k +1] = self.mf_update[start_high  - k]
                self.mf_update[start_low - k] = self.mf_update[start_low + 1 + k]

                self.new[start_high + k +1] = self.new[start_high  - k]
                self.new[start_low - k] = self.new[start_low + 1 + k]
        else:
            self.values[start_high] = 0.0
            self.values[start_low] = 0.0

            self.mf_update[start_high] = 0.0
            self.mf_update[start_low] = 0.0

            self.new[start_high] = 0.0
            self.new[start_low] = 0.0

            for k in range(1,Gr.gw):
                self.values[start_high+ k] = -self.values[start_high - k ]
                self.values[start_low- k] = -self.values[start_low + k  ]

                self.mf_update[start_high+ k] = -self.mf_update[start_high - k ]
                self.mf_update[start_low- k] = -self.mf_update[start_low + k  ]

                self.new[start_high+ k] = -self.new[start_high - k ]
                self.new[start_low- k] = -self.new[start_low + k  ]

        return

class VariableDiagnostic:

    def __init__(self, Gr, loc, kind, bc, name, units):
        # Value at the current timestep
        # Placement on staggered grid
        self.values = Field.field(Gr, loc)
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.bc = bc
        self.kind = kind
        self.name = name
        self.units = units
        return
    def set_bcs(self,Gr):
        start_low = Gr.gw - 1
        start_high = Gr.nzg - Gr.gw
        if self.bc == 'sym':
            for k in range(Gr.gw):
                self.values[start_high + k] = self.values[start_high  - 1]
                self.values[start_low - k] = self.values[start_low + 1]
        else:
            self.values[start_high] = 0.0
            self.values[start_low] = 0.0
            for k in range(1,Gr.gw):
                self.values[start_high+ k] = 0.0  #-self.values[start_high - k ]
                self.values[start_low- k] = 0.0 #-self.values[start_low + k ]
        return



class GridMeanVariables:
    def __init__(self, namelist, Gr, Ref):
        self.Gr = Gr
        self.Ref = Ref

        self.U = VariablePrognostic(Gr, 'half', 'velocity', 'sym','u', 'm/s' )
        self.V = VariablePrognostic(Gr, 'half', 'velocity','sym', 'v', 'm/s' )
        # Just leave this zero for now!
        self.W = VariablePrognostic(Gr, 'full', 'velocity','sym', 'v', 'm/s' )

        # Create thermodynamic variables
        self.QT = VariablePrognostic(Gr, 'half', 'scalar','sym', 'qt', 'kg/kg')
        self.QR = VariablePrognostic(Gr, 'half', 'scalar','sym', 'qr', 'kg/kg')

        if namelist['thermodynamics']['thermal_variable'] == 'entropy':
            self.H = VariablePrognostic(Gr, 'half', 'scalar', 'sym','s', 'J/kg/K' )
            self.t_to_prog_fp = t_to_entropy_c
            self.prog_to_t_fp = eos_first_guess_entropy
        elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
            self.H = VariablePrognostic(Gr, 'half', 'scalar', 'sym','thetal', 'K')
            self.t_to_prog_fp = t_to_thetali_c
            self.prog_to_t_fp = eos_first_guess_thetal
        else:
            sys.exit('Did not recognize thermal variable ' + namelist['thermodynamics']['thermal_variable'])

        # Diagnostic Variables--same class as the prognostic variables, but we append to diagnostics list
        # self.diagnostics_list  = []
        self.QL  = VariableDiagnostic(Gr, 'half', 'scalar','sym', 'ql', 'kg/kg')
        self.T   = VariableDiagnostic(Gr, 'half', 'scalar','sym', 'temperature', 'K')
        self.B   = VariableDiagnostic(Gr, 'half', 'scalar','sym', 'buoyancy', 'm^2/s^3')
        self.THL = VariableDiagnostic(Gr, 'half', 'scalar', 'sym', 'thetal','K')

        # TKE   TODO   repeated from EDMF_Environment.pyx logic
        if  namelist['turbulence']['scheme'] == 'EDMF_PrognosticTKE':
            self.calc_tke = True
        else:
            self.calc_tke = False
        try:
            self.calc_tke = namelist['turbulence']['EDMF_PrognosticTKE']['calculate_tke']
        except:
            pass

        try:
            self.calc_scalar_var = namelist['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var']
        except:
            self.calc_scalar_var = False

        try:
            self.EnvThermo_scheme = str(namelist['thermodynamics']['saturation'])
        except:
            self.EnvThermo_scheme = 'sa_mean'

        #Now add the 2nd moment variables
        if self.calc_tke:
            self.TKE = VariableDiagnostic(Gr, 'half', 'scalar','sym', 'tke','m^2/s^2' )

        if self.calc_scalar_var:
            self.QTvar = VariableDiagnostic(Gr, 'half', 'scalar','sym', 'qt_var','kg^2/kg^2' )
            if namelist['thermodynamics']['thermal_variable'] == 'entropy':
                self.Hvar = VariableDiagnostic(Gr, 'half', 'scalar', 'sym', 's_var', '(J/kg/K)^2')
                self.HQTcov = VariableDiagnostic(Gr, 'half', 'scalar', 'sym' ,'s_qt_covar', '(J/kg/K)(kg/kg)' )
            elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
                self.Hvar = VariableDiagnostic(Gr, 'half', 'scalar', 'sym' ,'thetal_var', 'K^2')
                self.HQTcov = VariableDiagnostic(Gr, 'half', 'scalar','sym' ,'thetal_qt_covar', 'K(kg/kg)' )

        if self.EnvThermo_scheme == 'sommeria_deardorff':
            self.THVvar = VariableDiagnostic(Gr, 'half', 'scalar','sym', 'thatav_var','K^2' )

        return

    def zero_tendencies(self):
        self.U.zero_tendencies(self.Gr)
        self.V.zero_tendencies(self.Gr)
        self.QT.zero_tendencies(self.Gr)
        self.QR.zero_tendencies(self.Gr)
        self.H.zero_tendencies(self.Gr)
        return

    def update(self, TS):
        for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            self.U.values[k]  +=  self.U.tendencies[k] * TS.dt
            self.V.values[k]  +=  self.V.tendencies[k] * TS.dt
            self.H.values[k]  +=  self.H.tendencies[k] * TS.dt
            self.QT.values[k] +=  self.QT.tendencies[k] * TS.dt
            self.QR.values[k] +=  self.QR.tendencies[k] * TS.dt

        self.U.set_bcs(self.Gr)
        self.V.set_bcs(self.Gr)
        self.H.set_bcs(self.Gr)
        self.QT.set_bcs(self.Gr)
        self.QR.set_bcs(self.Gr)

        if self.calc_tke:
            self.TKE.set_bcs(self.Gr)

        if self.calc_scalar_var:
            self.QTvar.set_bcs(self.Gr)
            self.Hvar.set_bcs(self.Gr)
            self.HQTcov.set_bcs(self.Gr)

        if self.EnvThermo_scheme == 'sommeria_deardorff':
            self.THVvar.set_bcs(self.Gr)

        self.zero_tendencies()
        return

    def initialize_io(self, Stats):
        Stats.add_profile('u_mean')
        Stats.add_profile('v_mean')
        Stats.add_profile('qt_mean')
        Stats.add_profile('qr_mean')
        if self.H.name == 's':
            Stats.add_profile('s_mean')
            Stats.add_profile('thetal_mean')
        elif self.H.name == 'thetal':
            Stats.add_profile('thetal_mean')

        Stats.add_profile('temperature_mean')
        Stats.add_profile('buoyancy_mean')
        Stats.add_profile('ql_mean')
        if self.calc_tke:
            Stats.add_profile('tke_mean')
        if self.calc_scalar_var:
            Stats.add_profile('Hvar_mean')
            Stats.add_profile('QTvar_mean')
            Stats.add_profile('HQTcov_mean')

        Stats.add_ts('lwp')
        return

    def io(self, Stats):
        lwp = 0.0
        Stats.write_profile_new('u_mean'           , self.Gr, self.U.values)
        Stats.write_profile_new('v_mean'           , self.Gr, self.V.values)
        Stats.write_profile_new('qt_mean'          , self.Gr, self.QT.values)
        Stats.write_profile_new('ql_mean'          , self.Gr, self.QL.values)
        Stats.write_profile_new('qr_mean'          , self.Gr, self.QR.values)
        Stats.write_profile_new('temperature_mean' , self.Gr, self.T.values)
        Stats.write_profile_new('buoyancy_mean'    , self.Gr, self.B.values)
        if self.H.name == 's':
            Stats.write_profile_new('s_mean'     , self.Gr, self.H.values)
            Stats.write_profile_new('thetal_mean', self.Gr, self.THL.values)
        elif self.H.name == 'thetal':
            Stats.write_profile_new('thetal_mean', self.Gr, self.H.values)
        if self.calc_tke:
            Stats.write_profile_new('tke_mean'   , self.Gr, self.TKE.values)
        if self.calc_scalar_var:
            Stats.write_profile_new('Hvar_mean'  , self.Gr, self.Hvar.values)
            Stats.write_profile_new('QTvar_mean' , self.Gr, self.QTvar.values)
            Stats.write_profile_new('HQTcov_mean', self.Gr, self.HQTcov.values)

        for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            lwp += self.Ref.rho0_half[k]*self.QL.values[k]*self.Gr.dz
        Stats.write_ts('lwp', lwp)

        return

    def satadjust(self):
        for k in range(self.Gr.nzg):
            h = self.H.values[k]
            qt = self.QT.values[k]
            p0 = self.Ref.p0_half[k]
            T, ql = eos(self.t_to_prog_fp,self.prog_to_t_fp, p0, qt, h )
            self.QL.values[k] = ql
            self.T.values[k] = T
            qv = qt - ql
            self.THL.values[k] = t_to_thetali_c(p0, T, qt, ql,0.0)
            alpha = alpha_c(p0, T, qt, qv)
            self.B.values[k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)
        return
