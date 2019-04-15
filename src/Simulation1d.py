import time
import numpy as np
from Variables import GridMeanVariables
from Turbulence_PrognosticTKE import ParameterizationFactory
from Cases import CasesFactory
import Grid
import ReferenceState
import Cases
from Surface import  SurfaceBase
from Cases import  CasesBase
from NetCDFIO import NetCDFIO_Stats
import TimeStepping

class Simulation1d:

    def __init__(self, namelist, paramlist, root_dir):
        self.Gr = Grid.Grid(namelist)
        self.Ref = ReferenceState.ReferenceState(self.Gr)
        self.GMV = GridMeanVariables(namelist, self.Gr, self.Ref)
        self.Case = CasesFactory(namelist, paramlist)
        self.Turb = ParameterizationFactory(namelist,paramlist, self.Gr, self.Ref)
        self.TS = TimeStepping.TimeStepping(namelist)
        self.Stats = NetCDFIO_Stats(namelist, paramlist, self.Gr, root_dir)
        return

    def initialize(self, namelist):
        self.Case.initialize_reference(self.Gr, self.Ref, self.Stats)
        self.Case.initialize_profiles(self.Gr, self.GMV, self.Ref)
        self.Case.initialize_surface(self.Gr, self.Ref )
        self.Case.initialize_forcing(self.Gr, self.Ref, self.GMV)
        self.Turb.initialize(self.GMV)
        self.initialize_io()
        self.io()
        return

    def run(self):
        sol = type('', (), {})()

        while self.TS.t <= self.TS.t_max:
            self.GMV.zero_tendencies()
            self.Case.update_surface(self.GMV, self.TS)
            self.Case.update_forcing(self.GMV, self.TS)
            self.Turb.update(self.GMV, self.Case, self.TS)

            self.TS.update()
            # Apply the tendencies, also update the BCs and diagnostic thermodynamics
            self.GMV.update(self.TS)
            self.Turb.update_GMV_diagnostics(self.GMV)
            if np.mod(self.TS.t, self.Stats.frequency) == 0:
                self.io()

        sol.z = self.Gr.z
        sol.z_half = self.Gr.z_half

        sol.e_W = self.Turb.EnvVar.W.values
        sol.e_QT = self.Turb.EnvVar.QT.values
        sol.e_QL = self.Turb.EnvVar.QL.values
        sol.e_QR = self.Turb.EnvVar.QR.values
        sol.e_H = self.Turb.EnvVar.H.values
        sol.e_THL = self.Turb.EnvVar.THL.values
        sol.e_T = self.Turb.EnvVar.T.values
        sol.e_B = self.Turb.EnvVar.B.values
        sol.e_CF = self.Turb.EnvVar.CF.values
        sol.e_TKE = self.Turb.EnvVar.TKE.values
        sol.e_Hvar = self.Turb.EnvVar.Hvar.values
        sol.e_QTvar = self.Turb.EnvVar.QTvar.values
        sol.e_HQTcov = self.Turb.EnvVar.HQTcov.values

        sol.ud_W = self.Turb.UpdVar.W.values[0]
        sol.ud_Area = self.Turb.UpdVar.Area.values[0]
        sol.ud_QT = self.Turb.UpdVar.QT.values[0]
        sol.ud_QL = self.Turb.UpdVar.QL.values[0]
        sol.ud_QR = self.Turb.UpdVar.QR.values[0]
        sol.ud_THL = self.Turb.UpdVar.THL.values[0]
        sol.ud_T = self.Turb.UpdVar.T.values[0]
        sol.ud_B = self.Turb.UpdVar.B.values[0]

        sol.gm_QT = self.GMV.QT.values
        sol.gm_U = self.GMV.U.values
        sol.gm_H = self.GMV.H.values
        sol.gm_T = self.GMV.T.values
        sol.gm_THL = self.GMV.THL.values
        sol.gm_V = self.GMV.V.values
        sol.gm_QL = self.GMV.QL.values
        sol.gm_B = self.GMV.B.values

        return sol

    def initialize_io(self):

        self.GMV.initialize_io(self.Stats)
        self.Case.initialize_io(self.Stats)
        self.Turb.initialize_io(self.Stats)
        return

    def io(self):
        self.Stats.open_files()
        self.Stats.write_simulation_time(self.TS.t)
        self.GMV.io(self.Stats)
        self.Case.io(self.Stats)
        self.Turb.io(self.Stats)
        self.Stats.close_files()
        return

    def force_io(self):
        return
