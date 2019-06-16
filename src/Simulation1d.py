import time
import numpy as np
from Variables import GridMeanVariables
from Turbulence_PrognosticTKE import ParameterizationFactory
from Cases import CasesFactory
from Grid import Grid, Zmin, Zmax, Center, Node
from StateVec import StateVec
from ReferenceState import ReferenceState
import matplotlib.pyplot as plt
import Cases
from Surface import  SurfaceBase
from Cases import  CasesBase
from NetCDFIO import NetCDFIO_Stats
from TimeStepping import TimeStepping

class Simulation1d:

    def __init__(self, namelist, paramlist, root_dir):
        z_min        = 0
        n_elems_real = namelist['grid']['nz']
        z_max        = namelist['grid']['dz']*namelist['grid']['nz']
        n_ghost      = namelist['grid']['gw']
        N_subdomains = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']+2

        unkowns = (('a', N_subdomains),
                   ('w', N_subdomains),
                   ('q_tot', N_subdomains),
                   ('θ_liq', N_subdomains),
                   ('tke', N_subdomains),
                   ('cv_q_tot', N_subdomains),
                   ('cv_θ_liq', N_subdomains),
                   ('cv_θ_liq_q_tot', N_subdomains),)
        temp_vars = (('ρ_0', 1),
                     ('α_0', 1),
                     ('p_0', 1),
                     )

        self.grid = Grid(z_min, z_max, n_elems_real, n_ghost)
        self.q = StateVec(unkowns, self.grid)
        self.tmp = StateVec(temp_vars, self.grid)

        self.Ref = ReferenceState(self.grid)
        self.GMV = GridMeanVariables(namelist, self.grid, self.Ref)
        self.Case = CasesFactory(namelist, paramlist)
        self.Turb = ParameterizationFactory(namelist, paramlist, self.grid, self.Ref)
        self.TS = TimeStepping(namelist)
        self.Stats = NetCDFIO_Stats(namelist, paramlist, self.grid, root_dir)
        return

    def initialize(self, namelist):
        self.Case.initialize_reference(self.grid, self.Ref, self.Stats, self.tmp)
        self.Case.initialize_profiles(self.grid, self.GMV, self.Ref, self.tmp, self.q)
        self.Case.initialize_surface(self.grid, self.Ref, self.tmp)
        self.Case.initialize_forcing(self.grid, self.Ref, self.GMV, self.tmp)
        self.Turb.initialize(self.GMV, self.tmp)
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

        sol.z = self.grid.z
        sol.z_half = self.grid.z_half

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


        plt.plot(sol.ud_W   , sol.z); plt.savefig(self.Stats.outpath+'ud_W.png'); plt.close()
        plt.plot(sol.ud_Area, sol.z); plt.savefig(self.Stats.outpath+'ud_Area.png'); plt.close()
        plt.plot(sol.ud_QT  , sol.z); plt.savefig(self.Stats.outpath+'ud_QT.png'); plt.close()
        plt.plot(sol.ud_QL  , sol.z); plt.savefig(self.Stats.outpath+'ud_QL.png'); plt.close()
        plt.plot(sol.ud_QR  , sol.z); plt.savefig(self.Stats.outpath+'ud_QR.png'); plt.close()
        plt.plot(sol.ud_THL , sol.z); plt.savefig(self.Stats.outpath+'ud_THL.png'); plt.close()
        plt.plot(sol.ud_T   , sol.z); plt.savefig(self.Stats.outpath+'ud_T.png'); plt.close()
        plt.plot(sol.ud_B   , sol.z); plt.savefig(self.Stats.outpath+'ud_B.png'); plt.close()
        plt.plot(sol.e_W     , sol.z); plt.savefig(self.Stats.outpath+'e_W.png'); plt.close()
        plt.plot(sol.e_QT    , sol.z); plt.savefig(self.Stats.outpath+'e_QT.png'); plt.close()
        plt.plot(sol.e_QL    , sol.z); plt.savefig(self.Stats.outpath+'e_QL.png'); plt.close()
        plt.plot(sol.e_QR    , sol.z); plt.savefig(self.Stats.outpath+'e_QR.png'); plt.close()
        plt.plot(sol.e_H     , sol.z); plt.savefig(self.Stats.outpath+'e_H.png'); plt.close()
        plt.plot(sol.e_THL   , sol.z); plt.savefig(self.Stats.outpath+'e_THL.png'); plt.close()
        plt.plot(sol.e_T     , sol.z); plt.savefig(self.Stats.outpath+'e_T.png'); plt.close()
        plt.plot(sol.e_B     , sol.z); plt.savefig(self.Stats.outpath+'e_B.png'); plt.close()
        plt.plot(sol.e_CF    , sol.z); plt.savefig(self.Stats.outpath+'e_CF.png'); plt.close()
        plt.plot(sol.e_TKE   , sol.z); plt.savefig(self.Stats.outpath+'e_TKE.png'); plt.close()
        plt.plot(sol.e_Hvar  , sol.z); plt.savefig(self.Stats.outpath+'e_Hvar.png'); plt.close()
        plt.plot(sol.e_QTvar , sol.z); plt.savefig(self.Stats.outpath+'e_QTvar.png'); plt.close()
        plt.plot(sol.e_HQTcov, sol.z); plt.savefig(self.Stats.outpath+'e_HQTcov.png'); plt.close()
        plt.plot(sol.gm_QT , sol.z); plt.savefig(self.Stats.outpath+'gm_QT.png'); plt.close()
        plt.plot(sol.gm_U  , sol.z); plt.savefig(self.Stats.outpath+'gm_U.png'); plt.close()
        plt.plot(sol.gm_H  , sol.z); plt.savefig(self.Stats.outpath+'gm_H.png'); plt.close()
        plt.plot(sol.gm_T  , sol.z); plt.savefig(self.Stats.outpath+'gm_T.png'); plt.close()
        plt.plot(sol.gm_THL, sol.z); plt.savefig(self.Stats.outpath+'gm_THL.png'); plt.close()
        plt.plot(sol.gm_V  , sol.z); plt.savefig(self.Stats.outpath+'gm_V.png'); plt.close()
        plt.plot(sol.gm_QL , sol.z); plt.savefig(self.Stats.outpath+'gm_QL.png'); plt.close()
        plt.plot(sol.gm_B  , sol.z); plt.savefig(self.Stats.outpath+'gm_B.png'); plt.close()

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
