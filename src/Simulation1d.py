import time
import copy
import numpy as np
from Variables import GridMeanVariables
from Turbulence_PrognosticTKE import ParameterizationFactory
from Cases import CasesFactory
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
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

        N_sd = N_subdomains

        unkowns = (
         ('a'             , Center() , Neumann() , N_sd),
         ('w'             , Node()   , Dirichlet() , N_sd),
         ('q_tot'         , Center() , Neumann() , N_sd),
         ('q_rai'         , Center() , Neumann() , N_sd),
         ('θ_liq'         , Center() , Neumann() , N_sd),
         ('tke'           , Center() , Neumann() , N_sd),
         ('cv_q_tot'      , Center() , Neumann() , N_sd),
         ('cv_θ_liq'      , Center() , Neumann() , N_sd),
         ('cv_θ_liq_q_tot', Center() , Neumann() , N_sd),
        )

        temp_vars = (
                     ('ρ_0'         , Node()   , Neumann(), 1),
                     ('α_0'         , Node()   , Neumann(), 1),
                     ('p_0'         , Node()   , Neumann(), 1),
                     ('ρ_0_half'    , Center() , Neumann(), 1),
                     ('α_0_half'    , Center() , Neumann(), 1),
                     ('p_0_half'    , Center() , Neumann(), 1),
                     ('K_m'         , Center() , Neumann(), N_sd),
                     ('K_h'         , Center() , Neumann(), N_sd),
                     ('ρaK_m'       , Node()   , Neumann(), N_sd),
                     ('ρaK_h'       , Node()   , Neumann(), N_sd),
                     ('T_bulkvalues', Center() , Neumann(), 1),
                     ('B_bulkvalues', Center() , Neumann(), 1),
                     )

        moment_2_vars = (
                     ('values'     , Center(), Neumann(), 1),
                     ('dissipation', Center(), Neumann(), 1),
                     ('entr_gain'  , Center(), Neumann(), 1),
                     ('detr_loss'  , Center(), Neumann(), 1),
                     ('buoy'       , Center(), Neumann(), 1),
                     ('press'      , Center(), Neumann(), 1),
                     ('shear'      , Center(), Neumann(), 1),
                     ('interdomain', Center(), Neumann(), 1),
                     ('rain_src'   , Center(), Neumann(), 1),
                     )

        self.grid = Grid(z_min, z_max, n_elems_real, n_ghost)

        self.q = StateVec(unkowns, self.grid)
        self.q_new = copy.deepcopy(self.q)
        self.q_old = copy.deepcopy(self.q)
        self.q_tendencies = copy.deepcopy(self.q)
        self.q_bulkvalues = copy.deepcopy(self.q)

        self.tmp = StateVec(temp_vars, self.grid)

        self.tmp_O2 = {}
        self.tmp_O2['TKE']    = StateVec(moment_2_vars, self.grid)
        self.tmp_O2['QTvar']  = StateVec(moment_2_vars, self.grid)
        self.tmp_O2['Hvar']   = StateVec(moment_2_vars, self.grid)
        self.tmp_O2['HQTcov'] = StateVec(moment_2_vars, self.grid)
        self.tmp_O2['TKE']    = StateVec(moment_2_vars, self.grid)


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
        self.Turb.initialize(self.GMV, self.tmp, self.q)
        self.initialize_io()
        self.io()
        return

    def run(self):
        self.GMV.zero_tendencies()
        self.Case.update_surface(self.GMV, self.TS, self.tmp)
        self.Case.update_forcing(self.GMV, self.TS, self.tmp)
        self.Turb.initialize_covariance(self.GMV, self.Case, self.tmp)
        for k in self.grid.over_elems(Center()):
            self.Turb.EnvVar.TKE.values[k] = self.GMV.TKE.values[k]
            self.Turb.EnvVar.Hvar.values[k] = self.GMV.Hvar.values[k]
            self.Turb.EnvVar.QTvar.values[k] = self.GMV.QTvar.values[k]
            self.Turb.EnvVar.HQTcov.values[k] = self.GMV.HQTcov.values[k]

        while self.TS.t <= self.TS.t_max:
            print('Percent complete: ', self.TS.t/self.TS.t_max*100)
            self.GMV.zero_tendencies()
            self.Case.update_surface(self.GMV, self.TS, self.tmp)
            self.Case.update_forcing(self.GMV, self.TS, self.tmp)
            self.Turb.update(self.GMV, self.Case, self.TS, self.tmp, self.q)

            self.TS.update()
            self.GMV.update(self.TS)
            self.Turb.update_GMV_diagnostics(self.GMV, self.tmp)
            if np.mod(self.TS.t, self.Stats.frequency) == 0:
                self.io()
        sol = self.package_sol()
        return sol

    def package_sol(self):
        sol = type('', (), {})()
        sol.z = self.grid.z
        sol.z_half = self.grid.z_half

        i_gm, i_env, i_uds = self.q.domain_idx()

        sol.e_W = self.q['w', i_env].values
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

        plt.plot(sol.ud_W   , sol.z); plt.savefig(self.Stats.figpath+'ud_W.png'); plt.close()
        plt.plot(sol.ud_Area, sol.z); plt.savefig(self.Stats.figpath+'ud_Area.png'); plt.close()
        plt.plot(sol.ud_QT  , sol.z); plt.savefig(self.Stats.figpath+'ud_QT.png'); plt.close()
        plt.plot(sol.ud_QL  , sol.z); plt.savefig(self.Stats.figpath+'ud_QL.png'); plt.close()
        plt.plot(sol.ud_QR  , sol.z); plt.savefig(self.Stats.figpath+'ud_QR.png'); plt.close()
        plt.plot(sol.ud_THL , sol.z); plt.savefig(self.Stats.figpath+'ud_THL.png'); plt.close()
        plt.plot(sol.ud_T   , sol.z); plt.savefig(self.Stats.figpath+'ud_T.png'); plt.close()
        plt.plot(sol.ud_B   , sol.z); plt.savefig(self.Stats.figpath+'ud_B.png'); plt.close()
        plt.plot(sol.e_W     , sol.z); plt.savefig(self.Stats.figpath+'e_W.png'); plt.close()
        plt.plot(sol.e_QT    , sol.z); plt.savefig(self.Stats.figpath+'e_QT.png'); plt.close()
        plt.plot(sol.e_QL    , sol.z); plt.savefig(self.Stats.figpath+'e_QL.png'); plt.close()
        plt.plot(sol.e_QR    , sol.z); plt.savefig(self.Stats.figpath+'e_QR.png'); plt.close()
        plt.plot(sol.e_H     , sol.z); plt.savefig(self.Stats.figpath+'e_H.png'); plt.close()
        plt.plot(sol.e_THL   , sol.z); plt.savefig(self.Stats.figpath+'e_THL.png'); plt.close()
        plt.plot(sol.e_T     , sol.z); plt.savefig(self.Stats.figpath+'e_T.png'); plt.close()
        plt.plot(sol.e_B     , sol.z); plt.savefig(self.Stats.figpath+'e_B.png'); plt.close()
        plt.plot(sol.e_CF    , sol.z); plt.savefig(self.Stats.figpath+'e_CF.png'); plt.close()
        plt.plot(sol.e_TKE   , sol.z); plt.savefig(self.Stats.figpath+'e_TKE.png'); plt.close()
        plt.plot(sol.e_Hvar  , sol.z); plt.savefig(self.Stats.figpath+'e_Hvar.png'); plt.close()
        plt.plot(sol.e_QTvar , sol.z); plt.savefig(self.Stats.figpath+'e_QTvar.png'); plt.close()
        plt.plot(sol.e_HQTcov, sol.z); plt.savefig(self.Stats.figpath+'e_HQTcov.png'); plt.close()
        plt.plot(sol.gm_QT , sol.z); plt.savefig(self.Stats.figpath+'gm_QT.png'); plt.close()
        plt.plot(sol.gm_U  , sol.z); plt.savefig(self.Stats.figpath+'gm_U.png'); plt.close()
        plt.plot(sol.gm_H  , sol.z); plt.savefig(self.Stats.figpath+'gm_H.png'); plt.close()
        plt.plot(sol.gm_T  , sol.z); plt.savefig(self.Stats.figpath+'gm_T.png'); plt.close()
        plt.plot(sol.gm_THL, sol.z); plt.savefig(self.Stats.figpath+'gm_THL.png'); plt.close()
        plt.plot(sol.gm_V  , sol.z); plt.savefig(self.Stats.figpath+'gm_V.png'); plt.close()
        plt.plot(sol.gm_QL , sol.z); plt.savefig(self.Stats.figpath+'gm_QL.png'); plt.close()
        plt.plot(sol.gm_B  , sol.z); plt.savefig(self.Stats.figpath+'gm_B.png'); plt.close()
        return sol

    def initialize_io(self):
        self.GMV.initialize_io(self.Stats)
        self.Case.initialize_io(self.Stats)
        self.Turb.initialize_io(self.Stats)
        return

    def io(self):
        self.Stats.open_files()
        self.Stats.write_simulation_time(self.TS.t)
        self.GMV.io(self.Stats, self.tmp)
        self.Case.io(self.Stats)
        self.Turb.io(self.Stats, self.tmp)
        self.Stats.close_files()
        return

    def force_io(self):
        return
