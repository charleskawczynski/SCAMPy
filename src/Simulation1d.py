import time
import copy
import numpy as np
from Variables import GridMeanVariables
from EDMF_Updrafts import *
from EDMF_Environment import *
from Turbulence_PrognosticTKE import EDMF_PrognosticTKE, compute_grid_means
from Cases import CasesFactory
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann, nice_name
from StateVec import StateVec
from ReferenceState import ReferenceState
import matplotlib.pyplot as plt
import Cases
from Surface import  SurfaceBase
from Cases import  CasesBase
from NetCDFIO import NetCDFIO_Stats
from TimeStepping import TimeStepping

def plot_solutions(sol, Stats):
    props = [x for x in dir(sol) if not (x.startswith('__') and x.endswith('__'))]
    props = [x for x in props if not x=='z']
    props = [x for x in props if not x=='z_half']
    for p in props:
        if 'T' in p:
            n = 3
            plt.plot(getattr(sol, p)[n:-n]     , sol.z[n:-n])
        else:
            plt.plot(getattr(sol, p)     , sol.z)
        file_name = nice_name(p+'.png')
        plt.savefig(Stats.figpath+file_name)
        plt.close()

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
         ('U'             , Center() , Neumann() , N_sd),
         ('V'             , Center() , Neumann() , N_sd),
        )

        temp_vars = (
                     ('mean_entr_sc'  , Center() , Neumann(), 1),
                     ('mean_detr_sc'  , Center() , Neumann(), 1),
                     ('massflux_half' , Center() , Neumann(), 1),
                     ('mf_q_tot_half' , Center() , Neumann(), 1),
                     ('mf_θ_liq_half' , Center() , Neumann(), 1),
                     ('temp_C'        , Center() , Neumann(), 1),
                     ('ρ_0'           , Node()   , Neumann(), 1),
                     ('α_0'           , Node()   , Neumann(), 1),
                     ('p_0'           , Node()   , Neumann(), 1),
                     ('ρ_0_half'      , Center() , Neumann(), 1),
                     ('α_0_half'      , Center() , Neumann(), 1),
                     ('p_0_half'      , Center() , Neumann(), 1),
                     ('K_m'           , Center() , Neumann(), 1),
                     ('K_h'           , Center() , Neumann(), 1),
                     ('l_mix'         , Center() , Neumann(), 1),
                     ('q_liq'         , Center() , Neumann(), N_sd),
                     ('T'             , Center() , Neumann(), N_sd),
                     ('mf_θ_liq'      , Node() , Neumann(), N_sd),
                     ('mf_q_tot'      , Node() , Neumann(), N_sd),
                     ('mf_tend_θ_liq' , Node() , Neumann(), N_sd),
                     ('mf_tend_q_tot' , Node() , Neumann(), N_sd),
                     ('mf_tmp'        , Node() , Neumann(), N_sd),
                     ('entr_sc'       , Center() , Neumann(), 1), # Entrainment/Detrainment rates
                     ('detr_sc'       , Center() , Neumann(), 1), # Entrainment/Detrainment rates
                     ('ρaK_m'         , Node()   , Neumann(), N_sd),
                     ('ρaK_h'         , Node()   , Neumann(), N_sd),
                     )

        q_2MO = (
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

        self.grid         = Grid(z_min, z_max, n_elems_real, n_ghost)
        self.q            = StateVec(unkowns, self.grid)
        self.q_new        = copy.deepcopy(self.q)
        self.q_old        = copy.deepcopy(self.q)
        self.q_tendencies = copy.deepcopy(self.q)
        self.tmp          = StateVec(temp_vars, self.grid)

        self.tmp_O2 = {}
        self.tmp_O2['tke']            = StateVec(q_2MO, self.grid)
        self.tmp_O2['cv_q_tot']       = copy.deepcopy(self.tmp_O2['tke'])
        self.tmp_O2['cv_θ_liq']       = copy.deepcopy(self.tmp_O2['tke'])
        self.tmp_O2['cv_θ_liq_q_tot'] = copy.deepcopy(self.tmp_O2['tke'])
        self.tmp_O2['tke']            = copy.deepcopy(self.tmp_O2['tke'])

        self.n_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']

        self.Ref       = ReferenceState(self.grid)
        self.Case      = CasesFactory(namelist, paramlist)

        self.GMV       = GridMeanVariables(namelist, self.grid)
        self.UpdVar    = UpdraftVariables(self.n_updrafts, namelist, paramlist, self.grid)
        self.EnvVar    = EnvironmentVariables(namelist, self.grid)

        self.UpdThermo = UpdraftThermodynamics(self.n_updrafts, self.grid, self.UpdVar)
        self.UpdMicro  = UpdraftMicrophysics(paramlist, self.n_updrafts, self.grid)
        self.EnvThermo = EnvironmentThermodynamics(namelist, paramlist, self.grid, self.EnvVar)

        self.Turb  = EDMF_PrognosticTKE(namelist, paramlist, self.grid)
        self.TS    = TimeStepping(namelist)
        self.Stats = NetCDFIO_Stats(namelist, paramlist, self.grid, root_dir)
        self.tri_diag = type('', (), {})()
        self.tri_diag.a = Half(self.grid)
        self.tri_diag.b = Half(self.grid)
        self.tri_diag.c = Half(self.grid)
        self.tri_diag.f = Half(self.grid)
        self.tri_diag.β = Half(self.grid)
        self.tri_diag.γ = Half(self.grid)
        self.tri_diag.xtemp = Half(self.grid)
        self.tri_diag.ρaK = Full(self.grid)
        return

    def initialize(self, namelist):
        self.Case.initialize_reference(self.grid, self.Ref, self.Stats, self.tmp)
        self.Case.initialize_profiles(self.grid, self.GMV, self.Ref, self.tmp, self.q)
        self.Case.initialize_surface(self.grid, self.Ref, self.tmp)
        self.Case.initialize_forcing(self.grid, self.Ref, self.GMV, self.tmp)
        self.UpdVar.initialize(self.grid, self.GMV, self.tmp, self.q)
        self.initialize_io()
        self.io()
        return

    def run(self):
        self.q_tendencies.assign(self.grid, ('U', 'V', 'q_tot', 'q_rai', 'θ_liq'), 0.0)
        self.Case.update_surface(self.grid, self.q, self.GMV, self.TS, self.tmp)
        self.Case.update_forcing(self.grid, self.q_tendencies, self.GMV, self.TS, self.tmp)
        self.Turb.initialize_vars(self.grid, self.q, self.q_tendencies, self.tmp, self.GMV,
        self.EnvVar, self.UpdVar, self.UpdMicro, self.EnvThermo, self.UpdThermo, self.Case, self.TS, self.tri_diag)
        for k in self.grid.over_elems(Center()):
            self.EnvVar.tke.values[k]            = self.GMV.tke.values[k]
            self.EnvVar.cv_θ_liq.values[k]       = self.GMV.cv_θ_liq.values[k]
            self.EnvVar.cv_q_tot.values[k]       = self.GMV.cv_q_tot.values[k]
            self.EnvVar.cv_θ_liq_q_tot.values[k] = self.GMV.cv_θ_liq_q_tot.values[k]

        while self.TS.t <= self.TS.t_max:
            if np.mod(self.TS.t, self.Stats.frequency) == 0:
                print('Percent complete: ', self.TS.t/self.TS.t_max*100)
            self.q_tendencies.assign(self.grid, ('U', 'V', 'q_tot', 'q_rai', 'θ_liq'), 0.0)
            self.Case.update_surface(self.grid, self.q, self.GMV, self.TS, self.tmp)
            self.Case.update_forcing(self.grid, self.q_tendencies, self.GMV, self.TS, self.tmp)
            self.Turb.update(self.grid, self.q, self.q_tendencies, self.tmp, self.GMV, self.EnvVar,
                             self.UpdVar, self.UpdMicro, self.EnvThermo,
                             self.UpdThermo, self.Case, self.TS, self.tri_diag)

            self.TS.update()
            compute_grid_means(self.grid, self.q, self.tmp, self.GMV, self.EnvVar, self.UpdVar)
            if np.mod(self.TS.t, self.Stats.frequency) == 0:
                self.io()
        sol = self.package_sol()
        return sol

    def package_sol(self):
        sol = type('', (), {})()
        sol.z = self.grid.z
        sol.z_half = self.grid.z_half

        i_gm, i_env, i_uds, i_sd = self.q.domain_idx()

        sol.e_W              = self.q['w', i_env].values
        sol.e_q_tot          = self.EnvVar.q_tot.values
        sol.e_q_liq          = self.EnvVar.q_liq.values
        sol.e_q_rai          = self.EnvVar.q_rai.values
        sol.e_θ_liq          = self.EnvVar.θ_liq.values
        sol.e_T              = self.EnvVar.T.values
        sol.e_B              = self.EnvVar.B.values
        sol.e_CF             = self.EnvVar.CF.values
        sol.e_tke            = self.EnvVar.tke.values
        sol.e_cv_θ_liq       = self.EnvVar.cv_θ_liq.values
        sol.e_cv_q_tot       = self.EnvVar.cv_q_tot.values
        sol.e_cv_θ_liq_q_tot = self.EnvVar.cv_θ_liq_q_tot.values

        sol.ud_W     = self.UpdVar.W.values[0]
        sol.ud_Area  = self.UpdVar.Area.values[0]
        sol.ud_q_tot = self.UpdVar.q_tot.values[0]
        sol.ud_q_liq = self.UpdVar.q_liq.values[0]
        sol.ud_q_rai = self.UpdVar.q_rai.values[0]
        sol.ud_T     = self.UpdVar.T.values[0]
        sol.ud_B     = self.UpdVar.B.values[0]

        sol.gm_q_tot = self.GMV.q_tot.values
        sol.gm_U     = self.GMV.U.values
        sol.gm_θ_liq = self.GMV.θ_liq.values
        sol.gm_T     = self.GMV.T.values
        sol.gm_V     = self.GMV.V.values
        sol.gm_q_liq = self.GMV.q_liq.values
        sol.gm_B     = self.GMV.B.values

        plot_solutions(sol, self.Stats)
        return sol

    def initialize_io(self):
        self.GMV.initialize_io(self.Stats)
        self.Case.initialize_io(self.Stats)
        self.Turb.initialize_io(self.Stats, self.EnvVar, self.UpdVar)
        return

    def io(self):
        self.Stats.open_files()
        self.Stats.write_simulation_time(self.TS.t)
        self.GMV.io(self.grid, self.Stats, self.tmp)
        self.Case.io(self.Stats)
        self.Turb.io(self.grid, self.q, self.tmp, self.Stats, self.EnvVar, self.UpdVar, self.UpdMicro)
        self.Stats.close_files()
        return

    def force_io(self):
        return
