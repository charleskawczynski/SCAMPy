import time
import copy
import numpy as np
from funcs_EDMF import *
from EDMF_Updrafts import *
from Turbulence_PrognosticTKE import EDMF_PrognosticTKE, compute_grid_means
from Cases import CasesFactory
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid
from Field import Field, Full, Half, Dirichlet, Neumann, nice_name
from StateVec import StateVec
from ReferenceState import ReferenceState
import matplotlib.pyplot as plt
import Cases
from Surface import  SurfaceBase
from Cases import  CasesBase
from NetCDFIO import NetCDFIO_Stats
from TimeStepping import TimeStepping
from DomainDecomp import *
from DomainSubSet import *

def plot_solutions_new(grid, sv, var_names, Stats):
    gm, en, ud, sd, al = sv.idx.allcombinations()
    n = 3
    for name in var_names:
        i_each = sv.over_sub_domains(name)
        for i in i_each:
            if 'T' in name:
                plt.plot(sv[name, i][n:-n]     , grid.z[n:-n])
            else:
                plt.plot(sv[name, i]           , grid.z)
            p_nice = nice_name(sv.var_string(name, i))
            file_name = p_nice+'.png'
            plt.title(p_nice+' vs z')
            plt.xlabel(p_nice)
            plt.ylabel('z')
            plt.savefig(Stats.figpath+file_name)
            plt.close()

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
        p_nice = nice_name(p)
        file_name = p_nice+'.png'
        plt.title(p_nice+' vs z')
        plt.xlabel(p_nice)
        plt.ylabel('z')
        plt.savefig(Stats.figpath+file_name)
        plt.close()

class Simulation1d:

    def __init__(self, namelist, paramlist, root_dir):
        z_min        = 0
        n_elems_real = namelist['grid']['nz']
        z_max        = namelist['grid']['dz']*namelist['grid']['nz']
        n_ghost      = namelist['grid']['gw']
        N_subdomains = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']+2

        N_sd = N_subdomains-2

        dd = DomainDecomp(gm=1,en=1,ud=N_sd)
        dss_all = DomainSubSet(gm=True,en=True,ud=True)
        unkowns = (
         ('a'             , dss_all, Center() , Neumann() ),
         ('w'             , dss_all, Center() , Dirichlet() ),
         ('q_tot'         , dss_all, Center() , Neumann() ),
         ('θ_liq'         , dss_all, Center() , Neumann() ),
         ('tke'           , dss_all, Center() , Neumann() ),
         ('u'             , dss_all, Center() , Neumann() ),
         ('v'             , dss_all, Center() , Neumann() ),
        )


        temp_vars = (
                     ('ρ_0'                    , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('p_0'                    , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('HVSD_a'                 , DomainSubSet(ud=True),  Center() , Neumann()),
                     ('HVSD_w'                 , DomainSubSet(ud=True),  Center() , Neumann()),
                     ('q_liq'                  , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('T'                      , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('buoy'                   , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('δ_model'                , DomainSubSet(ud=True),  Center() , Neumann()),
                     ('ε_model'                , DomainSubSet(ud=True),  Center() , Neumann()),
                     ('l_mix'                  , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('K_m'                    , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('K_h'                    , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('α_0'                    , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('mean_entr_sc'           , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('mean_detr_sc'           , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('massflux_half'          , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('mf_q_tot_half'          , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('mf_θ_liq_half'          , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('temp_C'                 , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('T_diff'                 , DomainSubSet(en=True),  Center() , Neumann()),
                     ('q_liq_diff'             , DomainSubSet(en=True),  Center() , Neumann()),
                     ('q_tot_dry'              , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('θ_dry'                  , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('t_cloudy'               , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('q_vap_cloudy'           , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('q_tot_cloudy'           , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('θ_cloudy'               , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('cv_θ_liq_rain_dt'       , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('cv_q_tot_rain_dt'       , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('cv_θ_liq_q_tot_rain_dt' , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('CF'                     , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('dTdt'                   , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('dqtdt'                  , DomainSubSet(gm=True),  Center() , Neumann()),
                     ('δ_src_a'                , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('δ_src_w'                , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('δ_src_q_tot'            , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('δ_src_θ_liq'            , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('δ_src_a_model'          , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('δ_src_w_model'          , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('δ_src_q_tot_model'      , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('δ_src_θ_liq_model'      , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('cloud_water_excess'     , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('θ_liq_src_rain'         , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('prec_src_θ_liq'         , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('prec_src_q_tot'         , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('mf_θ_liq'               , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('mf_q_tot'               , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('mf_tend_θ_liq'          , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('mf_tend_q_tot'          , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('mf_tmp'                 , DomainSubSet(gm=True,en=True,ud=True),  Center() , Neumann()),
                     ('tmp_n'                  , DomainSubSet(gm=True,en=True,ud=True),  Node() , Neumann()),
                     )

        q_2MO = (
                     ('values'     , DomainSubSet(gm=True), Center(), Neumann()),
                     ('dissipation', DomainSubSet(gm=True), Center(), Neumann()),
                     ('entr_gain'  , DomainSubSet(gm=True), Center(), Neumann()),
                     ('detr_loss'  , DomainSubSet(gm=True), Center(), Neumann()),
                     ('buoy'       , DomainSubSet(gm=True), Center(), Neumann()),
                     ('press'      , DomainSubSet(gm=True), Center(), Neumann()),
                     ('shear'      , DomainSubSet(gm=True), Center(), Neumann()),
                     ('interdomain', DomainSubSet(gm=True), Center(), Neumann()),
                     ('rain_src'   , DomainSubSet(gm=True), Center(), Neumann()),
                     )

        n_ghost = 1
        self.grid         = Grid(z_min, z_max, n_elems_real, n_ghost)
        self.q            = StateVec(unkowns, self.grid, dd)
        self.q_new        = copy.deepcopy(self.q)
        self.q_old        = copy.deepcopy(self.q)
        self.q_tendencies = copy.deepcopy(self.q)
        self.tmp          = StateVec(temp_vars, self.grid, dd)

        self.tmp_O2 = {}
        self.tmp_O2['tke']            = StateVec(q_2MO, self.grid, dd)
        self.tmp_O2['tke']            = copy.deepcopy(self.tmp_O2['tke'])

        self.n_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']

        self.Ref        = ReferenceState(self.grid)
        self.Case       = CasesFactory(namelist, paramlist)
        self.Turb       = EDMF_PrognosticTKE(namelist, paramlist, self.grid)
        self.UpdVar     = [UpdraftVariables(0, self.Turb.surface_area, self.n_updrafts) for i in self.q.idx.alldomains()]
        self.TS         = TimeStepping(namelist)
        self.Stats      = NetCDFIO_Stats(namelist, paramlist, self.grid, root_dir)
        self.tri_diag   = type('', (), {})()
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

        self.Case.initialize_profiles(self.grid, self.Ref, self.tmp, self.q)
        self.Case.initialize_surface(self.grid, self.Ref, self.tmp)
        self.Case.initialize_forcing(self.grid, self.Ref, self.tmp)

        initialize_updrafts(self.grid, self.tmp, self.q, self.Turb.params, self.Turb.updraft_fraction)
        distribute(self.grid, self.q, ('q_tot','θ_liq'))
        distribute(self.grid, self.tmp, ('q_liq','T'))
        diagnose_environment(self.grid, self.q)
        apply_bcs(self.grid, self.q)

        self.initialize_io()
        self.export_data()
        return

    def run(self):
        gm, en, ud, sd, al = self.q.idx.allcombinations()
        self.q_tendencies.assign(self.grid, ('u', 'v', 'q_tot', 'θ_liq'), 0.0)
        self.Case.update_surface(self.grid, self.q, self.TS, self.tmp)
        self.Case.update_forcing(self.grid, self.q, self.q_tendencies, self.TS, self.tmp)

        self.Turb.initialize_vars(self.grid, self.q, self.q_tendencies, self.tmp, self.tmp_O2,
        self.UpdVar, self.Case, self.TS, self.tri_diag)

        for k in self.grid.over_elems(Center()):
            self.q['tke', en][k]            = self.q['tke', gm][k]

        self.q.export_state(self.grid, "./", "Q_py")
        self.tmp.export_state(self.grid, "./", "tmp_py")

        while self.TS.t <= self.TS.t_max:
            if np.mod(self.TS.t, self.Stats.frequency) == 0:
                print('Percent complete: ', self.TS.t/self.TS.t_max*100)
            self.q_tendencies.assign(self.grid, ('u', 'v', 'q_tot', 'θ_liq'), 0.0)
            self.Case.update_surface(self.grid, self.q, self.TS, self.tmp)
            self.Case.update_forcing(self.grid, self.q, self.q_tendencies, self.TS, self.tmp)
            self.Turb.update(self.grid, self.q_new, self.q, self.q_tendencies, self.tmp, self.tmp_O2,
                             self.UpdVar,
                             self.Case, self.TS, self.tri_diag)

            self.TS.update()
            compute_grid_means(self.grid, self.q, self.tmp)
            if np.mod(self.TS.t, self.Stats.frequency) == 0:
                self.export_data()
        sol = self.package_sol()
        return sol

    def package_sol(self):
        sol = type('', (), {})()
        sol.grid = self.grid
        sol.z = self.grid.z
        sol.z_half = self.grid.z_half

        gm, en, ud, sd, al = self.q.idx.allcombinations()

        sol.e_W              = self.q['w', en]
        sol.e_q_tot          = self.q['q_tot', en]
        sol.e_θ_liq          = self.q['θ_liq', en]
        sol.e_q_liq          = self.tmp['q_liq', en]
        sol.e_T              = self.tmp['T', en]
        sol.e_B              = self.tmp['buoy', en]
        sol.e_CF             = self.tmp['CF']
        sol.e_tke            = self.q['tke', en]

        sol.ud_W     = self.q['w', ud[0]]
        sol.ud_Area  = self.q['a', ud[0]]
        sol.ud_q_tot = self.q['q_tot', ud[0]]
        sol.ud_q_liq = self.tmp['q_liq', ud[0]]
        sol.ud_T     = self.tmp['T', ud[0]]
        sol.ud_B     = self.tmp['buoy', ud[0]]

        sol.gm_q_tot = self.q['q_tot', gm]
        sol.gm_U     = self.q['u', gm]
        sol.gm_θ_liq = self.q['θ_liq', gm]
        sol.gm_T     = self.tmp['T', gm]
        sol.gm_V     = self.q['v', gm]
        sol.gm_q_liq = self.tmp['q_liq', gm]
        sol.gm_B     = self.tmp['buoy', gm]

        q_vars = ('w',
                  'q_tot',
                  'θ_liq',
                  'tke',
                  'a',
                  # 'u',
                  # 'v',
                  )
        tmp_vars = ('q_liq',
                    'T',
                    'T_diff',
                    'q_liq_diff',
                    'HVSD_a',
                    'HVSD_w',
                    'buoy',
                    'CF',
                    )
        plot_solutions_new(self.grid, self.q, q_vars, self.Stats)
        plot_solutions_new(self.grid, self.tmp, tmp_vars, self.Stats)

        return sol


    def initialize_io(self):
        for v in self.q.var_names:
          for i in self.q.over_sub_domains(v):
            self.Stats.add_profile(self.q.var_string(v,i))

        for v in self.tmp.var_names:
          for i in self.tmp.over_sub_domains(v):
            self.Stats.add_profile(self.tmp.var_string(v,i))

        for k in self.tmp_O2:
          q_local = self.tmp_O2[k]
          for v in q_local.var_names:
            for i in q_local.over_sub_domains(v):
              self.Stats.add_profile(k+'_'+q_local.var_string(v,i))

        self.Stats.add_ts('lwp')
        self.Case.initialize_io(self.Stats)
        initialize_io_updrafts(self.UpdVar, self.Stats)
        return

    def export_data(self):
        gm, en, ud, sd, al = self.q.idx.allcombinations()
        self.Stats.open_files()
        self.Stats.write_simulation_time(self.TS.t)
        self.Case.export_data(self.Stats)
        pre_export_data_compute(self.grid, self.q, self.tmp, self.tmp_O2, self.Stats, self.Turb.tke_diss_coeff)
        export_data_updrafts(self.grid, self.UpdVar, self.q, self.tmp, self.Stats)

        lwp = 0.0
        for k in self.grid.over_elems_real(Center()):
            lwp += self.tmp['ρ_0'][k]*self.tmp['q_liq', gm][k]*self.grid.dz
        self.Stats.write_ts('lwp', lwp)

        for v in self.q.var_names:
          for i in self.q.over_sub_domains(v):
            self.Stats.write_profile_new(self.q.var_string(v,i), self.grid, self.q[v, i])

        for v in self.tmp.var_names:
          for i in self.tmp.over_sub_domains(v):
            self.Stats.write_profile_new(self.tmp.var_string(v,i), self.grid, self.tmp[v, i])

        for k in self.tmp_O2:
          q_local = self.tmp_O2[k]
          for v in q_local.var_names:
            for i in q_local.over_sub_domains(v):
              self.Stats.write_profile_new(k+'_'+q_local.var_string(v,i), self.grid, q_local[v])

        self.Stats.close_files()
        return

    def force_io(self):
        return
