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

def plot_solutions_new(grid, sv, var_names, Stats):
    i_gm, i_env, i_uds, i_sd = sv.domain_idx()
    n = 3
    vars_exclude = (
                    ('gov_eq', [i_gm]+[i_env]),
                    ('heaviside_a', [i_gm]+[i_env]),
                    ('heaviside_w', [i_gm]+[i_env]),
                   )
    for var_name in var_names:
        for i in sv.subdomains(var_name):
            if not any([v in var_name and i in j for v,j in vars_exclude]):
                if 'T' in var_name:
                    plt.plot(sv[var_name, i][n:-n]     , grid.z[n:-n])
                else:
                    plt.plot(sv[var_name, i]           , grid.z)
                p_nice = nice_name(var_name+'_'+sv.idx_name(i))
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

        N_sd = N_subdomains

        unkowns = (
         ('a'             , Center() , Neumann() , N_sd),
         ('w'             , Center() , Dirichlet() , N_sd),
         ('q_tot'         , Center() , Neumann() , N_sd),
         ('θ_liq'         , Center() , Neumann() , N_sd),
         ('tke'           , Center() , Neumann() , N_sd),
         ('u'             , Center() , Neumann() , N_sd),
         ('v'             , Center() , Neumann() , N_sd),
        )


        temp_vars = (
                     ('ρ_0'                    , Center() , Neumann(), 1),
                     ('p_0'                    , Center() , Neumann(), 1),
                     ('heaviside_a'            , Center() , Neumann(), N_sd),
                     ('heaviside_w'            , Center() , Neumann(), N_sd),
                     ('q_liq'                  , Center() , Neumann(), N_sd),
                     ('T'                      , Center() , Neumann(), N_sd),
                     ('B'                      , Center() , Neumann(), N_sd),
                     ('entr_sc'                , Center() , Neumann(), N_sd), # Entrainment/Detrainment rates
                     ('detr_sc'                , Center() , Neumann(), N_sd), # Entrainment/Detrainment rates
                     ('l_mix'                  , Center() , Neumann(), 1),
                     ('K_m'                    , Center() , Neumann(), 1),
                     ('K_h'                    , Center() , Neumann(), 1),
                     ('α_0'                    , Center() , Neumann(), 1),
                     ('mean_entr_sc'           , Center() , Neumann(), 1),
                     ('mean_detr_sc'           , Center() , Neumann(), 1),
                     ('massflux_half'          , Center() , Neumann(), 1),
                     ('mf_q_tot_half'          , Center() , Neumann(), 1),
                     ('mf_θ_liq_half'          , Center() , Neumann(), 1),
                     ('temp_C'                 , Center() , Neumann(), 1),
                     ('q_tot_dry'              , Center() , Neumann(), 1),
                     ('θ_dry'                  , Center() , Neumann(), 1),
                     ('t_cloudy'               , Center() , Neumann(), 1),
                     ('q_vap_cloudy'           , Center() , Neumann(), 1),
                     ('q_tot_cloudy'           , Center() , Neumann(), 1),
                     ('θ_cloudy'               , Center() , Neumann(), 1),
                     ('cv_θ_liq_rain_dt'       , Center() , Neumann(), 1),
                     ('cv_q_tot_rain_dt'       , Center() , Neumann(), 1),
                     ('cv_θ_liq_q_tot_rain_dt' , Center() , Neumann(), 1),
                     ('CF'                     , Center() , Neumann(), 1),
                     ('gov_eq_θ_liq_ib'        , Center() , Neumann(), N_sd),
                     ('gov_eq_q_tot_ib'        , Center() , Neumann(), N_sd),
                     ('gov_eq_θ_liq_nb'        , Center() , Neumann(), N_sd),
                     ('gov_eq_q_tot_nb'        , Center() , Neumann(), N_sd),
                     ('δ_src_a'                , Center() , Neumann(), N_sd),
                     ('δ_src_w'                , Center() , Neumann(), N_sd),
                     ('δ_src_q_tot'            , Center() , Neumann(), N_sd),
                     ('δ_src_θ_liq'            , Center() , Neumann(), N_sd),
                     ('δ_src_a_model'          , Center() , Neumann(), N_sd),
                     ('δ_src_w_model'          , Center() , Neumann(), N_sd),
                     ('δ_src_q_tot_model'      , Center() , Neumann(), N_sd),
                     ('δ_src_θ_liq_model'      , Center() , Neumann(), N_sd),
                     ('cloud_water_excess'     , Center() , Neumann(), N_sd),
                     ('θ_liq_src_rain'         , Center() , Neumann(), N_sd),
                     ('prec_src_θ_liq'         , Center() , Neumann(), N_sd),
                     ('prec_src_q_tot'         , Center() , Neumann(), N_sd),
                     ('mf_θ_liq'               , Center() , Neumann(), N_sd),
                     ('mf_q_tot'               , Center() , Neumann(), N_sd),
                     ('mf_tend_θ_liq'          , Center() , Neumann(), N_sd),
                     ('mf_tend_q_tot'          , Center() , Neumann(), N_sd),
                     ('mf_tmp'                 , Center() , Neumann(), N_sd),
                     ('tmp_n'                  , Node() , Neumann(), N_sd),
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

        n_ghost = 1
        self.grid         = Grid(z_min, z_max, n_elems_real, n_ghost)
        self.q            = StateVec(unkowns, self.grid)
        self.q_new        = copy.deepcopy(self.q)
        self.q_old        = copy.deepcopy(self.q)
        self.q_tendencies = copy.deepcopy(self.q)
        self.tmp          = StateVec(temp_vars, self.grid)

        self.tmp_O2 = {}
        self.tmp_O2['tke']            = StateVec(q_2MO, self.grid)
        self.tmp_O2['tke']            = copy.deepcopy(self.tmp_O2['tke'])

        self.n_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']

        self.Ref        = ReferenceState(self.grid)
        self.Case       = CasesFactory(namelist, paramlist)
        self.Turb       = EDMF_PrognosticTKE(namelist, paramlist, self.grid)
        self.UpdVar     = [UpdraftVariables(i, self.Turb.surface_area, self.n_updrafts) for i in range(self.n_updrafts)]
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
        i_gm, i_env, i_uds, i_sd = self.q.domain_idx()
        self.q_tendencies.assign(self.grid, ('u', 'v', 'q_tot', 'θ_liq'), 0.0)
        self.Case.update_surface(self.grid, self.q, self.TS, self.tmp)
        self.Case.update_forcing(self.grid, self.q, self.q_tendencies, self.TS, self.tmp)

        self.q.export_state(self.grid, "./", "Q_py")
        self.tmp.export_state(self.grid, "./", "tmp_py")
        # raise NameError("Exported data!")

        self.Turb.initialize_vars(self.grid, self.q, self.q_tendencies, self.tmp, self.tmp_O2,
        self.UpdVar, self.Case, self.TS, self.tri_diag)
        for k in self.grid.over_elems(Center()):
            self.q['tke', i_env][k]            = self.q['tke', i_gm][k]

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

        i_gm, i_env, i_uds, i_sd = self.q.domain_idx()

        sol.e_W              = self.q['w', i_env]
        sol.e_q_tot          = self.q['q_tot', i_env]
        sol.e_θ_liq          = self.q['θ_liq', i_env]
        sol.e_q_liq          = self.tmp['q_liq', i_env]
        sol.e_T              = self.tmp['T', i_env]
        sol.e_B              = self.tmp['B', i_env]
        sol.e_CF             = self.tmp['CF']
        sol.e_tke            = self.q['tke', i_env]

        sol.ud_W     = self.q['w', i_uds[0]]
        sol.ud_Area  = self.q['a', i_uds[0]]
        sol.ud_q_tot = self.q['q_tot', i_uds[0]]
        sol.ud_q_liq = self.tmp['q_liq', i_uds[0]]
        sol.ud_T     = self.tmp['T', i_uds[0]]
        sol.ud_B     = self.tmp['B', i_uds[0]]

        sol.gm_q_tot = self.q['q_tot', i_gm]
        sol.gm_U     = self.q['u', i_gm]
        sol.gm_θ_liq = self.q['θ_liq', i_gm]
        sol.gm_T     = self.tmp['T', i_gm]
        sol.gm_V     = self.q['v', i_gm]
        sol.gm_q_liq = self.tmp['q_liq', i_gm]
        sol.gm_B     = self.tmp['B', i_gm]

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
                    'heaviside_a',
                    'heaviside_w',
                    # 'gov_eq_θ_liq_nb',
                    # 'gov_eq_q_tot_nb',
                    # 'gov_eq_θ_liq_ib',
                    # 'gov_eq_q_tot_ib',
                    'B',
                    'CF',
                    )
        plot_solutions_new(self.grid, self.q, q_vars, self.Stats)
        plot_solutions_new(self.grid, self.tmp, tmp_vars, self.Stats)

        i = i_uds[0]
        var_name = 'gov_eq_q_tot'
        plt.plot(self.tmp['gov_eq_q_tot_ib', i], self.grid.z,
                 self.tmp['gov_eq_q_tot_nb', i], self.grid.z)
        plt.legend(['ib','nb'])
        p_nice = nice_name(var_name+'_comparison')
        file_name = p_nice+'.png'
        plt.title(p_nice+' vs z')
        plt.xlabel(p_nice)
        plt.ylabel('z')
        plt.savefig(self.Stats.figpath+file_name)
        plt.close()

        n = 3
        var_name = 'gov_eq_θ_liq'
        plt.plot(self.tmp['gov_eq_θ_liq_ib', i][n:-n], self.grid.z[n:-n],
                 self.tmp['gov_eq_θ_liq_nb', i][n:-n], self.grid.z[n:-n])
        plt.legend(['ib','nb'])
        p_nice = nice_name(var_name+'_comparison')
        file_name = p_nice+'.png'
        plt.title(p_nice+' vs z')
        plt.xlabel(p_nice)
        plt.ylabel('z')
        plt.savefig(self.Stats.figpath+file_name)
        plt.close()

        # plot_solutions(sol, self.Stats)
        return sol


    def initialize_io(self):
        for v in self.q.var_names:
          for i in self.q.over_sub_domains(v):
            self.Stats.add_profile(v+'_'+self.q.idx_name(i))

        for v in self.tmp.var_names:
          for i in self.tmp.over_sub_domains(v):
            self.Stats.add_profile(v+'_'+self.tmp.idx_name(i))

        for k in self.tmp_O2:
          q_local = self.tmp_O2[k]
          for v in q_local.var_names:
            for i in q_local.over_sub_domains(v):
              self.Stats.add_profile(k+'_'+v+'_'+q_local.idx_name(i))

        self.Stats.add_ts('lwp')
        self.Case.initialize_io(self.Stats)
        initialize_io_updrafts(self.UpdVar, self.Stats)
        return

    def export_data(self):
        i_gm, i_env, i_uds, i_sd = self.q.domain_idx()
        self.Stats.open_files()
        self.Stats.write_simulation_time(self.TS.t)
        self.Case.export_data(self.Stats)
        pre_export_data_compute(self.grid, self.q, self.tmp, self.tmp_O2, self.Stats, self.Turb.tke_diss_coeff)
        export_data_updrafts(self.grid, self.UpdVar, self.q, self.tmp, self.Stats)

        lwp = 0.0
        for k in self.grid.over_elems_real(Center()):
            lwp += self.tmp['ρ_0'][k]*self.tmp['q_liq', i_gm][k]*self.grid.dz
        self.Stats.write_ts('lwp', lwp)

        for v in self.q.var_names:
          for i in self.q.over_sub_domains(v):
            self.Stats.write_profile_new(v+'_'+self.q.idx_name(i), self.grid, self.q[v, i])

        for v in self.tmp.var_names:
          for i in self.tmp.over_sub_domains(v):
            self.Stats.write_profile_new(v+'_'+self.tmp.idx_name(i), self.grid, self.tmp[v, i])

        for k in self.tmp_O2:
          q_local = self.tmp_O2[k]
          for v in q_local.var_names:
            for i in q_local.over_sub_domains(v):
              self.Stats.write_profile_new(k+'_'+v+'_'+q_local.idx_name(i), self.grid, q_local[v])

        self.Stats.close_files()
        return

    def force_io(self):
        return
