import numpy as np
from parameters import *
import sys
from EDMF_Updrafts import *
from EDMF_Environment import *
from Operators import advect, grad, Laplacian, grad_pos, grad_neg
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann

from TriDiagSolver import tridiag_solve, tridiag_solve_wrapper, construct_tridiag_diffusion_new_new, tridiag_solve_wrapper_new
from Variables import VariablePrognostic, VariableDiagnostic, GridMeanVariables
from Surface import SurfaceBase
from Cases import  CasesBase
from ReferenceState import  ReferenceState
from TimeStepping import TimeStepping
from NetCDFIO import NetCDFIO_Stats
from funcs_thermo import  *
from funcs_turbulence import *
from funcs_utility import *

def compute_inversion(grid, GMV, option, tmp, Ri_bulk_crit):
    theta_rho = Half(grid)
    maxgrad = 0.0
    theta_rho_bl = theta_rho.surface_bl(grid)
    for k in grid.over_elems_real(Center()):
        q_tot = GMV.q_tot.values[k]
        q_vap = q_tot - GMV.q_liq.values[k]
        theta_rho[k] = theta_rho_c(tmp['p_0_half'][k], GMV.T.values[k], q_tot, q_vap)
    if option == 'theta_rho':
        for k in grid.over_elems_real(Center()):
            if theta_rho[k] > theta_rho_bl:
                zi = grid.z_half[k]
                break
    elif option == 'thetal_maxgrad':
        for k in grid.over_elems_real(Center()):
            grad_TH = grad(GMV.θ_liq.values.Dual(k), grid)
            if grad_TH > maxgrad:
                maxgrad = grad_TH
                zi = grid.z[k]
    elif option == 'critical_Ri':
        zi = get_inversion(theta_rho, GMV.U.values, GMV.V.values, grid, Ri_bulk_crit)
    else:
        print('INVERSION HEIGHT OPTION NOT RECOGNIZED')
    return zi


def ParameterizationFactory(namelist, paramlist, grid, Ref):
    return EDMF_PrognosticTKE(namelist, paramlist, grid, Ref)

class ParameterizationBase:
    def __init__(self, paramlist, grid, Ref):
        self.turbulence_tendency = Half(grid)
        self.Ref = Ref
        self.KM = VariableDiagnostic(grid, Center(), Neumann())
        self.KH = VariableDiagnostic(grid, Center(), Neumann())
        self.prandtl_number = paramlist['turbulence']['prandtl_number']
        self.Ri_bulk_crit = paramlist['turbulence']['Ri_bulk_crit']
        return

    def update(self, grid, q, tmp, GMV, Case, TS):
        for k in grid.over_elems_real(Center()):
            GMV.H.tendencies[k] += (GMV.H.new[k] - GMV.H.values[k]) * TS.dti
            GMV.q_tot.tendencies[k] += (GMV.q_tot.new[k] - GMV.q_tot.values[k]) * TS.dti
            GMV.U.tendencies[k] += (GMV.U.new[k] - GMV.U.values[k]) * TS.dti
            GMV.V.tendencies[k] += (GMV.V.new[k] - GMV.V.values[k]) * TS.dti
        return

    # Compute eddy diffusivities from similarity theory (Siebesma 2007)
    def compute_eddy_diffusivities_similarity(self, grid, GMV, Case, tmp):
        self.zi = compute_inversion(grid, GMV, Case.inversion_option, tmp, self.Ri_bulk_crit)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)
        ustar = Case.Sur.ustar
        for k in grid.over_elems_real(Center()):
            zzi = grid.z_half[k]/self.zi
            self.KH.values[k] = 0.0
            self.KM.values[k] = 0.0
            if zzi <= 1.0 and not (self.wstar<1e-6):
                self.KH.values[k] = vkb * ( (ustar/self.wstar)**3 + 39.0*vkb*zzi)**(1.0/3.0) * zzi * (1.0-zzi) * (1.0-zzi) * self.wstar * self.zi
                self.KM.values[k] = self.KH.values[k] * self.prandtl_number
        return

class EDMF_PrognosticTKE(ParameterizationBase):
    def __init__(self, namelist, paramlist, grid, Ref):
        ParameterizationBase.__init__(self, paramlist,  grid, Ref)

        self.n_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']

        try:
            self.use_steady_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['use_steady_updrafts']
        except:
            self.use_steady_updrafts = False
        try:
            self.use_local_micro = namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro']
        except:
            self.use_local_micro = True
            print('Turbulence--EDMF_PrognosticTKE: defaulting to local (level-by-level) microphysics')

        try:
            if str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'inverse_z':
                self.entr_detr_fp = entr_detr_inverse_z
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'dry':
                self.entr_detr_fp = entr_detr_dry
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'inverse_w':
                self.entr_detr_fp = entr_detr_inverse_w
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'b_w2':
                self.entr_detr_fp = entr_detr_b_w2
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'entr_detr_tke':
                self.entr_detr_fp = entr_detr_tke
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'entr_detr_tke2':
                self.entr_detr_fp = entr_detr_tke2
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'suselj':
                self.entr_detr_fp = entr_detr_suselj
            elif str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']) == 'none':
                self.entr_detr_fp = entr_detr_none
            else:
                print('Turbulence--EDMF_PrognosticTKE: Entrainment rate namelist option is not recognized')
        except:
            self.entr_detr_fp = entr_detr_b_w2
            print('Turbulence--EDMF_PrognosticTKE: defaulting to cloudy entrainment formulation')

        try:
            self.similarity_diffusivity = namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity']
        except:
            self.similarity_diffusivity = False
            print('Turbulence--EDMF_PrognosticTKE: defaulting to tke-based eddy diffusivity')

        try:
            self.extrapolate_buoyancy = namelist['turbulence']['EDMF_PrognosticTKE']['extrapolate_buoyancy']
        except:
            self.extrapolate_buoyancy = True
            print('Turbulence--EDMF_PrognosticTKE: defaulting to extrapolation of updraft buoyancy along a pseudoadiabat')

        try:
            self.mixing_scheme = str(namelist['turbulence']['EDMF_PrognosticTKE']['mixing_length'])
        except:
            self.mixing_scheme = 'tke'
            print('Using tke mixing length formulation as default')

        # Get values from paramlist
        # set defaults at some point?
        self.surface_area = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']
        self.max_area_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor']
        self.entrainment_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor']
        self.detrainment_factor = paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor']
        self.pressure_buoy_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff']
        self.pressure_drag_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_drag_coeff']
        self.pressure_plume_spacing = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_plume_spacing']
        self.vel_pressure_coeff = self.pressure_drag_coeff/self.pressure_plume_spacing
        self.vel_buoy_coeff = 1.0-self.pressure_buoy_coeff
        self.tke_ed_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff']
        self.tke_diss_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff']

        # Need to code up as paramlist option?
        self.minimum_area = 1e-3

        a_ = self.surface_area/self.n_updrafts
        i_uds = range(self.n_updrafts)
        self.surface_scalar_coeff = np.zeros((self.n_updrafts,), dtype=np.double, order='c')
        # i_gm, i_env, i_ud = tmp.domain_idx()
        for i in i_uds:
            self.surface_scalar_coeff[i] = percentile_bounds_mean_norm(1.0-self.surface_area+i*a_,
                                                                       1.0-self.surface_area + (i+1)*a_ , 1000)

        # Entrainment/Detrainment rates
        self.entr_sc = [Half(grid) for i in i_uds]
        self.detr_sc = [Half(grid) for i in i_uds]

        # Mass flux
        self.m = [Full(grid) for i in i_uds]
        self.mixing_length = Half(grid)

        # Near-surface BC of updraft area fraction
        self.area_surface_bc = np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.w_surface_bc    = np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.h_surface_bc    = np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.qt_surface_bc   = np.zeros((self.n_updrafts,),dtype=np.double, order='c')

        # Mass flux tendencies of mean scalars (for output)
        self.massflux_tendency_h = Half(grid)
        self.massflux_tendency_qt = Half(grid)

        # Vertical fluxes for output
        self.massflux_h = Full(grid)
        self.massflux_qt = Full(grid)

        self.mls = Half(grid)
        self.ml_ratio = Half(grid)
        return

    def initialize(self, GMV, UpdVar, tmp, q):
        UpdVar.initialize(GMV, tmp, q)
        return

    # Initialize the IO pertaining to this class
    def initialize_io(self, Stats, EnvVar, UpdVar):

        UpdVar.initialize_io(Stats)
        EnvVar.initialize_io(Stats)

        Stats.add_profile('eddy_viscosity')
        Stats.add_profile('eddy_diffusivity')
        Stats.add_profile('entrainment_sc')
        Stats.add_profile('detrainment_sc')
        Stats.add_profile('massflux')
        Stats.add_profile('massflux_h')
        Stats.add_profile('massflux_qt')
        Stats.add_profile('massflux_tendency_h')
        Stats.add_profile('massflux_tendency_qt')
        Stats.add_profile('mixing_length')
        Stats.add_profile('updraft_qt_precip')
        Stats.add_profile('updraft_thetal_precip')

        Stats.add_profile('tke_buoy')
        Stats.add_profile('tke_dissipation')
        Stats.add_profile('tke_entr_gain')
        Stats.add_profile('tke_detr_loss')
        Stats.add_profile('tke_shear')
        Stats.add_profile('tke_pressure')
        Stats.add_profile('tke_interdomain')

        Stats.add_profile('Hvar_dissipation')
        Stats.add_profile('QTvar_dissipation')
        Stats.add_profile('HQTcov_dissipation')
        Stats.add_profile('Hvar_entr_gain')
        Stats.add_profile('QTvar_entr_gain')
        Stats.add_profile('Hvar_detr_loss')
        Stats.add_profile('QTvar_detr_loss')
        Stats.add_profile('HQTcov_detr_loss')
        Stats.add_profile('HQTcov_entr_gain')
        Stats.add_profile('Hvar_shear')
        Stats.add_profile('QTvar_shear')
        Stats.add_profile('HQTcov_shear')
        Stats.add_profile('Hvar_rain')
        Stats.add_profile('QTvar_rain')
        Stats.add_profile('HQTcov_rain')
        Stats.add_profile('Hvar_interdomain')
        Stats.add_profile('QTvar_interdomain')
        Stats.add_profile('HQTcov_interdomain')
        return

    def io(self, grid, q, tmp, Stats, EnvVar, UpdVar, UpdMicro):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        mean_entr_sc = Half(grid)
        mean_detr_sc = Half(grid)
        massflux     = Half(grid)
        mf_h         = Half(grid)
        mf_qt        = Half(grid)

        UpdVar.io(grid, Stats)
        EnvVar.io(grid, Stats)

        Stats.write_profile_new('eddy_viscosity'  , grid, self.KM.values)
        Stats.write_profile_new('eddy_diffusivity', grid, self.KH.values)
        for k in grid.over_elems_real(Center()):
            mf_h[k] = self.massflux_h.Mid(k)
            mf_qt[k] = self.massflux_qt.Mid(k)
            massflux[k] = self.m[0].Mid(k)
            a_bulk = sum([q['a', i][k] for i in i_uds])
            if a_bulk > 0.0:
                for i in i_uds:
                    mean_entr_sc[k] += q['a', i][k] * self.entr_sc[i][k]/a_bulk
                    mean_detr_sc[k] += q['a', i][k] * self.detr_sc[i][k]/a_bulk

        Stats.write_profile_new('entrainment_sc', grid, mean_entr_sc)
        Stats.write_profile_new('detrainment_sc', grid, mean_detr_sc)
        Stats.write_profile_new('massflux'      , grid, massflux)
        Stats.write_profile_new('massflux_h'    , grid, mf_h)
        Stats.write_profile_new('massflux_qt'   , grid, mf_qt)
        Stats.write_profile_new('massflux_tendency_h'  , grid, self.massflux_tendency_h)
        Stats.write_profile_new('massflux_tendency_qt' , grid, self.massflux_tendency_qt)
        Stats.write_profile_new('mixing_length'        , grid, self.mixing_length)
        Stats.write_profile_new('updraft_qt_precip'    , grid, UpdMicro.prec_source_qt_tot)
        Stats.write_profile_new('updraft_thetal_precip', grid, UpdMicro.prec_source_h_tot)

        self.compute_covariance_dissipation(grid, q, tmp, EnvVar.tke, EnvVar)
        Stats.write_profile_new('tke_dissipation', grid, EnvVar.tke.dissipation)
        Stats.write_profile_new('tke_entr_gain'  , grid, EnvVar.tke.entr_gain)
        self.compute_covariance_detr(grid, q, tmp, EnvVar.tke, UpdVar)
        Stats.write_profile_new('tke_detr_loss'  , grid, EnvVar.tke.detr_loss)
        Stats.write_profile_new('tke_shear'      , grid, EnvVar.tke.shear)
        Stats.write_profile_new('tke_buoy'       , grid, EnvVar.tke.buoy)
        Stats.write_profile_new('tke_pressure'   , grid, EnvVar.tke.press)
        Stats.write_profile_new('tke_interdomain', grid, EnvVar.tke.interdomain)

        self.compute_covariance_dissipation(grid, q, tmp, EnvVar.cv_θ_liq, EnvVar)
        Stats.write_profile_new('Hvar_dissipation'  , grid, EnvVar.cv_θ_liq.dissipation)
        self.compute_covariance_dissipation(grid, q, tmp, EnvVar.cv_q_tot, EnvVar)
        Stats.write_profile_new('QTvar_dissipation' , grid, EnvVar.cv_q_tot.dissipation)
        self.compute_covariance_dissipation(grid, q, tmp, EnvVar.cv_θ_liq_q_tot, EnvVar)
        Stats.write_profile_new('HQTcov_dissipation', grid, EnvVar.cv_θ_liq_q_tot.dissipation)
        Stats.write_profile_new('Hvar_entr_gain'    , grid, EnvVar.cv_θ_liq.entr_gain)
        Stats.write_profile_new('QTvar_entr_gain'   , grid, EnvVar.cv_q_tot.entr_gain)
        Stats.write_profile_new('HQTcov_entr_gain'  , grid, EnvVar.cv_θ_liq_q_tot.entr_gain)
        self.compute_covariance_detr(grid, q, tmp, EnvVar.cv_θ_liq, UpdVar)
        self.compute_covariance_detr(grid, q, tmp, EnvVar.cv_q_tot, UpdVar)
        self.compute_covariance_detr(grid, q, tmp, EnvVar.cv_θ_liq_q_tot, UpdVar)
        Stats.write_profile_new('Hvar_detr_loss'    , grid, EnvVar.cv_θ_liq.detr_loss)
        Stats.write_profile_new('QTvar_detr_loss'   , grid, EnvVar.cv_q_tot.detr_loss)
        Stats.write_profile_new('HQTcov_detr_loss'  , grid, EnvVar.cv_θ_liq_q_tot.detr_loss)
        Stats.write_profile_new('Hvar_shear'        , grid, EnvVar.cv_θ_liq.shear)
        Stats.write_profile_new('QTvar_shear'       , grid, EnvVar.cv_q_tot.shear)
        Stats.write_profile_new('HQTcov_shear'      , grid, EnvVar.cv_θ_liq_q_tot.shear)
        Stats.write_profile_new('Hvar_rain'         , grid, EnvVar.cv_θ_liq.rain_src)
        Stats.write_profile_new('QTvar_rain'        , grid, EnvVar.cv_q_tot.rain_src)
        Stats.write_profile_new('HQTcov_rain'       , grid, EnvVar.cv_θ_liq_q_tot.rain_src)
        Stats.write_profile_new('Hvar_interdomain'  , grid, EnvVar.cv_θ_liq.interdomain)
        Stats.write_profile_new('QTvar_interdomain' , grid, EnvVar.cv_q_tot.interdomain)
        Stats.write_profile_new('HQTcov_interdomain', grid, EnvVar.cv_θ_liq_q_tot.interdomain)
        return

    def update(self, grid, q, tmp, GMV, EnvVar, UpdVar, UpdMicro, EnvThermo, UpdThermo, Ref, Case, TS, tri_diag):

        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        self.zi = compute_inversion(grid, GMV, Case.inversion_option, tmp, self.Ri_bulk_crit)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)
        self.decompose_environment(grid, q, GMV, EnvVar, UpdVar)
        self.get_GMV_CoVar(grid, q, UpdVar.Area.values, UpdVar.W.values,  UpdVar.W.values,  q['w', i_env],  q['w', i_env],  EnvVar.tke.values,    GMV.W.values,  GMV.W.values,  GMV.tke.values, 'tke')
        self.get_GMV_CoVar(grid, q, UpdVar.Area.values, UpdVar.H.values,  UpdVar.H.values,  EnvVar.H.values,  EnvVar.H.values,  EnvVar.cv_θ_liq.values,   GMV.H.values,  GMV.H.values,  GMV.cv_θ_liq.values, '')
        self.get_GMV_CoVar(grid, q, UpdVar.Area.values, UpdVar.q_tot.values, UpdVar.q_tot.values, EnvVar.q_tot.values, EnvVar.q_tot.values, EnvVar.cv_q_tot.values,  GMV.q_tot.values, GMV.q_tot.values, GMV.cv_q_tot.values, '')
        self.get_GMV_CoVar(grid, q, UpdVar.Area.values, UpdVar.H.values,  UpdVar.q_tot.values, EnvVar.H.values,  EnvVar.q_tot.values, EnvVar.cv_θ_liq_q_tot.values, GMV.H.values,  GMV.q_tot.values, GMV.cv_θ_liq_q_tot.values, '')
        self.update_GMV_MF(grid, q, GMV, EnvVar, UpdVar, TS, tmp)
        self.compute_eddy_diffusivities_tke(grid, GMV, EnvVar, Case)
        self.compute_covariance(grid, Ref, q, GMV, EnvVar, UpdVar, EnvThermo, Case, TS, tmp, tri_diag)

        self.compute_prognostic_updrafts(grid, q, tmp, GMV, EnvVar, UpdVar, UpdMicro, EnvThermo, UpdThermo, Case, TS)

        self.update_GMV_ED(grid, q, GMV, UpdMicro, Case, TS, tmp, tri_diag)

        ParameterizationBase.update(self, grid, q, tmp, GMV, Case, TS)
        GMV.H.set_bcs(grid)
        GMV.q_tot.set_bcs(grid)
        GMV.q_rai.set_bcs(grid)
        GMV.U.set_bcs(grid)
        GMV.V.set_bcs(grid)

        return

    def compute_prognostic_updrafts(self, grid, q, tmp, GMV, EnvVar, UpdVar, UpdMicro, EnvThermo, UpdThermo, Case, TS):
        time_elapsed = 0.0
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        UpdVar.set_new_with_values(grid)
        UpdVar.set_old_with_values(grid)
        self.set_updraft_surface_bc(grid, GMV, Case, tmp)
        self.dt_upd = np.minimum(TS.dt, 0.5 * grid.dz/np.fmax(np.max(UpdVar.W.values),1e-10))
        while time_elapsed < TS.dt:
            self.compute_entrainment_detrainment(grid, GMV, EnvVar, UpdVar, Case, tmp, q)
            EnvThermo.eos_update_SA_mean(grid, EnvVar, False, tmp)
            UpdThermo.buoyancy(grid, q, tmp, UpdVar, EnvVar, GMV, self.extrapolate_buoyancy)
            UpdMicro.compute_sources(grid, UpdVar, tmp)
            UpdMicro.update_updraftvars(grid, UpdVar)

            self.solve_updraft_velocity_area(grid, q, tmp, GMV, UpdVar, TS)
            self.solve_updraft_scalars(grid, q, tmp, GMV, EnvVar, UpdVar, UpdMicro, TS)
            UpdVar.H.set_bcs(grid)
            UpdVar.q_tot.set_bcs(grid)
            UpdVar.q_rai.set_bcs(grid)
            q['w', i_env].apply_bc(grid, 0.0)
            EnvVar.H.set_bcs(grid)
            EnvVar.q_tot.set_bcs(grid)
            UpdVar.set_values_with_new(grid)
            time_elapsed += self.dt_upd
            self.dt_upd = np.minimum(TS.dt-time_elapsed,  0.5 * grid.dz/np.fmax(np.max(UpdVar.W.values),1e-10))
            self.decompose_environment(grid, q, GMV, EnvVar, UpdVar)
        EnvThermo.eos_update_SA_mean(grid, EnvVar, True, tmp)
        UpdThermo.buoyancy(grid, q, tmp, UpdVar, EnvVar, GMV, self.extrapolate_buoyancy)
        return

    def compute_mixing_length(self, grid, obukhov_length, EnvVar):
        tau = get_mixing_tau(self.zi, self.wstar)
        for k in grid.over_elems_real(Center()):
            l1 = tau * np.sqrt(np.fmax(EnvVar.tke.values[k],0.0))
            z_ = grid.z_half[k]
            if obukhov_length < 0.0: #unstable
                l2 = vkb * z_ * ( (1.0 - 100.0 * z_/obukhov_length)**0.2 )
            elif obukhov_length > 0.0: #stable
                l2 = vkb * z_ /  (1. + 2.7 *z_/obukhov_length)
            else:
                l2 = vkb * z_
            self.mixing_length[k] = np.fmax( 1.0/(1.0/np.fmax(l1,1e-10) + 1.0/l2), 1e-3)
        return


    def compute_eddy_diffusivities_tke(self, grid, GMV, EnvVar, Case):
        if self.similarity_diffusivity:
            ParameterizationBase.compute_eddy_diffusivities_similarity(self, grid, GMV, Case)
        else:
            self.compute_mixing_length(grid, Case.Sur.obukhov_length, EnvVar)
            for k in grid.over_elems_real(Center()):
                lm = self.mixing_length[k]
                self.KM.values[k] = self.tke_ed_coeff * lm * np.sqrt(np.fmax(EnvVar.tke.values[k],0.0) )
                self.KH.values[k] = self.KM.values[k] / self.prandtl_number
        return

    def set_updraft_surface_bc(self, grid, GMV, Case, tmp):
        i_gm, i_env, i_uds, i_sd = tmp.domain_idx()
        self.zi = compute_inversion(grid, GMV, Case.inversion_option, tmp, self.Ri_bulk_crit)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)
        k_1 = grid.first_interior(Zmin())
        zLL = grid.z_half[k_1]
        H_1 = GMV.H.values[k_1]
        QT_1 = GMV.q_tot.values[k_1]
        alpha0LL  = tmp['α_0_half'][k_1]
        ustar = Case.Sur.ustar
        oblength = Case.Sur.obukhov_length
        cv_q_tot = get_surface_variance(Case.Sur.rho_qtflux*alpha0LL, Case.Sur.rho_qtflux*alpha0LL, ustar, zLL, oblength)
        h_var  = get_surface_variance(Case.Sur.rho_hflux*alpha0LL,  Case.Sur.rho_hflux*alpha0LL,  ustar, zLL, oblength)
        for i in i_uds:
            self.area_surface_bc[i] = self.surface_area/self.n_updrafts
            self.w_surface_bc[i] = 0.0
            self.h_surface_bc[i] = (H_1 + self.surface_scalar_coeff[i] * np.sqrt(h_var))
            self.qt_surface_bc[i] = (QT_1 + self.surface_scalar_coeff[i] * np.sqrt(cv_q_tot))
        return

    def reset_surface_covariance(self, grid, Ref, q, tmp, GMV, Case):
        flux1 = Case.Sur.rho_hflux
        flux2 = Case.Sur.rho_qtflux
        k_1 = grid.first_interior(Zmin())
        zLL = grid.z_half[k_1]
        alpha0LL  = Ref.alpha0_half[k_1]
        ustar = Case.Sur.ustar
        oblength = Case.Sur.obukhov_length
        GMV.tke.values[k_1]            = get_surface_tke(Case.Sur.ustar, self.wstar, zLL, Case.Sur.obukhov_length)
        GMV.cv_θ_liq.values[k_1]       = get_surface_variance(flux1*alpha0LL,flux1*alpha0LL, ustar, zLL, oblength)
        GMV.cv_q_tot.values[k_1]       = get_surface_variance(flux2*alpha0LL,flux2*alpha0LL, ustar, zLL, oblength)
        GMV.cv_θ_liq_q_tot.values[k_1] = get_surface_variance(flux1*alpha0LL,flux2*alpha0LL, ustar, zLL, oblength)
        return

    # Find values of environmental variables by subtracting updraft values from grid mean values
    def decompose_environment(self, grid, q, GMV, EnvVar, UpdVar):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        for k in grid.over_elems(Center()):
            a_env = q['a', i_env][k]
            EnvVar.q_tot.values[k] = (GMV.q_tot.values[k] - sum([q['a', i][k]*UpdVar.q_tot.values[i][k] for i in i_uds]))/a_env
            EnvVar.H.values[k]     = (GMV.H.values[k]     - sum([q['a', i][k]*UpdVar.H.values[i][k] for i in i_uds]))/a_env
            # Assuming GMV.W = 0!
            a_env = q['a', i_env].Mid(k)
            q['w', i_env][k] = (0.0 - sum([q['a', i][k]*UpdVar.W.values[i][k] for i in i_uds]))/a_env
        return

    def get_GMV_CoVar(self, grid, q, au, phi_u, psi_u, phi_e,  psi_e, covar_e, gmv_phi, gmv_psi, gmv_covar, name):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        tke_factor = 1.0
        ae = q['a', i_env]

        for k in grid.over_elems(Center()):
            if name == 'tke':
                tke_factor = 0.5
                phi_diff = phi_e.Mid(k) - gmv_phi.Mid(k)
                psi_diff = psi_e.Mid(k) - gmv_psi.Mid(k)
            else:
                phi_diff = phi_e[k]-gmv_phi[k]
                psi_diff = psi_e[k]-gmv_psi[k]

            gmv_covar[k] = tke_factor * ae[k] * phi_diff * psi_diff + ae[k] * covar_e[k]
            for i in i_uds:
                if name == 'tke':
                    phi_diff = phi_u[i].Mid(k) - gmv_phi.Mid(k)
                    psi_diff = psi_u[i].Mid(k) - gmv_psi.Mid(k)
                else:
                    phi_diff = phi_u[i][k]-gmv_phi[k]
                    psi_diff = psi_u[i][k]-gmv_psi[k]

                gmv_covar[k] += tke_factor * au[i][k] * phi_diff * psi_diff
        return


    def get_env_covar_from_GMV(self, grid, q, au, phi_u, psi_u, phi_e, psi_e, covar_e, gmv_phi, gmv_psi, gmv_covar, name):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        tke_factor = 1.0
        ae = q['a', i_env]

        for k in grid.over_elems(Center()):
            if ae[k] > 0.0:
                if name == 'tke':
                    tke_factor = 0.5
                    phi_diff = phi_e.Mid(k) - gmv_phi.values.Mid(k)
                    psi_diff = psi_e.Mid(k) - gmv_psi.values.Mid(k)
                else:
                    phi_diff = phi_e[k] - gmv_phi.values[k]
                    psi_diff = psi_e[k] - gmv_psi.values[k]

                covar_e.values[k] = gmv_covar.values[k] - tke_factor * ae[k] * phi_diff * psi_diff
                for i in i_uds:
                    if name == 'tke':
                        phi_diff = phi_u.values[i].Mid(k) - gmv_phi.values.Mid(k)
                        psi_diff = psi_u.values[i].Mid(k) - gmv_psi.values.Mid(k)
                    else:
                        phi_diff = phi_u.values[i][k] - gmv_phi.values[k]
                        psi_diff = psi_u.values[i][k] - gmv_psi.values[k]

                    covar_e.values[k] -= tke_factor * au.values[i][k] * phi_diff * psi_diff
                covar_e.values[k] = covar_e.values[k]/ae[k]
            else:
                covar_e.values[k] = 0.0
        return

    def compute_entrainment_detrainment(self, grid, GMV, EnvVar, UpdVar, Case, tmp, q):
        quadrature_order = 3
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        UpdVar.get_cloud_base_top_cover(grid)

        input_st = type('', (), {})()
        input_st.wstar = self.wstar

        input_st.b_mean = 0
        input_st.dz = grid.dz
        input_st.zbl = self.compute_zbl_qt_grad(grid, GMV)
        for i in i_uds:
            input_st.zi = UpdVar.cloud_base[i]
            for k in grid.over_elems_real(Center()):
                input_st.quadrature_order = quadrature_order
                input_st.z = grid.z_half[k]
                input_st.ml = self.mixing_length[k]
                input_st.b = UpdVar.B.values[i][k]
                input_st.w = UpdVar.W.values[i].Mid(k)
                input_st.af = UpdVar.Area.values[i][k]
                input_st.tke = EnvVar.tke.values[k]
                input_st.qt_env = EnvVar.q_tot.values[k]
                input_st.ql_env = EnvVar.q_liq.values[k]
                input_st.H_env = EnvVar.H.values[k]
                input_st.b_env = EnvVar.B.values[k]
                input_st.w_env = q['w', i_env].values[k]
                input_st.H_up = UpdVar.H.values[i][k]
                input_st.qt_up = UpdVar.q_tot.values[i][k]
                input_st.ql_up = UpdVar.q_liq.values[i][k]
                input_st.env_Hvar = EnvVar.cv_θ_liq.values[k]
                input_st.env_QTvar = EnvVar.cv_q_tot.values[k]
                input_st.env_HQTcov = EnvVar.cv_θ_liq_q_tot.values[k]
                input_st.p0 = tmp['p_0_half'][k]
                input_st.alpha0 = tmp['α_0_half'][k]
                input_st.tke = EnvVar.tke.values[k]
                input_st.tke_ed_coeff  = self.tke_ed_coeff

                input_st.L = 20000.0 # need to define the scale of the GCM grid resolution
                input_st.n_up = self.n_updrafts

                w_cut = UpdVar.W.values[i].DualCut(k)
                w_env_cut = q['w', i_env].DualCut(k)
                a_cut = UpdVar.Area.values[i].Cut(k)
                a_env_cut = (1.0-UpdVar.Area.values[i].Cut(k))
                aw_cut = a_cut * w_cut + a_env_cut * w_env_cut

                input_st.dwdz = grad(aw_cut, grid)

                if input_st.zbl-UpdVar.cloud_base[i] > 0.0:
                    input_st.poisson = np.random.poisson(grid.dz/((input_st.zbl-UpdVar.cloud_base[i])/10.0))
                else:
                    input_st.poisson = 0.0
                ret = self.entr_detr_fp(input_st)
                self.entr_sc[i][k] = ret.entr_sc * self.entrainment_factor
                self.detr_sc[i][k] = ret.detr_sc * self.detrainment_factor

        return

    def compute_zbl_qt_grad(self, grid, GMV):
        # computes inversion height as z with max gradient of qt
        zbl_qt = 0.0
        qt_grad = 0.0
        for k in grid.over_elems_real(Center()):
            qt_grad_new = grad(GMV.q_tot.values.Dual(k), grid)
            if np.fabs(qt_grad) > qt_grad:
                qt_grad = np.fabs(qt_grad_new)
                zbl_qt = grid.z_half[k]
        return zbl_qt

    def solve_updraft_velocity_area(self, grid, q, tmp, GMV, UpdVar, TS):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        k_1 = grid.first_interior(Zmin())
        kb_1 = grid.boundary(Zmin())
        dzi = grid.dzi
        dti_ = 1.0/self.dt_upd
        dt_ = 1.0/dti_

        # Solve for area fraction
        for i in i_uds:
            au_lim = self.area_surface_bc[i] * self.max_area_factor
            for k in grid.over_elems_real(Center()):

                a_k = UpdVar.Area.values[i][k]
                α_0_kp = self.Ref.alpha0_half[k]
                w_k = UpdVar.W.values[i].Mid(k)

                w_cut = UpdVar.W.values[i].DualCut(k)
                a_cut = UpdVar.Area.values[i].Cut(k)
                ρ_cut = self.Ref.rho0_half.Cut(k)
                tendencies = 0.0

                ρaw_cut = ρ_cut*a_cut*w_cut
                adv = - α_0_kp * advect(ρaw_cut, w_cut, grid)
                tendencies+=adv

                ε_term = a_k * w_k * (+ self.entr_sc[i][k])
                tendencies+=ε_term
                δ_term = a_k * w_k * (- self.detr_sc[i][k])
                tendencies+=δ_term

                a_predict = a_k + dt_ * tendencies

                needs_limiter = a_predict>au_lim
                UpdVar.Area.new[i][k] = np.fmin(np.fmax(a_predict, 0.0), au_lim)

                unsteady = (UpdVar.Area.new[i][k]-a_k)*dti_
                # δ_limiter = unsteady - tendencies if needs_limiter else 0.0
                # tendencies+=δ_limiter
                # a_correct = a_k + dt_ * tendencies

                if needs_limiter:
                    δ_term_new = unsteady - adv - ε_term
                    if a_k > 0.0:
                        self.detr_sc[i][k] = δ_term_new/(-a_k  * w_k)
                    else:
                        self.detr_sc[i][k] = δ_term_new/(-au_lim  * w_k)

            self.entr_sc[i][k_1] = 2.0 * dzi
            self.detr_sc[i][k_1] = 0.0
            UpdVar.Area.new[i][k_1] = self.area_surface_bc[i]


        for k in grid.over_elems(Center()):
            for i in i_uds:
                q['a', i][k] = UpdVar.Area.new[i][k]
            q['a', i_env][k] = 1.0 - sum([UpdVar.Area.new[i][k] for i in i_uds])

        # Solve for updraft velocity
        for i in i_uds:
            UpdVar.W.new[i][kb_1] = self.w_surface_bc[i]
            for k in grid.over_elems_real(Center()):
                a_new_k = UpdVar.Area.new[i].Mid(k)
                if a_new_k >= self.minimum_area:

                    ρ_k = self.Ref.rho0[k]
                    w_i = UpdVar.W.values[i][k]
                    w_env = q['w', i_env].values[k]
                    a_k = UpdVar.Area.values[i].Mid(k)
                    entr_w = self.entr_sc[i].Mid(k)
                    detr_w = self.detr_sc[i].Mid(k)
                    B_k = UpdVar.B.values[i].Mid(k)

                    a_cut = UpdVar.Area.values[i].DualCut(k)
                    ρ_cut = self.Ref.rho0.Cut(k)
                    w_cut = UpdVar.W.values[i].Cut(k)

                    ρa_k = ρ_k * a_k
                    ρa_new_k = ρ_k * a_new_k
                    ρaw_k = ρa_k * w_i
                    ρaww_cut = ρ_cut*a_cut*w_cut*w_cut

                    adv = -advect(ρaww_cut, w_cut, grid)
                    exch = ρaw_k * (- detr_w * w_i + entr_w * w_env)
                    buoy = ρa_k * B_k
                    press_buoy = - ρa_k * B_k * self.pressure_buoy_coeff
                    press_drag = - ρa_k * (self.pressure_drag_coeff/self.pressure_plume_spacing * (w_i - w_env)**2.0/np.sqrt(np.fmax(a_k, self.minimum_area)))
                    nh_press = press_buoy + press_drag

                    UpdVar.W.new[i][k] = ρaw_k/ρa_new_k + dt_/ρa_new_k*(adv + exch + buoy + nh_press)

        # Filter results
        for i in i_uds:
            for k in grid.over_elems_real(Center()):
                if UpdVar.Area.new[i].Mid(k) >= self.minimum_area:
                    if UpdVar.W.new[i][k] <= 0.0:
                        UpdVar.W.new[i][k:] = 0.0
                        UpdVar.Area.new[i][k+1:] = 0.0
                        break
                else:
                    UpdVar.W.new[i][k:] = 0.0
                    UpdVar.Area.new[i][k+1:] = 0.0
                    break

        return

    def solve_updraft_scalars(self, grid, q, tmp, GMV, EnvVar, UpdVar, UpdMicro, TS):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        dzi = grid.dzi
        dti_ = 1.0/self.dt_upd
        k_1 = grid.first_interior(Zmin())

        for i in i_uds:
            UpdVar.H.new[i][k_1] = self.h_surface_bc[i]
            UpdVar.q_tot.new[i][k_1] = self.qt_surface_bc[i]

            for k in grid.over_elems_real(Center())[1:]:
                dt_ = 1.0/dti_
                H_env = EnvVar.H.values[k]
                QT_env = EnvVar.q_tot.values[k]

                if UpdVar.Area.new[i][k] >= self.minimum_area:
                    a_k = UpdVar.Area.values[i][k]
                    a_cut = UpdVar.Area.values[i].Cut(k)
                    a_k_new = UpdVar.Area.new[i][k]
                    H_cut = UpdVar.H.values[i].Cut(k)
                    QT_cut = UpdVar.q_tot.values[i].Cut(k)
                    ρ_k = self.Ref.rho0_half[k]
                    ρ_cut = self.Ref.rho0_half.Cut(k)
                    w_cut = UpdVar.W.values[i].DualCut(k)
                    ε_sc = self.entr_sc[i][k]
                    δ_sc = self.detr_sc[i][k]
                    ρa_k = ρ_k*a_k

                    ρaw_cut = ρ_cut * a_cut * w_cut
                    ρawH_cut = ρaw_cut * H_cut
                    ρawQT_cut = ρaw_cut * QT_cut
                    ρa_new_k = ρ_k * a_k_new

                    tendencies_H  = -advect(ρawH_cut , w_cut, grid) + ρaw_cut[1] * (ε_sc * H_env  - δ_sc * H_cut[1] )
                    tendencies_QT = -advect(ρawQT_cut, w_cut, grid) + ρaw_cut[1] * (ε_sc * QT_env - δ_sc * QT_cut[1])

                    UpdVar.H.new[i][k] = ρa_k/ρa_new_k * H_cut[1]  + dt_*tendencies_H/ρa_new_k
                    UpdVar.q_tot.new[i][k] = ρa_k/ρa_new_k * QT_cut[1] + dt_*tendencies_QT/ρa_new_k
                else:
                    UpdVar.H.new[i][k] = GMV.H.values[k]
                    UpdVar.q_tot.new[i][k] = GMV.q_tot.values[k]

        if self.use_local_micro:
            for i in i_uds:
                for k in grid.over_elems_real(Center()):
                    T, q_liq = eos(tmp['p_0_half'][k],
                                UpdVar.q_tot.new[i][k],
                                UpdVar.H.new[i][k])
                    UpdVar.T.new[i][k], UpdVar.q_liq.new[i][k] = T, q_liq
                    UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half, UpdVar.T.new,
                                                                       UpdVar.q_tot.new, UpdVar.q_liq.new,
                                                                       UpdVar.q_rai.new, UpdVar.H.new,
                                                                       i, k)
                UpdVar.q_rai.new[i][k_1] = 0.0

        return

    def update_GMV_MF(self, grid, q, GMV, EnvVar, UpdVar, TS, tmp):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        k_1 = grid.first_interior(Zmin())
        kb_1 = grid.boundary(Zmin())
        self.massflux_h[:] = 0.0
        self.massflux_qt[:] = 0.0

        for i in i_uds:
            self.m[i][kb_1] = 0.0
            for k in grid.over_elems_real(Center()):
                self.m[i][k] = ((UpdVar.W.values[i][k] - q['w', i_env].values[k] )* self.Ref.rho0[k]
                               * UpdVar.Area.values[i].Mid(k))

        for k in grid.over_elems_real(Center()):
            self.massflux_h[k] = 0.0
            self.massflux_qt[k] = 0.0
            for i in i_uds:
                self.massflux_h[k] += self.m[i][k] * (UpdVar.H.values[i].Mid(k) - EnvVar.H.values.Mid(k))
                self.massflux_qt[k] += self.m[i][k] * (UpdVar.q_tot.values[i].Mid(k) - EnvVar.q_tot.values.Mid(k))

        for k in grid.over_elems_real(Center()):
            self.massflux_tendency_h[k] = -tmp['α_0_half'][k]*grad(self.massflux_h.Dual(k), grid)
            self.massflux_tendency_qt[k] = -tmp['α_0_half'][k]*grad(self.massflux_qt.Dual(k), grid)
        return

    def update_GMV_ED(self, grid, q, GMV, UpdMicro, Case, TS, tmp, tri_diag):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        k_1 = grid.first_interior(Zmin())
        dzi = grid.dzi
        α_1 = tmp['α_0_half'][k_1]
        ρ_0_half = tmp['ρ_0_half']
        ae = q['a', i_env]

        ae_1 = ae[k_1]

        for k in grid.over_elems_real(Node()):
            tri_diag.ρaK[k] = ae.Mid(k)*self.KH.values.Mid(k)*ρ_0_half.Mid(k)

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion_new_new(grid, TS.dt, tri_diag, ρ_0_half, ae)

        # Solve q_tot
        for k in grid.over_elems(Center()):
            tri_diag.f[k] = GMV.q_tot.values[k] + TS.dt * self.massflux_tendency_qt[k] + UpdMicro.prec_source_qt_tot[k]
        tri_diag.f[k_1] = tri_diag.f[k_1] + TS.dt * Case.Sur.rho_qtflux * dzi * α_1/ae_1

        tridiag_solve_wrapper_new(grid, GMV.q_tot.new, tri_diag)

        # Solve H
        for k in grid.over_elems(Center()):
            tri_diag.f[k] = GMV.H.values[k] + TS.dt * self.massflux_tendency_h[k] + UpdMicro.prec_source_h_tot[k]
        tri_diag.f[k_1] = tri_diag.f[k_1] + TS.dt * Case.Sur.rho_hflux * dzi * α_1/ae_1

        tridiag_solve_wrapper_new(grid, GMV.H.new, tri_diag)

        # Solve U
        for k in grid.over_elems_real(Node()):
            tri_diag.ρaK[k] = ae.Mid(k)*self.KM.values.Mid(k)*ρ_0_half.Mid(k)

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion_new_new(grid, TS.dt, tri_diag, ρ_0_half, ae)

        for k in grid.over_elems(Center()):
            tri_diag.f[k] = GMV.U.values[k]
        tri_diag.f[k_1] = tri_diag.f[k_1] + TS.dt * Case.Sur.rho_uflux * dzi * α_1/ae_1

        tridiag_solve_wrapper_new(grid, GMV.U.new, tri_diag)

        # Solve V
        for k in grid.over_elems(Center()):
            tri_diag.f[k] = GMV.V.values[k]
        tri_diag.f[k_1] = tri_diag.f[k_1] + TS.dt * Case.Sur.rho_vflux * dzi * α_1/ae_1

        tridiag_solve_wrapper_new(grid, GMV.V.new, tri_diag)

        return

    def compute_tke_buoy(self, grid, q, GMV, EnvVar, EnvThermo, tmp):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        grad_thl_minus=0.0
        grad_qt_minus=0.0
        grad_thl_plus=0
        grad_qt_plus=0
        grad_θ_liq = Full(grid)
        grad_q_tot = Full(grid)
        ae = q['a', i_env]

        # Note that source terms at the first interior point are not really used because that is where tke boundary condition is
        # enforced (according to MO similarity). Thus here I am being sloppy about lowest grid point
        for k in grid.over_elems_real(Center()):
            qt_dry = EnvThermo.qt_dry[k]
            th_dry = EnvThermo.th_dry[k]
            t_cloudy = EnvThermo.t_cloudy[k]
            qv_cloudy = EnvThermo.qv_cloudy[k]
            qt_cloudy = EnvThermo.qt_cloudy[k]
            th_cloudy = EnvThermo.th_cloudy[k]

            lh = latent_heat(t_cloudy)
            cpm = cpm_c(qt_cloudy)
            grad_thl_minus = grad_thl_plus
            grad_qt_minus = grad_qt_plus
            grad_thl_plus = grad(EnvVar.θ_liq.values.Dual(k), grid)
            grad_qt_plus  = grad(EnvVar.q_tot.values.Dual(k), grid)

            prefactor = Rd * exner_c(tmp['p_0_half'][k])/tmp['p_0_half'][k]

            d_alpha_thetal_dry = prefactor * (1.0 + (eps_vi-1.0) * qt_dry)
            d_alpha_qt_dry = prefactor * th_dry * (eps_vi-1.0)

            if EnvVar.CF.values[k] > 0.0:
                d_alpha_thetal_cloudy = (prefactor * (1.0 + eps_vi * (1.0 + lh / Rv / t_cloudy) * qv_cloudy - qt_cloudy )
                                         / (1.0 + lh * lh / cpm / Rv / t_cloudy / t_cloudy * qv_cloudy))
                d_alpha_qt_cloudy = (lh / cpm / t_cloudy * d_alpha_thetal_cloudy - prefactor) * th_cloudy
            else:
                d_alpha_thetal_cloudy = 0.0
                d_alpha_qt_cloudy = 0.0

            d_alpha_thetal_total = (EnvVar.CF.values[k] * d_alpha_thetal_cloudy
                                    + (1.0-EnvVar.CF.values[k]) * d_alpha_thetal_dry)
            d_alpha_qt_total = (EnvVar.CF.values[k] * d_alpha_qt_cloudy
                                + (1.0-EnvVar.CF.values[k]) * d_alpha_qt_dry)

            # TODO - check
            EnvVar.tke.buoy[k] = g / tmp['α_0_half'][k] * ae[k] * tmp['ρ_0_half'][k] \
                               * ( \
                                   - self.KH.values[k] * interp2pt(grad_thl_plus, grad_thl_minus) * d_alpha_thetal_total \
                                   - self.KH.values[k] * interp2pt(grad_qt_plus,  grad_qt_minus)  * d_alpha_qt_total\
                                 )
        return

    def compute_tke_pressure(self, grid, tmp, q, EnvVar, UpdVar):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        for k in grid.over_elems_real(Center()):
            EnvVar.tke.press[k] = 0.0
            for i in i_uds:
                wu_half = UpdVar.W.values[i].Mid(k)
                we_half = q['w', i_env].Mid(k)
                a_i = UpdVar.Area.values[i][k]
                press_buoy= (-1.0 * tmp['ρ_0_half'][k] * a_i * UpdVar.B.values[i][k] * self.pressure_buoy_coeff)
                press_drag = (-1.0 * tmp['ρ_0_half'][k] * np.sqrt(a_i) * (self.pressure_drag_coeff/self.pressure_plume_spacing* (wu_half - we_half)*np.fabs(wu_half - we_half)))
                EnvVar.tke.press[k] += (we_half - wu_half) * (press_buoy + press_drag)
        return


    def update_GMV_diagnostics(self, grid, q, tmp, GMV, EnvVar, UpdVar):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        ae = q['a', i_env]
        for k in grid.over_elems_real(Center()):
            GMV.q_liq.values[k] = ae[k] * EnvVar.q_liq.values[k] + sum([ UpdVar.Area.values[i][k] * UpdVar.q_liq.values[i][k] for i in i_uds])
            GMV.q_rai.values[k] = ae[k] * EnvVar.q_rai.values[k] + sum([ UpdVar.Area.values[i][k] * UpdVar.q_rai.values[i][k] for i in i_uds])
            GMV.T.values[k]     = ae[k] * EnvVar.T.values[k]     + sum([ UpdVar.Area.values[i][k] * UpdVar.T.values[i][k] for i in i_uds])
            GMV.B.values[k]     = ae[k] * EnvVar.B.values[k]     + sum([ UpdVar.Area.values[i][k] * UpdVar.B.values[i][k] for i in i_uds])
            GMV.θ_liq.values[k] = t_to_thetali_c(tmp['p_0_half'][k], GMV.T.values[k], GMV.q_tot.values[k], GMV.q_liq.values[k], 0.0)
        return


    def compute_covariance(self, grid, Ref, q, GMV, EnvVar, UpdVar, EnvThermo, Case, TS, tmp, tri_diag):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        if self.similarity_diffusivity: # otherwise, we computed mixing length when we computed
            self.compute_mixing_length(grid, Case.Sur.obukhov_length, EnvVar)
        we = q['w', i_env]
        self.compute_tke_buoy(grid, q, GMV, EnvVar, EnvThermo, tmp)
        self.compute_covariance_entr(grid, q, tmp, UpdVar, EnvVar.tke, UpdVar.W, UpdVar.W, we, we, 'tke')
        self.compute_covariance_shear(grid, q, tmp, GMV, EnvVar.tke, UpdVar.W.values, UpdVar.W.values, we, we, 'tke')
        self.compute_covariance_interdomain_src(grid, q, tmp, UpdVar.Area,UpdVar.W,UpdVar.W, we, we, EnvVar.tke, 'tke')
        self.compute_tke_pressure(grid, tmp, q, EnvVar, UpdVar)
        self.compute_covariance_entr(grid, q, tmp, UpdVar, EnvVar.cv_θ_liq,   UpdVar.H,  UpdVar.H,  EnvVar.H,  EnvVar.H, '')
        self.compute_covariance_entr(grid, q, tmp, UpdVar, EnvVar.cv_q_tot,  UpdVar.q_tot, UpdVar.q_tot, EnvVar.q_tot, EnvVar.q_tot, '')
        self.compute_covariance_entr(grid, q, tmp, UpdVar, EnvVar.cv_θ_liq_q_tot, UpdVar.H,  UpdVar.q_tot, EnvVar.H,  EnvVar.q_tot, '')
        self.compute_covariance_shear(grid, q, tmp, GMV, EnvVar.cv_θ_liq,   UpdVar.H.values,  UpdVar.H.values,  EnvVar.H.values,  EnvVar.H.values, '')
        self.compute_covariance_shear(grid, q, tmp, GMV, EnvVar.cv_q_tot,  UpdVar.q_tot.values, UpdVar.q_tot.values, EnvVar.q_tot.values, EnvVar.q_tot.values, '')
        self.compute_covariance_shear(grid, q, tmp, GMV, EnvVar.cv_θ_liq_q_tot, UpdVar.H.values,  UpdVar.q_tot.values, EnvVar.H.values,  EnvVar.q_tot.values, '')
        self.compute_covariance_interdomain_src(grid, q, tmp, UpdVar.Area, UpdVar.H,  UpdVar.H,  EnvVar.H,  EnvVar.H,  EnvVar.cv_θ_liq, '')
        self.compute_covariance_interdomain_src(grid, q, tmp, UpdVar.Area, UpdVar.q_tot, UpdVar.q_tot, EnvVar.q_tot, EnvVar.q_tot, EnvVar.cv_q_tot, '')
        self.compute_covariance_interdomain_src(grid, q, tmp, UpdVar.Area, UpdVar.H,  UpdVar.q_tot, EnvVar.H,  EnvVar.q_tot, EnvVar.cv_θ_liq_q_tot, '')
        self.compute_covariance_rain(grid, q, tmp, TS, GMV, EnvVar, EnvThermo)

        self.reset_surface_covariance(grid, Ref, q, tmp, GMV, Case)
        self.update_covariance_ED(grid, q, tmp, GMV, EnvVar, UpdVar, Case,TS, GMV.W,     GMV.W,     GMV.tke,            EnvVar.tke,           we           , we           , UpdVar.W    , UpdVar.W, 'tke', tri_diag)
        self.update_covariance_ED(grid, q, tmp, GMV, EnvVar, UpdVar, Case,TS, GMV.H,     GMV.H,     GMV.cv_θ_liq,       EnvVar.cv_θ_liq,      EnvVar.H.values    , EnvVar.H.values    , UpdVar.H    , UpdVar.H, 'cv_θ_liq', tri_diag)
        self.update_covariance_ED(grid, q, tmp, GMV, EnvVar, UpdVar, Case,TS, GMV.q_tot, GMV.q_tot, GMV.cv_q_tot,       EnvVar.cv_q_tot,      EnvVar.q_tot.values, EnvVar.q_tot.values, UpdVar.q_tot, UpdVar.q_tot, 'cv_q_tot', tri_diag)
        self.update_covariance_ED(grid, q, tmp, GMV, EnvVar, UpdVar, Case,TS, GMV.H,     GMV.q_tot, GMV.cv_θ_liq_q_tot, EnvVar.cv_θ_liq_q_tot,EnvVar.H.values    , EnvVar.q_tot.values, UpdVar.H    , UpdVar.q_tot, 'cv_θ_liq_q_tot', tri_diag)
        self.cleanup_covariance(grid, GMV, EnvVar, UpdVar)
        return


    def initialize_covariance(self, grid, q, tmp, GMV, EnvVar, Ref, Case):
        self.zi = compute_inversion(grid, GMV, Case.inversion_option, tmp, self.Ri_bulk_crit)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)
        ws = self.wstar
        us = Case.Sur.ustar
        zs = self.zi
        k_1 = grid.first_interior(Zmin())
        Hvar_1 = GMV.cv_θ_liq.values[k_1]
        QTvar_1 = GMV.cv_q_tot.values[k_1]
        HQTcov_1 = GMV.cv_θ_liq_q_tot.values[k_1]
        self.reset_surface_covariance(grid, Ref, q, tmp, GMV, Case)
        if ws > 0.0:
            for k in grid.over_elems(Center()):
                z = grid.z_half[k]
                temp = ws * 1.3 * np.cbrt((us*us*us)/(ws*ws*ws) + 0.6 * z/zs) * np.sqrt(np.fmax(1.0-z/zs,0.0))
                GMV.tke.values[k] = temp
                GMV.cv_θ_liq.values[k]   = Hvar_1 * temp
                GMV.cv_q_tot.values[k]  = QTvar_1 * temp
                GMV.cv_θ_liq_q_tot.values[k] = HQTcov_1 * temp
            self.reset_surface_covariance(grid, Ref, q, tmp, GMV, Case)
            self.compute_mixing_length(grid, Case.Sur.obukhov_length, EnvVar)
        return

    def cleanup_covariance(self, grid, GMV, EnvVar, UpdVar):
        tmp_eps = 1e-18
        for k in grid.over_elems_real(Center()):
            if GMV.tke.values[k] < tmp_eps:                     GMV.tke.values[k] = 0.0
            if GMV.cv_θ_liq.values[k] < tmp_eps:                    GMV.cv_θ_liq.values[k] = 0.0
            if GMV.cv_q_tot.values[k] < tmp_eps:                   GMV.cv_q_tot.values[k] = 0.0
            if np.fabs(GMV.cv_θ_liq_q_tot.values[k]) < tmp_eps:         GMV.cv_θ_liq_q_tot.values[k] = 0.0
            if EnvVar.cv_θ_liq.values[k] < tmp_eps:            EnvVar.cv_θ_liq.values[k] = 0.0
            if EnvVar.tke.values[k] < tmp_eps:             EnvVar.tke.values[k] = 0.0
            if EnvVar.cv_q_tot.values[k] < tmp_eps:           EnvVar.cv_q_tot.values[k] = 0.0
            if np.fabs(EnvVar.cv_θ_liq_q_tot.values[k]) < tmp_eps: EnvVar.cv_θ_liq_q_tot.values[k] = 0.0


    def compute_covariance_shear(self, grid, q, tmp, GMV, Covar, UpdVar1, UpdVar2, EnvVar1, EnvVar2, name):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        diff_var1 = 0.0
        diff_var2 = 0.0
        du = 0.0
        dv = 0.0
        tke_factor = 1.0
        du_high = 0.0
        dv_high = 0.0
        ae = q['a', i_env]

        for k in grid.over_elems_real(Center()):
            if name == 'tke':
                du_low = du_high
                dv_low = dv_high
                du_high = grad(GMV.U.values.Dual(k), grid)
                dv_high = grad(GMV.V.values.Dual(k), grid)
                diff_var2 = grad(EnvVar2.Dual(k), grid)
                diff_var1 = grad(EnvVar1.Dual(k), grid)
                tke_factor = 0.5
            else:
                du_low = 0.0
                dv_low = 0.0
                du_high = 0.0
                dv_high = 0.0
                diff_var2 = grad(EnvVar2.Cut(k), grid)
                diff_var1 = grad(EnvVar1.Cut(k), grid)
            Covar.shear[k] = tke_factor*2.0*(tmp['ρ_0_half'][k] * ae[k] * self.KH.values[k] *
                        (diff_var1*diff_var2 +
                            pow(interp2pt(du_low, du_high),2.0) +
                            pow(interp2pt(dv_low, dv_high),2.0)))
        return

    def compute_covariance_interdomain_src(self, grid, q, tmp, au, phi_u, psi_u, phi_e, psi_e, Covar, name):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        for k in grid.over_elems(Center()):
            Covar.interdomain[k] = 0.0
            for i in i_uds:
                if name == 'tke':
                    tke_factor = 0.5
                    phi_diff = phi_u.values[i].Mid(k) - phi_e.Mid(k)
                    psi_diff = psi_u.values[i].Mid(k) - psi_e.Mid(k)
                else:
                    tke_factor = 1.0
                    phi_diff = phi_u.values[i][k]-phi_e.values[k]
                    psi_diff = psi_u.values[i][k]-psi_e.values[k]

                Covar.interdomain[k] += tke_factor*au.values[i][k] * (1.0-au.values[i][k]) * phi_diff * psi_diff
        return

    def compute_covariance_entr(self, grid, q, tmp, UpdVar, Covar, UpdVar1, UpdVar2, EnvVar1, EnvVar2, name):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        for k in grid.over_elems_real(Center()):
            Covar.entr_gain[k] = 0.0
            for i in i_uds:
                if name =='tke':
                    updvar1 = UpdVar1.values[i].Mid(k)
                    updvar2 = UpdVar2.values[i].Mid(k)
                    envvar1 = EnvVar1.Mid(k)
                    envvar2 = EnvVar2.Mid(k)
                    tke_factor = 0.5
                else:
                    updvar1 = UpdVar1.values[i][k]
                    updvar2 = UpdVar2.values[i][k]
                    envvar1 = EnvVar1.values[k]
                    envvar2 = EnvVar2.values[k]
                    tke_factor = 1.0
                w_u = UpdVar.W.values[i].Mid(k)
                Covar.entr_gain[k] += tke_factor*UpdVar.Area.values[i][k] * np.fabs(w_u) * self.detr_sc[i][k] * \
                                             (updvar1 - envvar1) * (updvar2 - envvar2)
            Covar.entr_gain[k] *= tmp['ρ_0_half'][k]
        return

    def compute_covariance_detr(self, grid, q, tmp, Covar, UpdVar):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        ae = q['a', i_env]

        for k in grid.over_elems_real(Center()):
            Covar.detr_loss[k] = 0.0
            for i in i_uds:
                w_u = UpdVar.W.values[i].Mid(k)
                Covar.detr_loss[k] += UpdVar.Area.values[i][k] * np.fabs(w_u) * self.entr_sc[i][k]
            Covar.detr_loss[k] *= tmp['ρ_0_half'][k] * Covar.values[k]
        return

    def compute_covariance_rain(self, grid, q, tmp, TS, GMV, EnvVar, EnvThermo):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        ae = q['a', i_env]

        for k in grid.over_elems_real(Center()):
            EnvVar.tke.rain_src[k] = 0.0
            EnvVar.cv_θ_liq.rain_src[k]   = tmp['ρ_0_half'][k] * ae[k] * 2. * EnvThermo.Hvar_rain_dt[k]   * TS.dti
            EnvVar.cv_q_tot.rain_src[k]  = tmp['ρ_0_half'][k] * ae[k] * 2. * EnvThermo.QTvar_rain_dt[k]  * TS.dti
            EnvVar.cv_θ_liq_q_tot.rain_src[k] = tmp['ρ_0_half'][k] * ae[k] *      EnvThermo.HQTcov_rain_dt[k] * TS.dti
        return


    def compute_covariance_dissipation(self, grid, q, tmp, Covar, EnvVar):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        ae = q['a', i_env]

        for k in grid.over_elems_real(Center()):
            l_mix = np.fmax(self.mixing_length[k], 1.0)
            tke_env = np.fmax(EnvVar.tke.values[k], 0.0)

            Covar.dissipation[k] = (tmp['ρ_0_half'][k] * ae[k] * Covar.values[k] * pow(tke_env, 0.5)/l_mix * self.tke_diss_coeff)
        return

    def update_covariance_ED(self, grid, q, tmp, GMV, EnvVar, UpdVar, Case,TS, GmvVar1, GmvVar2, GmvCovar, Covar,  EnvVar1,  EnvVar2, UpdVar1,  UpdVar2, name, tri_diag):
        i_gm, i_env, i_uds, i_sd = q.domain_idx()
        dzi = grid.dzi
        dzi2 = grid.dzi**2.0
        dti = TS.dti
        k_1 = grid.first_interior(Zmin())
        k_2 = grid.first_interior(Zmax())

        alpha0LL = self.Ref.alpha0_half[k_1]
        zLL = grid.z_half[k_1]

        x = Half(grid)
        ae_old = Half(grid)

        for k in grid.over_elems(Center()):
            ae_old[k] = 1.0 - np.sum([UpdVar.Area.old[i][k] for i in i_uds])

        S = Case.Sur

        if name=='tke':
            GmvCovar.values[k_1] = get_surface_tke(S.ustar, self.wstar, zLL, S.obukhov_length)
        elif name=='cv_θ_liq':
            GmvCovar.values[k_1] = get_surface_variance(S.rho_hflux * alpha0LL, S.rho_hflux * alpha0LL, S.ustar, zLL, S.obukhov_length)
        elif name=='cv_q_tot':
            GmvCovar.values[k_1] = get_surface_variance(S.rho_qtflux * alpha0LL, S.rho_qtflux * alpha0LL, S.ustar, zLL, S.obukhov_length)
        elif name=='cv_θ_liq_q_tot':
            GmvCovar.values[k_1] = get_surface_variance(S.rho_hflux * alpha0LL, S.rho_qtflux * alpha0LL, S.ustar, zLL, S.obukhov_length)

        self.get_env_covar_from_GMV(grid, q, UpdVar.Area, UpdVar1, UpdVar2, EnvVar1, EnvVar2, Covar, GmvVar1, GmvVar2, GmvCovar, name)

        Covar_surf = Covar.values[k_1]
        a_env = q['a', i_env]
        w_env = q['w', i_env]
        ρ_0_half = tmp['ρ_0_half']

        for k in grid.over_elems_real(Center()):
            ρ_0_cut = ρ_0_half.Cut(k)
            ae_cut = a_env.Cut(k)
            w_cut = w_env.DualCut(k)
            ρa_K_cut = a_env.DualCut(k) * self.KH.values.DualCut(k) * ρ_0_half.DualCut(k)

            D_env = sum([ρ_0_cut[1] *
                         UpdVar.Area.values[i][k] *
                         UpdVar.W.values[i].Mid(k) *
                         self.entr_sc[i][k] for i in i_uds])

            l_mix = np.fmax(self.mixing_length[k], 1.0)
            tke_env = np.fmax(EnvVar.tke.values[k], 0.0)

            tri_diag.a[k] = (- ρa_K_cut[0] * dzi2 )
            tri_diag.b[k] = (ρ_0_cut[1] * ae_cut[1] * dti
                     - ρ_0_cut[1] * ae_cut[1] * w_cut[1] * dzi
                     + ρa_K_cut[1] * dzi2 + ρa_K_cut[0] * dzi2
                     + D_env
                     + ρ_0_cut[1] * ae_cut[1] * self.tke_diss_coeff * np.sqrt(tke_env)/l_mix)
            tri_diag.c[k] = (ρ_0_cut[2] * ae_cut[2] * w_cut[2] * dzi - ρa_K_cut[1] * dzi2)

            tri_diag.f[k] = (ρ_0_cut[1] * ae_old[k] * Covar.values[k] * dti
                     + Covar.press[k]
                     + Covar.buoy[k]
                     + Covar.shear[k]
                     + Covar.entr_gain[k]
                     + Covar.rain_src[k])

            tri_diag.a[k_1] = 0.0
            tri_diag.b[k_1] = 1.0
            tri_diag.c[k_1] = 0.0
            tri_diag.f[k_1] = Covar_surf

            tri_diag.b[k_2] += tri_diag.c[k_2]
            tri_diag.c[k_2] = 0.0

        tridiag_solve_wrapper_new(grid, x, tri_diag)

        for k in grid.over_elems_real(Center()):
            if name == 'cv_θ_liq_q_tot':
                Covar.values[k] = np.fmax(x[k], - np.sqrt(EnvVar.cv_θ_liq.values[k]*EnvVar.cv_q_tot.values[k]))
                Covar.values[k] = np.fmin(x[k],   np.sqrt(EnvVar.cv_θ_liq.values[k]*EnvVar.cv_q_tot.values[k]))
            else:
                Covar.values[k] = np.fmax(x[k], 0.0)

        return
