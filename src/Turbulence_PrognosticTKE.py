import numpy as np
from parameters import *
import sys
from EDMF_Updrafts import *
from EDMF_Environment import *
from Grid import Grid, Zmin, Zmax, Center, Node
from Field import Field
from Field import Cut
from Field import Dual
from TriDiagSolver import tridiag_solve, construct_tridiag_diffusion, construct_tridiag_diffusion_new
from Variables import VariablePrognostic, VariableDiagnostic, GridMeanVariables
from Surface import SurfaceBase
from Cases import  CasesBase
from ReferenceState import  ReferenceState
from TimeStepping import TimeStepping
from NetCDFIO import NetCDFIO_Stats
from thermodynamic_functions import  *
from turbulence_functions import *
from utility_functions import *

def ParameterizationFactory(namelist, paramlist, Gr, Ref):
    scheme = namelist['turbulence']['scheme']
    if scheme == 'EDMF_PrognosticTKE':
        return EDMF_PrognosticTKE(namelist, paramlist, Gr, Ref)
    elif scheme == 'SimilarityED':
        return SimilarityED(namelist, paramlist, Gr, Ref)
    else:
        print('Did not recognize parameterization ' + scheme)
        return

def advect(f, grid):
    return (f[2]-f[1])*grid.dzi

# A base class common to all turbulence parameterizations
class ParameterizationBase:
    def __init__(self, paramlist, Gr, Ref):
        self.turbulence_tendency = Field.half(Gr)
        self.grid = Gr # grid class
        self.Ref = Ref # reference state class
        self.KM = VariableDiagnostic(Gr, Center(), 'sym', 'diffusivity', 'm^2/s') # eddy viscosity
        self.KH = VariableDiagnostic(Gr, Center(), 'sym', 'viscosity', 'm^2/s') # eddy diffusivity
        # get values from paramlist
        self.prandtl_number = paramlist['turbulence']['prandtl_number']
        self.Ri_bulk_crit = paramlist['turbulence']['Ri_bulk_crit']

        return
    def initialize(self, GMV):
        return

    def initialize_io(self, Stats):
        return

    def io(self, Stats):
        return

    # Calculate the tendency of the grid mean variables due to turbulence as the difference between the values at the beginning
    # and  end of all substeps taken
    def update(self,GMV, Case, TS):
        for k in self.grid.over_elems_real(Center()):
            GMV.H.tendencies[k] += (GMV.H.new[k] - GMV.H.values[k]) * TS.dti
            GMV.QT.tendencies[k] += (GMV.QT.new[k] - GMV.QT.values[k]) * TS.dti
            GMV.U.tendencies[k] += (GMV.U.new[k] - GMV.U.values[k]) * TS.dti
            GMV.V.tendencies[k] += (GMV.V.new[k] - GMV.V.values[k]) * TS.dti
        return

    # Update the diagnosis of the inversion height, using the maximum temperature gradient method
    def update_inversion(self, GMV, option, tmp):
        theta_rho = Field.half(self.grid)
        maxgrad = 0.0
        theta_rho_bl = theta_rho.surface_bl(self.grid)

        Ri_bulk_crit = 0.0

        for k in self.grid.over_elems_real(Center()):
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            theta_rho[k] = theta_rho_c(tmp['p_0', k], GMV.T.values[k], GMV.QT.values[k], qv)

        if option == 'theta_rho':
            for k in self.grid.over_elems_real(Center()):
                if theta_rho[k] > theta_rho_bl:
                    self.zi = self.grid.z_half[k]
                    break
        elif option == 'thetal_maxgrad':

            for k in self.grid.over_elems_real(Center()):
                grad =  (GMV.THL.values[k+1] - GMV.THL.values[k])*self.grid.dzi
                if grad > maxgrad:
                    maxgrad = grad
                    self.zi = self.grid.z[k]
        elif option == 'critical_Ri':
            self.zi = get_inversion(theta_rho, GMV.U.values, GMV.V.values, self.grid, self.Ri_bulk_crit)

        else:
            print('INVERSION HEIGHT OPTION NOT RECOGNIZED')

        # print('Inversion height ', self.zi)

        return



    # Compute eddy diffusivities from similarity theory (Siebesma 2007)
    def compute_eddy_diffusivities_similarity(self, GMV, Case, tmp):
        self.update_inversion(GMV, Case.inversion_option, tmp)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)

        ustar = Case.Sur.ustar
        for k in self.grid.over_elems_real(Center()):
            zzi = self.grid.z_half[k]/self.zi
            if zzi <= 1.0:
                if self.wstar<1e-6:
                    self.KH.values[k] = 0.0
                    self.KM.values[k] = 0.0
                else:
                    self.KH.values[k] = vkb * ( (ustar/self.wstar)**3 + 39.0*vkb*zzi)**(1.0/3.0) * zzi * (1.0-zzi) * (1.0-zzi) * self.wstar * self.zi
                    self.KM.values[k] = self.KH.values[k] * self.prandtl_number
            else:
                self.KH.values[k] = 0.0
                self.KM.values[k] = 0.0

        # Set the boundary points at top and bottom of domain
        self.KH.set_bcs(self.grid)
        self.KM.set_bcs(self.grid)
        return

    def update_GMV_diagnostics(self, GMV):
        return

class SimilarityED(ParameterizationBase):
    def __init__(self, namelist, paramlist, Gr, Ref):
        self.extrapolate_buoyancy = False
        ParameterizationBase.__init__(self, paramlist, Gr, Ref)
        return
    def initialize(self, GMV):
        return

    def initialize_io(self, Stats):
        Stats.add_profile('eddy_viscosity')
        Stats.add_profile('eddy_diffusivity')
        return

    def io(self, Stats):
        Stats.write_profile_new('eddy_viscosity'  , self.grid, self.KM.values)
        Stats.write_profile_new('eddy_diffusivity', self.grid, self.KH.values)
        return

    def update(self,GMV, Case, TS, tmp, q):

        GMV.H.set_bcs(self.grid)
        GMV.QT.set_bcs(self.grid)

        ParameterizationBase.compute_eddy_diffusivities_similarity(self, GMV, Case)

        gw = self.grid.gw
        k_i = self.grid.first_interior(Zmin())
        nzg = self.grid.nzg
        nz = self.grid.nz
        a = Field.half(self.grid)
        b = Field.half(self.grid)
        c = Field.half(self.grid)
        x = Field.half(self.grid)
        dummy_ae = Field.half(self.grid)
        rho_K_m = Field.full(self.grid)

        slice_real = self.grid.slice_real(Center())

        for k in self.grid.over_elems(Center()):
            dummy_ae[k] = 1.0 - self.UpdVar.Area.bulkvalues[k]

        for k in self.grid.over_elems_real(Node()):
            rho_K_m[k] = 0.5 * (self.KH.values[k]+ self.KH.values[k+1]) * self.Ref.rho0[k]

        # Matrix is the same for all variables that use the same eddy diffusivity
        construct_tridiag_diffusion(nzg, gw, self.grid.dzi, TS.dt, rho_K_m, self.Ref.rho0_half, dummy_ae, a[slice_real], b[slice_real], c[slice_real])

        # Solve QT
        for k in self.grid.over_elems(Center()):
            x[k] = GMV.QT.values[k]
        x[k_i] = x[k_i] + TS.dt * Case.Sur.rho_qtflux * self.grid.dzi * tmp['α_0', k_i]

        tridiag_solve(self.grid.nz, x, a[slice_real], b[slice_real], c[slice_real])
        for k in self.grid.over_elems(Center()):
            GMV.QT.new[k] = x[k]

        # Solve H
        for k in self.grid.over_elems(Center()):
            x[k] = GMV.H.values[k]
        x[k_i] = x[k_i] + TS.dt * Case.Sur.rho_hflux * self.grid.dzi * tmp['α_0', k_i]

        tridiag_solve(self.grid.nz, x, a[slice_real], b[slice_real], c[slice_real])
        for k in self.grid.over_elems(Center()):
            GMV.H.new[k] = x[k]

        # Solve U
        for k in self.grid.over_elems(Center()):
            x[k] = GMV.U.values[k]
        x[k_i] = x[k_i] + TS.dt * Case.Sur.rho_uflux * self.grid.dzi * tmp['α_0', k_i]

        tridiag_solve(self.grid.nz, x, a[slice_real], b[slice_real], c[slice_real])
        for k in self.grid.over_elems(Center()):
            GMV.U.new[k] = x[k]

        # Solve V
        for k in self.grid.over_elems(Center()):
            x[k] = GMV.V.values[k]
        x[k_i] = x[k_i] + TS.dt * Case.Sur.rho_vflux * self.grid.dzi * tmp['α_0', k_i]

        tridiag_solve(self.grid.nz, x, a[slice_real], b[slice_real], c[slice_real])
        for k in self.grid.over_elems(Center()):
            GMV.V.new[k] = x[k]

        self.update_GMV_diagnostics(GMV)
        ParameterizationBase.update(self, GMV,Case, TS)

        return

    def update_inversion(self, GMV, option, tmp):
        ParameterizationBase.update_inversion(self, GMV, option, tmp)
        return

    def update_GMV_diagnostics(self, GMV):
        # Ideally would write this to be able to use an SGS condensation closure, but unless the need arises,
        # we will just do an all-or-nothing treatment as a placeholder
        GMV.satadjust()
        return


class EDMF_PrognosticTKE(ParameterizationBase):
    # Initialize the class
    def __init__(self, namelist, paramlist, Gr, Ref):
        # Initialize the base parameterization class
        ParameterizationBase.__init__(self, paramlist,  Gr, Ref)

        # Set the number of updrafts (1)
        try:
            self.n_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']
        except:
            self.n_updrafts = 1
            print('Turbulence--EDMF_PrognosticTKE: defaulting to single updraft')
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
            self.calc_tke = namelist['turbulence']['EDMF_PrognosticTKE']['calculate_tke']
        except:
            self.calc_tke = True

        try:
            self.calc_scalar_var = namelist['turbulence']['EDMF_PrognosticTKE']['calc_scalar_var']
        except:
            self.calc_scalar_var = False
        if (self.calc_scalar_var==True and self.calc_tke==False):
            sys.exit('Turbulence--EDMF_PrognosticTKE: >>calculate_tke<< must be set to True when >>calc_scalar_var<< is True (to calculate the mixing length for the variance and covariance calculations')

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
        if(self.calc_tke == False and 'tke' in str(namelist['turbulence']['EDMF_PrognosticTKE']['entrainment'])):
             sys.exit('Turbulence--EDMF_PrognosticTKE: >>calc_tke<< must be set to True when entrainment is using tke')

        try:
            self.similarity_diffusivity = namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity']
        except:
            self.similarity_diffusivity = False
            print('Turbulence--EDMF_PrognosticTKE: defaulting to TKE-based eddy diffusivity')
        if(self.similarity_diffusivity == False and self.calc_tke ==False):
            sys.exit('Turbulence--EDMF_PrognosticTKE: either >>use_similarity_diffusivity<< or >>calc_tke<< flag is needed to get the eddy diffusivities')

        if(self.similarity_diffusivity == True and self.calc_tke == True):
           print("TKE will be calculated but not used for eddy diffusivity calculation")

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
        # "Legacy" coefficients used by the steady updraft routine
        self.vel_pressure_coeff = self.pressure_drag_coeff/self.pressure_plume_spacing
        self.vel_buoy_coeff = 1.0-self.pressure_buoy_coeff
        if self.calc_tke == True:
            self.tke_ed_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff']
            self.tke_diss_coeff = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff']

        # Need to code up as paramlist option?
        self.minimum_area = 1e-3

        # Create the updraft variable class (major diagnostic and prognostic variables)
        self.UpdVar = UpdraftVariables(self.n_updrafts, namelist,paramlist, Gr)
        # Create the class for updraft thermodynamics
        self.UpdThermo = UpdraftThermodynamics(self.n_updrafts, Gr, Ref, self.UpdVar)
        # Create the class for updraft microphysics
        self.UpdMicro = UpdraftMicrophysics(paramlist, self.n_updrafts, Gr, Ref)

        # Create the environment variable class (major diagnostic and prognostic variables)
        self.EnvVar = EnvironmentVariables(namelist,Gr)
        # Create the class for environment thermodynamics
        self.EnvThermo = EnvironmentThermodynamics(namelist, paramlist, Gr, Ref, self.EnvVar)

        a_ = self.surface_area/self.n_updrafts
        self.surface_scalar_coeff = np.zeros((self.n_updrafts,), dtype=np.double, order='c')
        for i in range(self.n_updrafts):
            self.surface_scalar_coeff[i] = percentile_bounds_mean_norm(1.0-self.surface_area+i*a_,
                                                                       1.0-self.surface_area + (i+1)*a_ , 1000)

        # Entrainment rates
        self.entr_sc = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')
        #self.press = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double,order='c')

        # Detrainment rates
        self.detr_sc = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')

        # Pressure term in updraft vertical momentum equation
        self.updraft_pressure_sink = np.zeros((self.n_updrafts, Gr.nzg,),dtype=np.double,order='c')
        # Mass flux
        self.m = np.zeros((self.n_updrafts, Gr.nzg),dtype=np.double, order='c')

        # mixing length
        self.mixing_length = Field.half(Gr)

        # Near-surface BC of updraft area fraction
        self.area_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.w_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.h_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')
        self.qt_surface_bc= np.zeros((self.n_updrafts,),dtype=np.double, order='c')

        # Mass flux tendencies of mean scalars (for output)
        self.massflux_tendency_h = Field.half(Gr)
        self.massflux_tendency_qt = Field.half(Gr)

        # (Eddy) diffusive tendencies of mean scalars (for output)
        self.diffusive_tendency_h = Field.half(Gr)
        self.diffusive_tendency_qt = Field.half(Gr)

        # Vertical fluxes for output
        self.massflux_h = Field.half(Gr)
        self.massflux_qt = Field.half(Gr)
        self.diffusive_flux_h = Field.half(Gr)
        self.diffusive_flux_qt = Field.half(Gr)
        if self.calc_tke:
            self.massflux_tke = Field.half(Gr)

        # Added by Ignacio : Length scheme in use (mls), and smooth min effect (ml_ratio)
        self.mls = Field.half(Gr)
        self.ml_ratio = Field.half(Gr)
        return

    def initialize(self, GMV, tmp):
        self.UpdVar.initialize(GMV, tmp)
        return

    # Initialize the IO pertaining to this class
    def initialize_io(self, Stats):

        self.UpdVar.initialize_io(Stats)
        self.EnvVar.initialize_io(Stats)

        Stats.add_profile('eddy_viscosity')
        Stats.add_profile('eddy_diffusivity')
        Stats.add_profile('entrainment_sc')
        Stats.add_profile('detrainment_sc')
        Stats.add_profile('massflux')
        Stats.add_profile('massflux_h')
        Stats.add_profile('massflux_qt')
        Stats.add_profile('massflux_tendency_h')
        Stats.add_profile('massflux_tendency_qt')
        Stats.add_profile('diffusive_flux_h')
        Stats.add_profile('diffusive_flux_qt')
        Stats.add_profile('diffusive_tendency_h')
        Stats.add_profile('diffusive_tendency_qt')
        Stats.add_profile('total_flux_h')
        Stats.add_profile('total_flux_qt')
        Stats.add_profile('mixing_length')
        Stats.add_profile('updraft_qt_precip')
        Stats.add_profile('updraft_thetal_precip')

        if self.calc_tke:
            Stats.add_profile('tke_buoy')
            Stats.add_profile('tke_dissipation')
            Stats.add_profile('tke_entr_gain')
            Stats.add_profile('tke_detr_loss')
            Stats.add_profile('tke_shear')
            Stats.add_profile('tke_pressure')
            Stats.add_profile('tke_interdomain')

        if self.calc_scalar_var:
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

    def io(self, Stats, tmp):
        mean_entr_sc = Field.half(self.grid)
        mean_detr_sc = Field.half(self.grid)
        massflux     = Field.half(self.grid)
        mf_h         = Field.half(self.grid)
        mf_qt        = Field.half(self.grid)

        self.UpdVar.io(Stats)
        self.EnvVar.io(Stats)

        Stats.write_profile_new('eddy_viscosity'  , self.grid, self.KM.values)
        Stats.write_profile_new('eddy_diffusivity', self.grid, self.KH.values)
        for k in self.grid.over_elems_real(Center()):
            mf_h[k] = interp2pt(self.massflux_h[k], self.massflux_h[k-1])
            mf_qt[k] = interp2pt(self.massflux_qt[k], self.massflux_qt[k-1])
            massflux[k] = interp2pt(self.m[0,k], self.m[0,k-1])
            if self.UpdVar.Area.bulkvalues[k] > 0.0:
                for i in range(self.n_updrafts):
                    mean_entr_sc[k] += self.UpdVar.Area.values[i][k] * self.entr_sc[i][k]/self.UpdVar.Area.bulkvalues[k]
                    mean_detr_sc[k] += self.UpdVar.Area.values[i][k] * self.detr_sc[i][k]/self.UpdVar.Area.bulkvalues[k]

        Stats.write_profile_new('entrainment_sc', self.grid, mean_entr_sc)
        Stats.write_profile_new('detrainment_sc', self.grid, mean_detr_sc)
        Stats.write_profile_new('massflux'      , self.grid, massflux)
        Stats.write_profile_new('massflux_h'    , self.grid, mf_h)
        Stats.write_profile_new('massflux_qt'   , self.grid, mf_qt)
        Stats.write_profile_new('massflux_tendency_h'  , self.grid, self.massflux_tendency_h)
        Stats.write_profile_new('massflux_tendency_qt' , self.grid, self.massflux_tendency_qt)
        Stats.write_profile_new('diffusive_flux_h'     , self.grid, self.diffusive_flux_h)
        Stats.write_profile_new('diffusive_flux_qt'    , self.grid, self.diffusive_flux_qt)
        Stats.write_profile_new('diffusive_tendency_h' , self.grid, self.diffusive_tendency_h)
        Stats.write_profile_new('diffusive_tendency_qt', self.grid, self.diffusive_tendency_qt)
        total_flux_h = mf_h[:] + self.diffusive_flux_h[:]
        total_flux_qt = mf_qt[:] + self.diffusive_flux_qt[:]
        Stats.write_profile_new('total_flux_h', self.grid, total_flux_h)
        Stats.write_profile_new('total_flux_qt',self.grid, total_flux_qt)
        Stats.write_profile_new('mixing_length'        , self.grid, self.mixing_length)
        Stats.write_profile_new('updraft_qt_precip'    , self.grid, self.UpdMicro.prec_source_qt_tot)
        Stats.write_profile_new('updraft_thetal_precip', self.grid, self.UpdMicro.prec_source_h_tot)

        if self.calc_tke:
            self.compute_covariance_dissipation(self.EnvVar.TKE, tmp)
            Stats.write_profile_new('tke_dissipation', self.grid, self.EnvVar.TKE.dissipation)
            Stats.write_profile_new('tke_entr_gain'  , self.grid, self.EnvVar.TKE.entr_gain)
            self.compute_covariance_detr(self.EnvVar.TKE, tmp)
            Stats.write_profile_new('tke_detr_loss'  , self.grid, self.EnvVar.TKE.detr_loss)
            Stats.write_profile_new('tke_shear'      , self.grid, self.EnvVar.TKE.shear)
            Stats.write_profile_new('tke_buoy'       , self.grid, self.EnvVar.TKE.buoy)
            Stats.write_profile_new('tke_pressure'   , self.grid, self.EnvVar.TKE.press)
            Stats.write_profile_new('tke_interdomain', self.grid, self.EnvVar.TKE.interdomain)

        if self.calc_scalar_var:
            self.compute_covariance_dissipation(self.EnvVar.Hvar, tmp)
            Stats.write_profile_new('Hvar_dissipation'  , self.grid, self.EnvVar.Hvar.dissipation)
            self.compute_covariance_dissipation(self.EnvVar.QTvar, tmp)
            Stats.write_profile_new('QTvar_dissipation' , self.grid, self.EnvVar.QTvar.dissipation)
            self.compute_covariance_dissipation(self.EnvVar.HQTcov, tmp)
            Stats.write_profile_new('HQTcov_dissipation', self.grid, self.EnvVar.HQTcov.dissipation)
            Stats.write_profile_new('Hvar_entr_gain'    , self.grid, self.EnvVar.Hvar.entr_gain)
            Stats.write_profile_new('QTvar_entr_gain'   , self.grid, self.EnvVar.QTvar.entr_gain)
            Stats.write_profile_new('HQTcov_entr_gain'  , self.grid, self.EnvVar.HQTcov.entr_gain)
            self.compute_covariance_detr(self.EnvVar.Hvar, tmp)
            self.compute_covariance_detr(self.EnvVar.QTvar, tmp)
            self.compute_covariance_detr(self.EnvVar.HQTcov, tmp)
            Stats.write_profile_new('Hvar_detr_loss'    , self.grid, self.EnvVar.Hvar.detr_loss)
            Stats.write_profile_new('QTvar_detr_loss'   , self.grid, self.EnvVar.QTvar.detr_loss)
            Stats.write_profile_new('HQTcov_detr_loss'  , self.grid, self.EnvVar.HQTcov.detr_loss)
            Stats.write_profile_new('Hvar_shear'        , self.grid, self.EnvVar.Hvar.shear)
            Stats.write_profile_new('QTvar_shear'       , self.grid, self.EnvVar.QTvar.shear)
            Stats.write_profile_new('HQTcov_shear'      , self.grid, self.EnvVar.HQTcov.shear)
            Stats.write_profile_new('Hvar_rain'         , self.grid, self.EnvVar.Hvar.rain_src)
            Stats.write_profile_new('QTvar_rain'        , self.grid, self.EnvVar.QTvar.rain_src)
            Stats.write_profile_new('HQTcov_rain'       , self.grid, self.EnvVar.HQTcov.rain_src)
            Stats.write_profile_new('Hvar_interdomain'  , self.grid, self.EnvVar.Hvar.interdomain)
            Stats.write_profile_new('QTvar_interdomain' , self.grid, self.EnvVar.QTvar.interdomain)
            Stats.write_profile_new('HQTcov_interdomain', self.grid, self.EnvVar.HQTcov.interdomain)
        return


    # Perform the update of the scheme

    def update(self, GMV, Case, TS, tmp, q):

        self.update_inversion(GMV, Case.inversion_option, tmp)

        self.wstar = get_wstar(Case.Sur.bflux, self.zi)

        if TS.nstep == 0:
            self.initialize_covariance(GMV, Case, tmp)
            for k in self.grid.over_elems(Center()):
                if self.calc_tke:
                    self.EnvVar.TKE.values[k] = GMV.TKE.values[k]
                if self.calc_scalar_var:
                    self.EnvVar.Hvar.values[k] = GMV.Hvar.values[k]
                    self.EnvVar.QTvar.values[k] = GMV.QTvar.values[k]
                    self.EnvVar.HQTcov.values[k] = GMV.HQTcov.values[k]

        self.decompose_environment(GMV, 'values')

        if self.use_steady_updrafts:
            self.compute_diagnostic_updrafts(GMV, Case, tmp)
        else:
            self.compute_prognostic_updrafts(GMV, Case, TS, tmp)

        # TODO -maybe not needed? - both diagnostic and prognostic updrafts end with decompose_environment
        # But in general ok here without thermodynamics because MF doesnt depend directly on buoyancy
        self.decompose_environment(GMV, 'values')

        self.update_GMV_MF(GMV, TS, tmp)
        # (###)
        # decompose_environment +  EnvThermo.satadjust + UpdThermo.buoyancy should always be used together
        # This ensures that:
        #   - the buoyancy of updrafts and environment is up to date with the most recent decomposition,
        #   - the buoyancy of updrafts and environment is updated such that
        #     the mean buoyancy with repect to reference state alpha_0 is zero.
        self.decompose_environment(GMV, 'mf_update')
        self.EnvThermo.satadjust(self.EnvVar, True, tmp)
        self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy, tmp)

        self.compute_eddy_diffusivities_tke(GMV, Case)

        self.update_GMV_ED(GMV, Case, TS, tmp, q)
        self.compute_covariance(GMV, Case, TS, tmp)

        # Back out the tendencies of the grid mean variables for the whole timestep by differencing GMV.new and
        # GMV.values
        ParameterizationBase.update(self, GMV, Case, TS)

        return

    def compute_prognostic_updrafts(self, GMV, Case, TS, tmp):

        time_elapsed = 0.0

        self.UpdVar.set_new_with_values()
        self.UpdVar.set_old_with_values()
        self.set_updraft_surface_bc(GMV, Case, tmp)
        self.dt_upd = np.minimum(TS.dt, 0.5 * self.grid.dz/np.fmax(np.max(self.UpdVar.W.values),1e-10))
        while time_elapsed < TS.dt:
            self.compute_entrainment_detrainment(GMV, Case, tmp)
            self.solve_updraft_velocity_area(GMV, TS, tmp)
            self.solve_updraft_scalars(GMV, Case, TS, tmp)
            self.UpdVar.set_values_with_new()
            time_elapsed += self.dt_upd
            self.dt_upd = np.minimum(TS.dt-time_elapsed,  0.5 * self.grid.dz/np.fmax(np.max(self.UpdVar.W.values),1e-10))
            # (####)
            # TODO - see comment (###)
            # It would be better to have a simple linear rule for updating environment here
            # instead of calling EnvThermo saturation adjustment scheme for every updraft.
            # If we are using quadratures this is expensive and probably unnecessary.
            self.decompose_environment(GMV, 'values')
            self.EnvThermo.satadjust(self.EnvVar, False, tmp)
            self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy, tmp)
        return

    def compute_diagnostic_updrafts(self, GMV, Case, tmp):
        gw = self.grid.gw
        dz = self.grid.dz
        dzi = self.grid.dzi

        self.set_updraft_surface_bc(GMV, Case)
        self.compute_entrainment_detrainment(GMV, Case)


        for i in range(self.n_updrafts):
            self.UpdVar.H.values[i][gw] = self.h_surface_bc[i]
            self.UpdVar.QT.values[i][gw] = self.qt_surface_bc[i]
            # Find the cloud liquid content
            T, ql = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, tmp['p_0', gw], self.UpdVar.QT.values[i][gw], self.UpdVar.H.values[i][gw])
            self.UpdVar.QL.values[i][gw] = ql
            self.UpdVar.T.values[i][gw] = T
            self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half, self.UpdVar.T.values,
                                                               self.UpdVar.QT.values, self.UpdVar.QL.values,
                                                               self.UpdVar.QR.values, self.UpdVar.H.values,
                                                               i, gw)
            for k in self.grid.over_elems_real(Center()):
                denom = 1.0 + self.entr_sc[i][k] * dz
                self.UpdVar.H.values[i][k] = (self.UpdVar.H.values[i][k-1] + self.entr_sc[i][k] * dz * GMV.H.values[k])/denom
                self.UpdVar.QT.values[i][k] = (self.UpdVar.QT.values[i][k-1] + self.entr_sc[i][k] * dz * GMV.QT.values[k])/denom


                T, ql = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, tmp['p_0', k], self.UpdVar.QT.values[i][k], self.UpdVar.H.values[i][k])
                self.UpdVar.QL.values[i][k] = ql
                self.UpdVar.T.values[i][k] = T
                self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half, self.UpdVar.T.values,
                                                                   self.UpdVar.QT.values, self.UpdVar.QL.values,
                                                                   self.UpdVar.QR.values, self.UpdVar.H.values,
                                                                   i, k)
        self.UpdVar.QT.set_bcs(self.grid)
        self.UpdVar.QR.set_bcs(self.grid)
        self.UpdVar.H.set_bcs(self.grid)
        # TODO - see comment (####)
        self.decompose_environment(GMV, 'values')
        self.EnvThermo.satadjust(self.EnvVar, False)
        self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy)

        # Solve updraft velocity equation
        for i in range(self.n_updrafts):
            self.UpdVar.W.values[i, self.grid.gw-1] = self.w_surface_bc[i]
            self.entr_sc[i][gw] = 2.0 /dz
            self.detr_sc[i][gw] = 0.0
            for k in self.grid.over_elems_real(Center()):
                area_k = interp2pt(self.UpdVar.Area.values[i][k], self.UpdVar.Area.values[i][k+1])
                if area_k >= self.minimum_area:
                    w_km = self.UpdVar.W.values[i][k-1]
                    entr_w = interp2pt(self.entr_sc[i][k], self.entr_sc[i][k+1])
                    detr_w = interp2pt(self.detr_sc[i][k], self.detr_sc[i][k+1])
                    B_k = interp2pt(self.UpdVar.B.values[i][k], self.UpdVar.B.values[i][k+1])
                    w2 = ((self.vel_buoy_coeff * B_k + 0.5 * w_km * w_km * dzi)
                          /(0.5 * dzi +entr_w + self.vel_pressure_coeff/np.sqrt(np.fmax(area_k,self.minimum_area))))
                    if w2 > 0.0:
                        self.UpdVar.W.values[i][k] = np.sqrt(w2)
                    else:
                        self.UpdVar.W.values[i][k:] = 0
                        break
                else:
                    self.UpdVar.W.values[i][k:] = 0

        self.UpdVar.W.set_bcs(self.grid)

        for i in range(self.n_updrafts):
            au_lim = self.max_area_factor * self.area_surface_bc[i]
            self.UpdVar.Area.values[i][gw] = self.area_surface_bc[i]
            w_mid = 0.5* (self.UpdVar.W.values[i][gw])
            for k in range(gw+1, self.grid.nzg):
                w_low = w_mid
                w_mid = interp2pt(self.UpdVar.W.values[i][k],self.UpdVar.W.values[i][k-1])
                if w_mid > 0.0:
                    if self.entr_sc[i][k]>(0.9/dz):
                        self.entr_sc[i][k] = 0.9/dz

                    self.UpdVar.Area.values[i][k] = (tmp['ρ_0', k-1]*self.UpdVar.Area.values[i][k-1]*w_low/
                                                    (1.0-(self.entr_sc[i][k]-self.detr_sc[i][k])*dz)/w_mid/tmp['ρ_0', k])
                    # # Limit the increase in updraft area when the updraft decelerates
                    if self.UpdVar.Area.values[i][k] >  au_lim:
                        self.UpdVar.Area.values[i][k] = au_lim
                        self.detr_sc[i][k] =(tmp['ρ_0', k-1] * self.UpdVar.Area.values[i][k-1]
                                            * w_low / au_lim / w_mid / tmp['ρ_0', k] + self.entr_sc[i][k] * dz -1.0)/dz
                else:
                    # the updraft has terminated so set its area fraction to zero at this height and all heights above
                    self.UpdVar.Area.values[i][k] = 0.0
                    self.UpdVar.H.values[i][k] = GMV.H.values[k]
                    self.UpdVar.QT.values[i][k] = GMV.QT.values[k]
                    self.UpdVar.QR.values[i][k] = GMV.QR.values[k]
                    #TODO wouldnt it be more consistent to have here?
                    #self.UpdVar.QL.values[i][k] = GMV.QL.values[k]
                    #self.UpdVar.T.values[i][k] = GMV.T.values[k]
                    T, ql = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, tmp['p_0', k], self.UpdVar.QT.values[i][k], self.UpdVar.H.values[i][k])
                    self.UpdVar.QL.values[i][k] = ql
                    self.UpdVar.T.values[i][k] = T

        # TODO - see comment (####)
        self.decompose_environment(GMV, 'values')
        self.EnvThermo.satadjust(self.EnvVar, False)
        self.UpdThermo.buoyancy(self.UpdVar, self.EnvVar, GMV, self.extrapolate_buoyancy)

        self.UpdVar.Area.set_bcs(self.grid)

        self.UpdMicro.prec_source_h_tot  = np.sum(np.multiply(self.UpdMicro.prec_source_h,  self.UpdVar.Area.values), axis=0)
        self.UpdMicro.prec_source_qt_tot = np.sum(np.multiply(self.UpdMicro.prec_source_qt, self.UpdVar.Area.values), axis=0)

        return

    def update_inversion(self,GMV, option, tmp):
        ParameterizationBase.update_inversion(self, GMV,option, tmp)
        return

    def compute_mixing_length(self, obukhov_length):
        tau =  get_mixing_tau(self.zi, self.wstar)

        for k in self.grid.over_elems_real(Center()):
            l1 = tau * np.sqrt(np.fmax(self.EnvVar.TKE.values[k],0.0))
            z_ = self.grid.z_half[k]
            if obukhov_length < 0.0: #unstable
                l2 = vkb * z_ * ( (1.0 - 100.0 * z_/obukhov_length)**0.2 )
            elif obukhov_length > 0.0: #stable
                l2 = vkb * z_ /  (1. + 2.7 *z_/obukhov_length)
            else:
                l2 = vkb * z_
            self.mixing_length[k] = np.fmax( 1.0/(1.0/np.fmax(l1,1e-10) + 1.0/l2), 1e-3)
        return


    def compute_eddy_diffusivities_tke(self, GMV, Case):
        if self.similarity_diffusivity:
            ParameterizationBase.compute_eddy_diffusivities_similarity(self,GMV, Case)
        else:
            self.compute_mixing_length(Case.Sur.obukhov_length)
            for k in self.grid.over_elems_real(Center()):
                lm = self.mixing_length[k]
                self.KM.values[k] = self.tke_ed_coeff * lm * np.sqrt(np.fmax(self.EnvVar.TKE.values[k],0.0) )
                # Prandtl number is fixed. It should be defined as a function of height - Ignacio
                self.KH.values[k] = self.KM.values[k] / self.prandtl_number

        return

    def set_updraft_surface_bc(self, GMV, Case, tmp):

        self.update_inversion(GMV, Case.inversion_option, tmp)
        self.wstar = get_wstar(Case.Sur.bflux, self.zi)

        gw = self.grid.gw
        zLL = self.grid.z_half[gw]
        ustar = Case.Sur.ustar
        oblength = Case.Sur.obukhov_length
        alpha0LL  = tmp['α_0', gw]
        qt_var = get_surface_variance(Case.Sur.rho_qtflux*alpha0LL, Case.Sur.rho_qtflux*alpha0LL, ustar, zLL, oblength)
        h_var  = get_surface_variance(Case.Sur.rho_hflux*alpha0LL,  Case.Sur.rho_hflux*alpha0LL,  ustar, zLL, oblength)

        for i in range(self.n_updrafts):
            self.area_surface_bc[i] = self.surface_area/self.n_updrafts
            self.w_surface_bc[i] = 0.0
            self.h_surface_bc[i] = (GMV.H.values[gw] + self.surface_scalar_coeff[i] * np.sqrt(h_var))
            self.qt_surface_bc[i] = (GMV.QT.values[gw] + self.surface_scalar_coeff[i] * np.sqrt(qt_var))
        return

    def reset_surface_covariance(self, GMV, Case, tmp):
        flux1 = Case.Sur.rho_hflux
        flux2 = Case.Sur.rho_qtflux
        k_1 = self.grid.first_interior(Zmin())
        zLL = self.grid.z_half[k_1]
        ustar = Case.Sur.ustar
        oblength = Case.Sur.obukhov_length
        alpha0LL  = self.Ref.alpha0_half[k_1]
        if self.calc_tke:
            GMV.TKE.values[k_1] = get_surface_tke(Case.Sur.ustar,
                                                     self.wstar,
                                                     self.grid.z_half[k_1],
                                                     Case.Sur.obukhov_length)
        if self.calc_scalar_var:
            GMV.Hvar.values[k_1]   = get_surface_variance(flux1*alpha0LL,flux1*alpha0LL, ustar, zLL, oblength)
            GMV.QTvar.values[k_1]  = get_surface_variance(flux2*alpha0LL,flux2*alpha0LL, ustar, zLL, oblength)
            GMV.HQTcov.values[k_1] = get_surface_variance(flux1*alpha0LL,flux2*alpha0LL, ustar, zLL, oblength)
        return


    # Find values of environmental variables by subtracting updraft values from grid mean values
    # whichvals used to check which substep we are on--correspondingly use 'GMV.SomeVar.value' (last timestep value)
    # or GMV.SomeVar.mf_update (GMV value following massflux substep)
    def decompose_environment(self, GMV, whichvals):

        # first make sure the 'bulkvalues' of the updraft variables are updated
        self.UpdVar.set_means(GMV)

        if whichvals == 'values':
            for k in self.grid.over_elems(Center()):
                val1 = 1.0/(1.0-self.UpdVar.Area.bulkvalues[k])
                val2 = self.UpdVar.Area.bulkvalues[k] * val1
                self.EnvVar.QT.values[k] = val1 * GMV.QT.values[k] - val2 * self.UpdVar.QT.bulkvalues[k]
                self.EnvVar.H.values[k] = val1 * GMV.H.values[k] - val2 * self.UpdVar.H.bulkvalues[k]
                # Have to account for staggering of W--interpolate area fraction to the "full" grid points
                # Assuming GMV.W = 0!
                au_full = 0.5 * (self.UpdVar.Area.bulkvalues[k+1] + self.UpdVar.Area.bulkvalues[k])
                self.EnvVar.W.values[k] = -au_full/(1.0-au_full) * self.UpdVar.W.bulkvalues[k]
        elif whichvals == 'mf_update':
            # same as above but replace GMV.SomeVar.values with GMV.SomeVar.mf_update
            for k in self.grid.over_elems(Center()):
                val1 = 1.0/(1.0-self.UpdVar.Area.bulkvalues[k])
                val2 = self.UpdVar.Area.bulkvalues[k] * val1

                self.EnvVar.QT.values[k] = val1 * GMV.QT.mf_update[k] - val2 * self.UpdVar.QT.bulkvalues[k]
                self.EnvVar.H.values[k] = val1 * GMV.H.mf_update[k] - val2 * self.UpdVar.H.bulkvalues[k]
                # Have to account for staggering of W
                # Assuming GMV.W = 0!
                au_full = 0.5 * (self.UpdVar.Area.bulkvalues[k+1] + self.UpdVar.Area.bulkvalues[k])
                self.EnvVar.W.values[k] = -au_full/(1.0-au_full) * self.UpdVar.W.bulkvalues[k]

        if self.calc_tke:
            self.get_GMV_CoVar(self.UpdVar.Area, self.UpdVar.W,  self.UpdVar.W,  self.EnvVar.W,  self.EnvVar.W,  self.EnvVar.TKE,    GMV.W.values,  GMV.W.values,  GMV.TKE.values)
        if self.calc_scalar_var:
            self.get_GMV_CoVar(self.UpdVar.Area, self.UpdVar.H,  self.UpdVar.H,  self.EnvVar.H,  self.EnvVar.H,  self.EnvVar.Hvar,   GMV.H.values,  GMV.H.values,  GMV.Hvar.values)
            self.get_GMV_CoVar(self.UpdVar.Area, self.UpdVar.QT, self.UpdVar.QT, self.EnvVar.QT, self.EnvVar.QT, self.EnvVar.QTvar,  GMV.QT.values, GMV.QT.values, GMV.QTvar.values)
            self.get_GMV_CoVar(self.UpdVar.Area, self.UpdVar.H,  self.UpdVar.QT, self.EnvVar.H,  self.EnvVar.QT, self.EnvVar.HQTcov, GMV.H.values,  GMV.QT.values, GMV.HQTcov.values)


        return

    # Note: this assumes all variables are defined on half levels not full levels (i.e. phi, psi are not w)
    def get_GMV_CoVar(self, au,
                        phi_u, psi_u,
                        phi_e,  psi_e,
                        covar_e,
                       gmv_phi, gmv_psi, gmv_covar):
        ae = Field.half(self.grid)
        for k in self.grid.over_elems(Center()):
            ae[k] = 1.0 - self.UpdVar.Area.bulkvalues[k]
        tke_factor = 1.0


        for k in self.grid.over_elems(Center()):
            if covar_e.name == 'tke':
                tke_factor = 0.5
                phi_diff = interp2pt(phi_e.values[k-1]-gmv_phi[k-1], phi_e.values[k]-gmv_phi[k])
                psi_diff = interp2pt(psi_e.values[k-1]-gmv_psi[k-1], psi_e.values[k]-gmv_psi[k])
            else:
                tke_factor = 1.0
                phi_diff = phi_e.values[k]-gmv_phi[k]
                psi_diff = psi_e.values[k]-gmv_psi[k]


            gmv_covar[k] = tke_factor * ae[k] * phi_diff * psi_diff + ae[k] * covar_e.values[k]
            for i in range(self.n_updrafts):
                if covar_e.name == 'tke':
                    phi_diff = interp2pt(phi_u.values[i][k-1]-gmv_phi[k-1], phi_u.values[i][k]-gmv_phi[k])
                    psi_diff = interp2pt(psi_u.values[i][k-1]-gmv_psi[k-1], psi_u.values[i][k]-gmv_psi[k])
                else:
                    phi_diff = phi_u.values[i][k]-gmv_phi[k]
                    psi_diff = psi_u.values[i][k]-gmv_psi[k]

                gmv_covar[k] += tke_factor * au.values[i][k] * phi_diff * psi_diff
        return


    def get_env_covar_from_GMV(self, au,
                                phi_u, psi_u,
                                phi_e, psi_e,
                                covar_e,
                                gmv_phi, gmv_psi, gmv_covar):
        ae = Field.half(self.grid)
        for k in self.grid.over_elems(Center()):
            ae[k] = 1.0 - self.UpdVar.Area.bulkvalues[k]
        tke_factor = 1.0
        if covar_e.name == 'tke':
            tke_factor = 0.5

        for k in self.grid.over_elems(Center()):
            if ae[k] > 0.0:
                if covar_e.name == 'tke':
                    phi_diff = interp2pt(phi_e.values[k-1] - gmv_phi[k-1],phi_e.values[k] - gmv_phi[k])
                    psi_diff = interp2pt(psi_e.values[k-1] - gmv_psi[k-1],psi_e.values[k] - gmv_psi[k])
                else:
                    phi_diff = phi_e.values[k] - gmv_phi[k]
                    psi_diff = psi_e.values[k] - gmv_psi[k]

                covar_e.values[k] = gmv_covar[k] - tke_factor * ae[k] * phi_diff * psi_diff
                for i in range(self.n_updrafts):
                    if covar_e.name == 'tke':
                        phi_diff = interp2pt(phi_u.values[i][k-1] - gmv_phi[k-1],phi_u.values[i][k] - gmv_phi[k])
                        psi_diff = interp2pt(psi_u.values[i][k-1] - gmv_psi[k-1],psi_u.values[i][k] - gmv_psi[k])
                    else:
                        phi_diff = phi_u.values[i][k] - gmv_phi[k]
                        psi_diff = psi_u.values[i][k] - gmv_psi[k]

                    covar_e.values[k] -= tke_factor * au.values[i][k] * phi_diff * psi_diff
                covar_e.values[k] = covar_e.values[k]/ae[k]
            else:
                covar_e.values[k] = 0.0
        return

    def compute_entrainment_detrainment(self, GMV, Case, tmp):
        quadrature_order = 3


        self.UpdVar.get_cloud_base_top_cover()

        input_st = type('', (), {})()
        input_st.wstar = self.wstar

        input_st.b_mean = 0
        input_st.dz = self.grid.dz
        input_st.zbl = self.compute_zbl_qt_grad(GMV)
        for i in range(self.n_updrafts):
            input_st.zi = self.UpdVar.cloud_base[i]
            for k in self.grid.over_elems_real(Center()):
                input_st.quadrature_order = quadrature_order
                input_st.b = self.UpdVar.B.values[i][k]
                input_st.w = interp2pt(self.UpdVar.W.values[i][k],self.UpdVar.W.values[i][k-1])
                input_st.z = self.grid.z_half[k]
                input_st.af = self.UpdVar.Area.values[i][k]
                input_st.tke = self.EnvVar.TKE.values[k]
                input_st.ml = self.mixing_length[k]
                input_st.qt_env = self.EnvVar.QT.values[k]
                input_st.ql_env = self.EnvVar.QL.values[k]
                input_st.H_env = self.EnvVar.H.values[k]
                input_st.b_env = self.EnvVar.B.values[k]
                input_st.w_env = self.EnvVar.W.values[k]
                input_st.H_up = self.UpdVar.H.values[i][k]
                input_st.qt_up = self.UpdVar.QT.values[i][k]
                input_st.ql_up = self.UpdVar.QL.values[i][k]
                input_st.p0 = tmp['p_0', k]
                input_st.alpha0 = tmp['α_0', k]
                input_st.env_Hvar = self.EnvVar.Hvar.values[k]
                input_st.env_QTvar = self.EnvVar.QTvar.values[k]
                input_st.env_HQTcov = self.EnvVar.HQTcov.values[k]

                if self.calc_tke:
                        input_st.tke = self.EnvVar.TKE.values[k]
                        input_st.tke_ed_coeff  = self.tke_ed_coeff

                input_st.T_mean = (self.EnvVar.T.values[k]+self.UpdVar.T.values[i][k])/2
                input_st.L = 20000.0 # need to define the scale of the GCM grid resolution
                ## Ignacio
                input_st.n_up = self.n_updrafts
                input_st.thv_e = theta_virt_c(tmp['p_0', k], self.EnvVar.T.values[k], self.EnvVar.QT.values[k],
                     self.EnvVar.QL.values[k], self.EnvVar.QR.values[k])
                input_st.thv_u = theta_virt_c(tmp['p_0', k], self.UpdVar.T.bulkvalues[k], self.UpdVar.QT.bulkvalues[k],
                     self.UpdVar.QL.bulkvalues[k], self.UpdVar.QR.bulkvalues[k])
                input_st.dwdz = (self.UpdVar.Area.values[i][k+1]*
                    interp2pt(self.UpdVar.W.values[i][k+1],self.UpdVar.W.values[i][k]) +
                    (1.0-self.UpdVar.Area.values[i][k+1])*self.EnvVar.W.values[k+1] -
                    (self.UpdVar.Area.values[i][k-1]*
                    interp2pt(self.UpdVar.W.values[i][k-1],self.UpdVar.W.values[i][k-2]) +
                    (1.0-self.UpdVar.Area.values[i][k-1])*self.EnvVar.W.values[k-1]) )/(2.0*self.grid.dz)

                transport_plus = ( self.UpdVar.Area.values[i][k+1]*(1.0-self.UpdVar.Area.values[i][k+1])*
                    (interp2pt(self.UpdVar.W.values[i][k+1],self.UpdVar.W.values[i][k]) - self.EnvVar.W.values[k+1])*
                    (1.0-2.0*self.UpdVar.Area.values[i][k+1])*
                    (interp2pt(self.UpdVar.W.values[i][k+1],self.UpdVar.W.values[i][k]) - self.EnvVar.W.values[k+1])*
                    (interp2pt(self.UpdVar.W.values[i][k+1],self.UpdVar.W.values[i][k]) - self.EnvVar.W.values[k+1]) )

                transport_minus = ( self.UpdVar.Area.values[i][k-1]*(1.0-self.UpdVar.Area.values[i][k-1])*
                    (interp2pt(self.UpdVar.W.values[i][k-1],self.UpdVar.W.values[i][k-2]) - self.EnvVar.W.values[k-1])*
                    (1.0-2.0*self.UpdVar.Area.values[i][k+1])*
                    (interp2pt(self.UpdVar.W.values[i][k-1],self.UpdVar.W.values[i][k-2]) - self.EnvVar.W.values[k-1])*
                    (interp2pt(self.UpdVar.W.values[i][k-1],self.UpdVar.W.values[i][k-2]) - self.EnvVar.W.values[k-1]) )

                input_st.transport_der = (transport_plus - transport_minus)/2.0/self.grid.dz

                if input_st.zbl-self.UpdVar.cloud_base[i] > 0.0:
                    input_st.poisson = np.random.poisson(self.grid.dz/((input_st.zbl-self.UpdVar.cloud_base[i])/10.0))
                else:
                    input_st.poisson = 0.0
                ## End: Ignacio
                ret = self.entr_detr_fp(input_st)
                self.entr_sc[i][k] = ret.entr_sc * self.entrainment_factor
                self.detr_sc[i][k] = ret.detr_sc * self.detrainment_factor

        return

    def compute_zbl_qt_grad(self, GMV):
    # computes inversion height as z with max gradient of qt
        zbl_qt = 0.0
        qt_grad = 0.0

        for k in self.grid.over_elems_real(Center()):
            z_ = self.grid.z_half[k]
            qt_up = GMV.QT.values[k+1]
            qt_ = GMV.QT.values[k]

            if np.fabs(qt_up-qt_)*self.grid.dzi > qt_grad:
                qt_grad = np.fabs(qt_up-qt_)*self.grid.dzi
                zbl_qt = z_

        return zbl_qt

    def solve_updraft_velocity_area(self, GMV, TS, tmp):
        gw = self.grid.gw
        dzi = self.grid.dzi
        dti_ = 1.0/self.dt_upd
        dt_ = 1.0/dti_

        for i in range(self.n_updrafts):
            self.entr_sc[i][gw] = 2.0 * dzi
            self.detr_sc[i][gw] = 0.0
            self.UpdVar.W.new[i][gw-1] = self.w_surface_bc[i]
            self.UpdVar.Area.new[i][gw] = self.area_surface_bc[i]
            au_lim = self.area_surface_bc[i] * self.max_area_factor

            for k in self.grid.over_elems_real(Center()):

                # First solve for updated area fraction at k+1
                whalf_kp = interp2pt(self.UpdVar.W.values[i][k], self.UpdVar.W.values[i][k+1])
                whalf_k = interp2pt(self.UpdVar.W.values[i][k-1], self.UpdVar.W.values[i][k])
                a_kp = self.UpdVar.Area.values[i][k+1]
                a_k = self.UpdVar.Area.values[i][k]
                rho_kp = tmp['ρ_0', k+1]
                rho_k = tmp['ρ_0', k]
                alpha0_kp = tmp['α_0', k+1]
                adv = - alpha0_kp * dzi *( rho_kp * a_kp * whalf_kp - rho_k * a_k * whalf_k)

                entr_term = self.UpdVar.Area.values[i][k+1] * whalf_kp * (self.entr_sc[i][k+1] )
                detr_term = self.UpdVar.Area.values[i][k+1] * whalf_kp * (- self.detr_sc[i][k+1])


                self.UpdVar.Area.new[i][k+1]  = np.fmax(dt_ * (adv + entr_term + detr_term) + self.UpdVar.Area.values[i][k+1], 0.0)
                if self.UpdVar.Area.new[i][k+1] > au_lim:
                    self.UpdVar.Area.new[i][k+1] = au_lim
                    if self.UpdVar.Area.values[i][k+1] > 0.0:
                        self.detr_sc[i][k+1] = (((au_lim-self.UpdVar.Area.values[i][k+1])* dti_ - adv -entr_term)/(-self.UpdVar.Area.values[i][k+1]  * whalf_kp))
                    else:
                        # this detrainment rate won't affect scalars but would affect velocity
                        self.detr_sc[i][k+1] = (((au_lim-self.UpdVar.Area.values[i][k+1])* dti_ - adv -entr_term)/(-au_lim  * whalf_kp))

                # Now solve for updraft velocity at k
                rho_ratio = self.Ref.rho0[k-1]/self.Ref.rho0[k]
                anew_k = interp2pt(self.UpdVar.Area.new[i][k], self.UpdVar.Area.new[i][k+1])
                if anew_k >= self.minimum_area:
                    a_k = interp2pt(self.UpdVar.Area.values[i][k], self.UpdVar.Area.values[i][k+1])
                    a_km = interp2pt(self.UpdVar.Area.values[i][k-1], self.UpdVar.Area.values[i][k])
                    entr_w = interp2pt(self.entr_sc[i][k], self.entr_sc[i][k+1])
                    detr_w = interp2pt(self.detr_sc[i][k], self.detr_sc[i][k+1])
                    B_k = interp2pt(self.UpdVar.B.values[i][k], self.UpdVar.B.values[i][k+1])
                    adv = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i][k] * self.UpdVar.W.values[i][k] * dzi
                           - self.Ref.rho0[k-1] * a_km * self.UpdVar.W.values[i][k-1] * self.UpdVar.W.values[i][k-1] * dzi)
                    exch = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i][k]
                            * (entr_w * self.EnvVar.W.values[k] - detr_w * self.UpdVar.W.values[i][k] ))
                    buoy= self.Ref.rho0[k] * a_k * B_k
                    press_buoy =  -1.0 * self.Ref.rho0[k] * a_k * B_k * self.pressure_buoy_coeff
                    press_drag = -1.0 * self.Ref.rho0[k] * a_k * (self.pressure_drag_coeff/self.pressure_plume_spacing
                                                                 * (self.UpdVar.W.values[i][k] -self.EnvVar.W.values[k])**2.0/np.sqrt(np.fmax(a_k,self.minimum_area)))
                    press = press_buoy + press_drag
                    self.updraft_pressure_sink[i][k] = press
                    self.UpdVar.W.new[i][k] = (self.Ref.rho0[k] * a_k * self.UpdVar.W.values[i][k] * dti_
                                              -adv + exch + buoy + press)/(self.Ref.rho0[k] * anew_k * dti_)
                    if self.UpdVar.W.new[i][k] <= 0.0:
                        self.UpdVar.W.new[i][k:] = 0.0
                        self.UpdVar.Area.new[i][k+1:] = 0.0
                        break
                else:
                    self.UpdVar.W.new[i][k:] = 0.0
                    self.UpdVar.Area.new[i][k+1:] = 0.0
                    # keep this in mind if we modify updraft top treatment!
                    self.updraft_pressure_sink[i][k:] = 0.0
                    break

        return

    def solve_updraft_scalars(self, GMV, Case, TS, tmp):
        dzi = self.grid.dzi
        dti_ = 1.0/self.dt_upd
        gw = self.grid.gw

        for i in range(self.n_updrafts):
            self.UpdVar.H.new[i][gw] = self.h_surface_bc[i]
            self.UpdVar.QT.new[i][gw] = self.qt_surface_bc[i]
            self.UpdVar.QR.new[i][gw] = 0.0 #TODO

            if self.use_local_micro:
                # do saturation adjustment
                T, ql = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, tmp['p_0', gw], self.UpdVar.QT.new[i][gw], self.UpdVar.H.new[i][gw])
                self.UpdVar.QL.new[i][gw] = ql
                self.UpdVar.T.new[i][gw] = T
                # remove precipitation (update QT, QL and H)
                self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half, self.UpdVar.T.new,
                                                                   self.UpdVar.QT.new, self.UpdVar.QL.new,
                                                                   self.UpdVar.QR.new, self.UpdVar.H.new,
                                                                   i, gw)

            # starting from the bottom do entrainment at each level
            for k in range(gw+1, self.grid.nzg-gw):
                H_entr = self.EnvVar.H.values[k]
                QT_entr = self.EnvVar.QT.values[k]

                # write the discrete equations in form:
                # c1 * phi_new[k] = c2 * phi[k] + c3 * phi[k-1] + c4 * phi_entr
                if self.UpdVar.Area.new[i][k] >= self.minimum_area:
                    m_k = (tmp['ρ_0', k] * self.UpdVar.Area.values[i][k]
                           * interp2pt(self.UpdVar.W.values[i][k-1], self.UpdVar.W.values[i][k]))
                    m_km = (tmp['ρ_0', k-1] * self.UpdVar.Area.values[i][k-1]
                           * interp2pt(self.UpdVar.W.values[i][k-2], self.UpdVar.W.values[i][k-1]))
                    c1 = tmp['ρ_0', k] * self.UpdVar.Area.new[i][k] * dti_
                    c2 = (tmp['ρ_0', k] * self.UpdVar.Area.values[i][k] * dti_
                          - m_k * (dzi + self.detr_sc[i][k]))
                    c3 = m_km * dzi
                    c4 = m_k * self.entr_sc[i][k]

                    self.UpdVar.H.new[i][k] =  (c2 * self.UpdVar.H.values[i][k]  + c3 * self.UpdVar.H.values[i][k-1]
                                               + c4 * H_entr)/c1
                    self.UpdVar.QT.new[i][k] = (c2 * self.UpdVar.QT.values[i][k] + c3 * self.UpdVar.QT.values[i][k-1]
                                               + c4* QT_entr)/c1
                else:
                    self.UpdVar.H.new[i][k] = GMV.H.values[k]
                    self.UpdVar.QT.new[i][k] = GMV.QT.values[k]

                # find new temperature
                T, ql = eos(self.UpdThermo.t_to_prog_fp,self.UpdThermo.prog_to_t_fp, tmp['p_0', k], self.UpdVar.QT.new[i][k], self.UpdVar.H.new[i][k])
                self.UpdVar.QL.new[i][k] = ql
                self.UpdVar.T.new[i][k] = T

                if self.use_local_micro:
                    # remove precipitation (pdate QT, QL and H)
                    self.UpdMicro.compute_update_combined_local_thetal(self.Ref.p0_half, self.UpdVar.T.new,
                                                                       self.UpdVar.QT.new, self.UpdVar.QL.new,
                                                                       self.UpdVar.QR.new, self.UpdVar.H.new,
                                                                       i, k)

        if self.use_local_micro:
            # save the total source terms for H and QT due to precipitation
            # TODO - add QR source
            self.UpdMicro.prec_source_h_tot = np.sum(np.multiply(self.UpdMicro.prec_source_h,
                                                                 self.UpdVar.Area.values), axis=0)
            self.UpdMicro.prec_source_qt_tot = np.sum(np.multiply(self.UpdMicro.prec_source_qt,
                                                                  self.UpdVar.Area.values), axis=0)
        else:
            # Compute the updraft microphysical sources (precipitation)
            #after the entrainment loop is finished
            self.UpdMicro.compute_sources(self.UpdVar)
            # Update updraft variables with microphysical source tendencies
            self.UpdMicro.update_updraftvars(self.UpdVar)

        self.UpdVar.H.set_bcs(self.grid)
        self.UpdVar.QT.set_bcs(self.grid)
        self.UpdVar.QR.set_bcs(self.grid)
        return

    # After updating the updraft variables themselves:
    # 1. compute the mass fluxes (currently not stored as class members, probably will want to do this
    # for output purposes)
    # 2. Apply mass flux tendencies and updraft microphysical tendencies to GMV.SomeVar.Values (old time step values)
    # thereby updating to GMV.SomeVar.mf_update
    # mass flux tendency is computed as 1st order upwind

    def update_GMV_MF(self, GMV, TS, tmp):
        gw = self.grid.gw
        mf_tend_h=0.0
        mf_tend_qt=0.0
        self.massflux_h[:] = 0.0
        self.massflux_qt[:] = 0.0

        # Compute the mass flux and associated scalar fluxes
        for i in range(self.n_updrafts):
            self.m[i][gw-1] = 0.0
            for k in self.grid.over_elems_real(Center()):
                self.m[i][k] = ((self.UpdVar.W.values[i][k] - self.EnvVar.W.values[k] )* self.Ref.rho0[k]
                               * interp2pt(self.UpdVar.Area.values[i][k],self.UpdVar.Area.values[i][k+1]))

        self.massflux_h[gw-1] = 0.0
        self.massflux_qt[gw-1] = 0.0
        for k in range(gw, self.grid.nzg-gw-1):
            self.massflux_h[k] = 0.0
            self.massflux_qt[k] = 0.0
            env_h_interp = interp2pt(self.EnvVar.H.values[k], self.EnvVar.H.values[k+1])
            env_qt_interp = interp2pt(self.EnvVar.QT.values[k], self.EnvVar.QT.values[k+1])
            for i in range(self.n_updrafts):
                self.massflux_h[k] += self.m[i][k] * (interp2pt(self.UpdVar.H.values[i][k],
                                                               self.UpdVar.H.values[i][k+1]) - env_h_interp )
                self.massflux_qt[k] += self.m[i][k] * (interp2pt(self.UpdVar.QT.values[i][k],
                                                                self.UpdVar.QT.values[i][k+1]) - env_qt_interp )

        # Compute the  mass flux tendencies
        # Adjust the values of the grid mean variables

        for k in self.grid.over_elems_real(Center()):
            mf_tend_h = -(self.massflux_h[k] - self.massflux_h[k-1]) * (tmp['α_0', k] * self.grid.dzi)
            mf_tend_qt = -(self.massflux_qt[k] - self.massflux_qt[k-1]) * (tmp['α_0', k] * self.grid.dzi)

            GMV.H.mf_update[k] = GMV.H.values[k] +  TS.dt * mf_tend_h + self.UpdMicro.prec_source_h_tot[k]
            GMV.QT.mf_update[k] = GMV.QT.values[k] + TS.dt * mf_tend_qt + self.UpdMicro.prec_source_qt_tot[k]

            #No mass flux tendency for U, V
            GMV.U.mf_update[k] = GMV.U.values[k]
            GMV.V.mf_update[k] = GMV.V.values[k]
            # Prepare the output
            self.massflux_tendency_h[k] = mf_tend_h
            self.massflux_tendency_qt[k] = mf_tend_qt


        GMV.H.set_bcs(self.grid)
        GMV.QT.set_bcs(self.grid)
        GMV.QR.set_bcs(self.grid)
        GMV.U.set_bcs(self.grid)
        GMV.V.set_bcs(self.grid)

        return

    # Update the grid mean variables with the tendency due to eddy diffusion
    # Km and Kh have already been updated
    # 2nd order finite differences plus implicit time step allows solution with tridiagonal matrix solver
    # Update from GMV.SomeVar.mf_update to GMV.SomeVar.new
    def update_GMV_ED(self, GMV, Case, TS, tmp, q):
        gw = self.grid.gw
        nzg = self.grid.nzg
        nz = self.grid.nz
        dzi = self.grid.dzi
        a = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
        b = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
        c = np.zeros((nz,),dtype=np.double, order='c') # for tridiag solver
        x = Field.half(self.grid)
        ae = Field.half(self.grid)
        rho_ae_K_m = Field.full(self.grid)

        for k in self.grid.over_elems(Center()):
            ae[k] = 1.0 - self.UpdVar.Area.bulkvalues[k]

        for k in self.grid.over_elems_real(Node()):
            rho_ae_K_m[k] = 0.5 * (ae[k]+ae[k+1]) * 0.5 * (self.KH.values[k]+self.KH.values[k+1]) * self.Ref.rho0[k]

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion(nzg, gw, dzi, TS.dt, rho_ae_K_m, self.Ref.rho0_half, ae, a, b, c)

        slice_real = self.grid.slice_real(Center())
        # Solve QT
        for k in self.grid.over_elems(Center()):
            x[k] =  self.EnvVar.QT.values[k]
        ki = self.grid.first_interior(Zmin())
        x[ki] = x[ki] + TS.dt * Case.Sur.rho_qtflux * dzi * self.Ref.alpha0_half[ki]/ae[ki]
        tridiag_solve(self.grid.nz, x[slice_real], a, b, c)

        for k in self.grid.over_elems(Center()):
            GMV.QT.new[k] = GMV.QT.mf_update[k] + ae[k] *(x[k] - self.EnvVar.QT.values[k])

        # get the diffusive flux
        self.diffusive_tendency_qt[k] = (GMV.QT.new[k] - GMV.QT.mf_update[k]) * TS.dti
        for k in self.grid.over_elems(Center()):
            self.diffusive_flux_qt[k] = -0.5 * tmp['ρ_0', k]*ae[k] * self.KH.values[k] * dzi * (self.EnvVar.QT.values[k+1]-self.EnvVar.QT.values[k-1])
        bo = self.grid.boundary(Zmin())
        self.diffusive_flux_qt[ki] = interp2pt(Case.Sur.rho_qtflux, -rho_ae_K_m[bo+1] * dzi *(self.EnvVar.QT.values[ki+1]-self.EnvVar.QT.values[ki]) )

        # Solve H
        for k in self.grid.over_elems(Center()):
            x[k] = self.EnvVar.H.values[k]
        x[ki] = x[ki] + TS.dt * Case.Sur.rho_hflux * dzi * self.Ref.alpha0_half[ki]/ae[ki]
        tridiag_solve(self.grid.nz, x[slice_real], a, b, c)

        for k in self.grid.over_elems(Center()):
            GMV.H.new[k] = GMV.H.mf_update[k] + ae[k] *(x[k] - self.EnvVar.H.values[k])
            self.diffusive_flux_qt[k] = (GMV.H.new[k] - GMV.H.mf_update[k]) * TS.dti
        # get the diffusive flux
        for k in self.grid.over_elems(Center()):
            self.diffusive_flux_h[k] = -0.5 * tmp['ρ_0', k]*ae[k] * self.KH.values[k] * dzi * (self.EnvVar.H.values[k+1]-self.EnvVar.H.values[k-1])

        bo = self.grid.boundary(Zmin())
        self.diffusive_flux_h[ki] = interp2pt(Case.Sur.rho_hflux, -rho_ae_K_m[bo+1] * dzi *(self.EnvVar.H.values[ki+1]-self.EnvVar.H.values[ki]) )

        # Solve U
        for k in self.grid.over_elems_real(Node()):
            rho_ae_K_m[k] = 0.5 * (ae[k]+ae[k+1]) * 0.5 * (self.KH.values[k]+self.KH.values[k+1]) * self.Ref.rho0[k]

        # Matrix is the same for all variables that use the same eddy diffusivity, we can construct once and reuse
        construct_tridiag_diffusion(nzg, gw, dzi, TS.dt, rho_ae_K_m, self.Ref.rho0_half, ae, a, b, c)

        for k in self.grid.over_elems(Center()):
            x[k] = GMV.U.values[k]
        x[ki] = x[ki] + TS.dt * Case.Sur.rho_uflux * dzi * self.Ref.alpha0_half[ki]/ae[ki]
        tridiag_solve(self.grid.nz, x[slice_real], a, b, c)

        for k in self.grid.over_elems(Center()):
            GMV.U.new[k] = x[k]

        # Solve V
        for k in self.grid.over_elems(Center()):
            x[k] = GMV.V.values[k]
        x[ki] = x[ki] + TS.dt * Case.Sur.rho_vflux * dzi * self.Ref.alpha0_half[ki]/ae[ki]
        tridiag_solve(self.grid.nz, x[slice_real], a, b, c)

        for k in self.grid.over_elems(Center()):
            GMV.V.new[k] = x[k]

        GMV.QT.set_bcs(self.grid)
        GMV.QR.set_bcs(self.grid)
        GMV.H.set_bcs(self.grid)
        GMV.U.set_bcs(self.grid)
        GMV.V.set_bcs(self.grid)

        return

    def compute_tke_buoy(self, GMV, tmp):
        grad_thl_minus=0.0
        grad_qt_minus=0.0
        grad_thl_plus=0
        grad_qt_plus=0
        ae = Field.half(self.grid)
        for k in self.grid.over_elems(Center()):
            ae[k] = 1.0 - self.UpdVar.Area.bulkvalues[k]

        # Note that source terms at the gw grid point are not really used because that is where tke boundary condition is
        # enforced (according to MO similarity). Thus here I am being sloppy about lowest grid point
        for k in self.grid.over_elems_real(Center()):
            qt_dry = self.EnvThermo.qt_dry[k]
            th_dry = self.EnvThermo.th_dry[k]
            t_cloudy = self.EnvThermo.t_cloudy[k]
            qv_cloudy = self.EnvThermo.qv_cloudy[k]
            qt_cloudy = self.EnvThermo.qt_cloudy[k]
            th_cloudy = self.EnvThermo.th_cloudy[k]

            lh = latent_heat(t_cloudy)
            cpm = cpm_c(qt_cloudy)
            grad_thl_minus = grad_thl_plus
            grad_qt_minus = grad_qt_plus
            grad_thl_plus = (self.EnvVar.THL.values[k+1] - self.EnvVar.THL.values[k]) * self.grid.dzi
            grad_qt_plus  = (self.EnvVar.QT.values[k+1]  - self.EnvVar.QT.values[k])  * self.grid.dzi

            prefactor = Rd * exner_c(tmp['p_0', k])/tmp['p_0', k]

            d_alpha_thetal_dry = prefactor * (1.0 + (eps_vi-1.0) * qt_dry)
            d_alpha_qt_dry = prefactor * th_dry * (eps_vi-1.0)

            if self.EnvVar.CF.values[k] > 0.0:
                d_alpha_thetal_cloudy = (prefactor * (1.0 + eps_vi * (1.0 + lh / Rv / t_cloudy) * qv_cloudy - qt_cloudy )
                                         / (1.0 + lh * lh / cpm / Rv / t_cloudy / t_cloudy * qv_cloudy))
                d_alpha_qt_cloudy = (lh / cpm / t_cloudy * d_alpha_thetal_cloudy - prefactor) * th_cloudy
            else:
                d_alpha_thetal_cloudy = 0.0
                d_alpha_qt_cloudy = 0.0

            d_alpha_thetal_total = (self.EnvVar.CF.values[k] * d_alpha_thetal_cloudy
                                    + (1.0-self.EnvVar.CF.values[k]) * d_alpha_thetal_dry)
            d_alpha_qt_total = (self.EnvVar.CF.values[k] * d_alpha_qt_cloudy
                                + (1.0-self.EnvVar.CF.values[k]) * d_alpha_qt_dry)

            # TODO - check
            self.EnvVar.TKE.buoy[k] = g / tmp['α_0', k] * ae[k] * tmp['ρ_0', k] \
                               * ( \
                                   - self.KH.values[k] * interp2pt(grad_thl_plus, grad_thl_minus) * d_alpha_thetal_total \
                                   - self.KH.values[k] * interp2pt(grad_qt_plus,  grad_qt_minus)  * d_alpha_qt_total\
                                 )
        return

    def compute_tke_pressure(self, tmp):
        for k in self.grid.over_elems_real(Center()):
            self.EnvVar.TKE.press[k] = 0.0
            for i in range(self.n_updrafts):
                wu_half = interp2pt(self.UpdVar.W.values[i][k-1], self.UpdVar.W.values[i][k])
                we_half = interp2pt(self.EnvVar.W.values[k-1], self.EnvVar.W.values[k])
                press_buoy= (-1.0 * tmp['ρ_0', k] * self.UpdVar.Area.values[i][k]
                             * self.UpdVar.B.values[i][k] * self.pressure_buoy_coeff)
                press_drag = (-1.0 * tmp['ρ_0', k] * np.sqrt(self.UpdVar.Area.values[i][k])
                              * (self.pressure_drag_coeff/self.pressure_plume_spacing* (wu_half -we_half)*np.fabs(wu_half -we_half)))
                self.EnvVar.TKE.press[k] += (we_half - wu_half) * (press_buoy + press_drag)
        return


    def update_GMV_diagnostics(self, GMV, tmp):
        for k in self.grid.over_elems_real(Center()):
            GMV.QL.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.QL.bulkvalues[k] + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.QL.values[k])
            GMV.QR.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.QR.bulkvalues[k] + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.QR.values[k])
            GMV.T.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.T.bulkvalues[k] + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.T.values[k])
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.THL.values[k] = t_to_thetali_c(tmp['p_0', k], GMV.T.values[k], GMV.QT.values[k], GMV.QL.values[k], 0.0)
            GMV.B.values[k] = (self.UpdVar.Area.bulkvalues[k] * self.UpdVar.B.bulkvalues[k] + (1.0 - self.UpdVar.Area.bulkvalues[k]) * self.EnvVar.B.values[k])
        return


    def compute_covariance(self, GMV, Case, TS, tmp):

        #if TS.nstep > 0:
        if self.similarity_diffusivity: # otherwise, we computed mixing length when we computed
            self.compute_mixing_length(Case.Sur.obukhov_length)
        if self.calc_tke:
            self.compute_tke_buoy(GMV, tmp)
            self.compute_covariance_entr(self.EnvVar.TKE, self.UpdVar.W, self.UpdVar.W, self.EnvVar.W, self.EnvVar.W, tmp)
            self.compute_covariance_shear(GMV, self.EnvVar.TKE, self.UpdVar.W.values, self.UpdVar.W.values, self.EnvVar.W.values, self.EnvVar.W.values, tmp)
            self.compute_covariance_interdomain_src(self.UpdVar.Area,self.UpdVar.W,self.UpdVar.W,self.EnvVar.W, self.EnvVar.W, self.EnvVar.TKE)
            self.compute_tke_pressure(tmp)
        if self.calc_scalar_var:
            self.compute_covariance_entr(self.EnvVar.Hvar, self.UpdVar.H, self.UpdVar.H, self.EnvVar.H, self.EnvVar.H, tmp)
            self.compute_covariance_entr(self.EnvVar.QTvar, self.UpdVar.QT, self.UpdVar.QT, self.EnvVar.QT, self.EnvVar.QT, tmp)
            self.compute_covariance_entr(self.EnvVar.HQTcov, self.UpdVar.H, self.UpdVar.QT, self.EnvVar.H, self.EnvVar.QT, tmp)
            self.compute_covariance_shear(GMV, self.EnvVar.Hvar, self.UpdVar.H.values, self.UpdVar.H.values, self.EnvVar.H.values, self.EnvVar.H.values, tmp)
            self.compute_covariance_shear(GMV, self.EnvVar.QTvar, self.UpdVar.QT.values, self.UpdVar.QT.values, self.EnvVar.QT.values, self.EnvVar.QT.values, tmp)
            self.compute_covariance_shear(GMV, self.EnvVar.HQTcov, self.UpdVar.H.values, self.UpdVar.QT.values, self.EnvVar.H.values, self.EnvVar.QT.values, tmp)
            self.compute_covariance_interdomain_src(self.UpdVar.Area,self.UpdVar.H,self.UpdVar.H,self.EnvVar.H, self.EnvVar.H, self.EnvVar.Hvar)
            self.compute_covariance_interdomain_src(self.UpdVar.Area,self.UpdVar.QT,self.UpdVar.QT,self.EnvVar.QT, self.EnvVar.QT, self.EnvVar.QTvar)
            self.compute_covariance_interdomain_src(self.UpdVar.Area,self.UpdVar.H,self.UpdVar.QT,self.EnvVar.H, self.EnvVar.QT, self.EnvVar.HQTcov)
            self.compute_covariance_rain(TS, GMV, tmp) # need to update this one

        self.reset_surface_covariance(GMV, Case, tmp)
        if self.calc_tke:
            self.update_covariance_ED(GMV, Case,TS, GMV.W, GMV.W, GMV.TKE, self.EnvVar.TKE, self.EnvVar.W, self.EnvVar.W, self.UpdVar.W, self.UpdVar.W, tmp)
        if self.calc_scalar_var:
            self.update_covariance_ED(GMV, Case,TS, GMV.H, GMV.H, GMV.Hvar, self.EnvVar.Hvar, self.EnvVar.H, self.EnvVar.H, self.UpdVar.H, self.UpdVar.H, tmp)
            self.update_covariance_ED(GMV, Case,TS, GMV.QT,GMV.QT, GMV.QTvar, self.EnvVar.QTvar, self.EnvVar.QT, self.EnvVar.QT, self.UpdVar.QT, self.UpdVar.QT, tmp)
            self.update_covariance_ED(GMV, Case,TS, GMV.H, GMV.QT, GMV.HQTcov, self.EnvVar.HQTcov, self.EnvVar.H, self.EnvVar.QT, self.UpdVar.H, self.UpdVar.QT, tmp)
            self.cleanup_covariance(GMV)
        return


    def initialize_covariance(self, GMV, Case, tmp):

        ws= self.wstar
        us = Case.Sur.ustar
        zs = self.zi

        self.reset_surface_covariance(GMV, Case, tmp)

        if self.calc_tke:
            if ws > 0.0:
                for k in self.grid.over_elems(Center()):
                    z = self.grid.z_half[k]
                    temp = ws * 1.3 * np.cbrt((us*us*us)/(ws*ws*ws) + 0.6 * z/zs) * np.sqrt(np.fmax(1.0-z/zs,0.0))
                    GMV.TKE.values[k] = temp
        if self.calc_scalar_var:
            if ws > 0.0:
                for k in self.grid.over_elems(Center()):
                    z = self.grid.z_half[k]
                    # need to rethink of how to initilize the covarinace profiles - for nowmI took the TKE profile
                    temp = ws * 1.3 * np.cbrt((us*us*us)/(ws*ws*ws) + 0.6 * z/zs) * np.sqrt(np.fmax(1.0-z/zs,0.0))
                    GMV.Hvar.values[k]   = GMV.Hvar.values[self.grid.gw] * temp
                    GMV.QTvar.values[k]  = GMV.QTvar.values[self.grid.gw] * temp
                    GMV.HQTcov.values[k] = GMV.HQTcov.values[self.grid.gw] * temp
            self.reset_surface_covariance(GMV, Case, tmp)
            self.compute_mixing_length(Case.Sur.obukhov_length)

        return


    def cleanup_covariance(self, GMV):
        tmp_eps = 1e-18
        for k in self.grid.over_elems_real(Center()):
            if GMV.TKE.values[k] < tmp_eps:                     GMV.TKE.values[k] = 0.0
            if GMV.Hvar.values[k] < tmp_eps:                    GMV.Hvar.values[k] = 0.0
            if GMV.QTvar.values[k] < tmp_eps:                   GMV.QTvar.values[k] = 0.0
            if np.fabs(GMV.HQTcov.values[k]) < tmp_eps:         GMV.HQTcov.values[k] = 0.0
            if self.EnvVar.Hvar.values[k] < tmp_eps:            self.EnvVar.Hvar.values[k] = 0.0
            if self.EnvVar.TKE.values[k] < tmp_eps:             self.EnvVar.TKE.values[k] = 0.0
            if self.EnvVar.QTvar.values[k] < tmp_eps:           self.EnvVar.QTvar.values[k] = 0.0
            if np.fabs(self.EnvVar.HQTcov.values[k]) < tmp_eps: self.EnvVar.HQTcov.values[k] = 0.0


    def compute_covariance_shear(self, GMV, Covar, UpdVar1, UpdVar2, EnvVar1, EnvVar2, tmp):
        ae = Field.half(self.grid)
        for k in self.grid.over_elems(Center()):
            ae[k] = 1.0 - self.UpdVar.Area.bulkvalues[k]
        diff_var1 = 0.0
        diff_var2 = 0.0
        du = 0.0
        dv = 0.0
        tke_factor = 1.0
        du_high = 0.0
        dv_high = 0.0

        for k in self.grid.over_elems_real(Center()):
            if Covar.name == 'tke':
                du_low = du_high
                dv_low = dv_high
                du_high = (GMV.U.values[k+1] - GMV.U.values[k]) * self.grid.dzi
                dv_high = (GMV.V.values[k+1] - GMV.V.values[k]) * self.grid.dzi
                diff_var2 = (EnvVar2[k] - EnvVar2[k-1]) * self.grid.dzi
                diff_var1 = (EnvVar1[k] - EnvVar1[k-1]) * self.grid.dzi
                tke_factor = 0.5
            else:
                du_low = 0.0
                dv_low = 0.0
                du_high = 0.0
                dv_high = 0.0
                diff_var2 = interp2pt((EnvVar2[k+1] - EnvVar2[k]),(EnvVar2[k] - EnvVar2[k-1])) * self.grid.dzi
                diff_var1 = interp2pt((EnvVar1[k+1] - EnvVar1[k]),(EnvVar1[k] - EnvVar1[k-1])) * self.grid.dzi
                tke_factor = 1.0
            Covar.shear[k] = tke_factor*2.0*(tmp['ρ_0', k] * ae[k] * self.KH.values[k] *
                        (diff_var1*diff_var2 +  pow(interp2pt(du_low, du_high),2.0)  +  pow(interp2pt(dv_low, dv_high),2.0)))
        return

    def compute_covariance_interdomain_src(self, au,
                        phi_u, psi_u,
                        phi_e,  psi_e,
                        Covar):

        for k in self.grid.over_elems(Center()):
            Covar.interdomain[k] = 0.0
            for i in range(self.n_updrafts):
                if Covar.name == 'tke':
                    tke_factor = 0.5
                    phi_diff = interp2pt(phi_u.values[i][k-1], phi_u.values[i][k])-interp2pt(phi_e.values[k-1], phi_e.values[k])
                    psi_diff = interp2pt(psi_u.values[i][k-1], psi_u.values[i][k])-interp2pt(psi_e.values[k-1], psi_e.values[k])
                else:
                    tke_factor = 1.0
                    phi_diff = phi_u.values[i][k]-phi_e.values[k]
                    psi_diff = psi_u.values[i][k]-psi_e.values[k]

                Covar.interdomain[k] += tke_factor*au.values[i][k] * (1.0-au.values[i][k]) * phi_diff * psi_diff
        return

    def compute_covariance_entr(self, Covar, UpdVar1, UpdVar2, EnvVar1, EnvVar2, tmp):

        for k in self.grid.over_elems_real(Center()):
            Covar.entr_gain[k] = 0.0
            for i in range(self.n_updrafts):
                if Covar.name =='tke':
                    updvar1 = interp2pt(UpdVar1.values[i][k], UpdVar1.values[i][k-1])
                    updvar2 = interp2pt(UpdVar2.values[i][k], UpdVar2.values[i][k-1])
                    envvar1 = interp2pt(EnvVar1.values[k], EnvVar1.values[k-1])
                    envvar2 = interp2pt(EnvVar2.values[k], EnvVar2.values[k-1])
                    tke_factor = 0.5
                else:
                    updvar1 = UpdVar1.values[i][k]
                    updvar2 = UpdVar2.values[i][k]
                    envvar1 = EnvVar1.values[k]
                    envvar2 = EnvVar2.values[k]
                    tke_factor = 1.0
                w_u = interp2pt(self.UpdVar.W.values[i][k-1], self.UpdVar.W.values[i][k])
                Covar.entr_gain[k] +=  tke_factor*self.UpdVar.Area.values[i][k] * np.fabs(w_u) * self.detr_sc[i][k] * \
                                             (updvar1 - envvar1) * (updvar2 - envvar2)
            Covar.entr_gain[k] *= tmp['ρ_0', k]
        return

    def compute_covariance_detr(self, Covar, tmp):

        for k in self.grid.over_elems_real(Center()):
            Covar.detr_loss[k] = 0.0
            for i in range(self.n_updrafts):
                w_u = interp2pt(self.UpdVar.W.values[i][k-1], self.UpdVar.W.values[i][k])
                Covar.detr_loss[k] += self.UpdVar.Area.values[i][k] * np.fabs(w_u) * self.entr_sc[i][k]
            Covar.detr_loss[k] *= tmp['ρ_0', k] * Covar.values[k]
        return

    def compute_covariance_rain(self, TS, GMV, tmp):
        # TODO defined again in compute_covariance_shear and compute_covaraince
        ae = Field.half(self.grid)
        for k in self.grid.over_elems(Center()):
            ae[k] = 1.0 - self.UpdVar.Area.bulkvalues[k]

        for k in self.grid.over_elems_real(Center()):
            self.EnvVar.TKE.rain_src[k] = 0.0
            self.EnvVar.Hvar.rain_src[k]   = tmp['ρ_0', k] * ae[k] * 2. * self.EnvThermo.Hvar_rain_dt[k]   * TS.dti
            self.EnvVar.QTvar.rain_src[k]  = tmp['ρ_0', k] * ae[k] * 2. * self.EnvThermo.QTvar_rain_dt[k]  * TS.dti
            self.EnvVar.HQTcov.rain_src[k] = tmp['ρ_0', k] * ae[k] *      self.EnvThermo.HQTcov_rain_dt[k] * TS.dti
        return


    def compute_covariance_dissipation(self, Covar, tmp):
        ae = Field.half(self.grid)
        for k in self.grid.over_elems(Center()):
            ae[k] = 1.0 - self.UpdVar.Area.bulkvalues[k]

        for k in self.grid.over_elems_real(Center()):
            Covar.dissipation[k] = (tmp['ρ_0', k] * ae[k] * Covar.values[k]
                                *pow(np.fmax(self.EnvVar.TKE.values[k],0), 0.5)/np.fmax(self.mixing_length[k],1.0) * self.tke_diss_coeff)
        return



    def update_covariance_ED(self, GMV, Case,TS, GmvVar1, GmvVar2, GmvCovar, Covar,  EnvVar1,  EnvVar2, UpdVar1,  UpdVar2, tmp):
        gw = self.grid.gw
        nzg = self.grid.nzg
        nz = self.grid.nz
        dzi = self.grid.dzi
        dti = TS.dti
        alpha0LL  = self.Ref.alpha0_half[self.grid.gw]
        zLL = self.grid.z_half[self.grid.gw]
        a = np.zeros((nz,),dtype=np.double, order='c')
        b = np.zeros((nz,),dtype=np.double, order='c')
        c = np.zeros((nz,),dtype=np.double, order='c')
        x = np.zeros((nz,),dtype=np.double, order='c')
        ae = Field.half(self.grid)
        for k in self.grid.over_elems(Center()):
            ae[k] = 1.0 - self.UpdVar.Area.bulkvalues[k]
        ae_old = np.subtract(np.ones((nzg,),dtype=np.double, order='c'), np.sum(self.UpdVar.Area.old,axis=0))
        rho_ae_K_m = np.zeros((nzg,),dtype=np.double, order='c')
        whalf = np.zeros((nzg,),dtype=np.double, order='c')
        D_env = 0.0

        for k in range(1,nzg-1):
            rho_ae_K_m[k] = 0.5 * (ae[k]+ae[k+1]) * 0.5 * (self.KH.values[k]+self.KH.values[k+1]) * self.Ref.rho0[k]
            whalf[k] = interp2pt(self.EnvVar.W.values[k-1], self.EnvVar.W.values[k])
        wu_half = interp2pt(self.UpdVar.W.bulkvalues[gw-1], self.UpdVar.W.bulkvalues[gw])

        if GmvCovar.name=='tke':
            GmvCovar.values[gw] =get_surface_tke(Case.Sur.ustar, self.wstar, self.grid.z_half[gw], Case.Sur.obukhov_length)

        elif GmvCovar.name=='thetal_var':
            GmvCovar.values[gw] = get_surface_variance(Case.Sur.rho_hflux * alpha0LL, Case.Sur.rho_hflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)
        elif GmvCovar.name=='qt_var':
            GmvCovar.values[gw] = get_surface_variance(Case.Sur.rho_qtflux * alpha0LL, Case.Sur.rho_qtflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)
        elif GmvCovar.name=='thetal_qt_covar':
            GmvCovar.values[gw] = get_surface_variance(Case.Sur.rho_hflux * alpha0LL, Case.Sur.rho_qtflux * alpha0LL, Case.Sur.ustar, zLL, Case.Sur.obukhov_length)

        self.get_env_covar_from_GMV(self.UpdVar.Area, UpdVar1, UpdVar2, EnvVar1, EnvVar2, Covar, GmvVar1.values, GmvVar2.values, GmvCovar.values)

        Covar_surf = Covar.values[gw]

        for kk in range(nz):
            k = kk+gw
            D_env = 0.0

            for i in range(self.n_updrafts):
                wu_half = interp2pt(self.UpdVar.W.values[i][k-1], self.UpdVar.W.values[i][k])
                D_env += tmp['ρ_0', k] * self.UpdVar.Area.values[i][k] * wu_half * self.entr_sc[i][k]


            a[kk] = (- rho_ae_K_m[k-1] * dzi * dzi )
            b[kk] = (tmp['ρ_0', k] * ae[k] * dti - tmp['ρ_0', k] * ae[k] * whalf[k] * dzi
                     + rho_ae_K_m[k] * dzi * dzi + rho_ae_K_m[k-1] * dzi * dzi
                     + D_env
                     + tmp['ρ_0', k] * ae[k] * self.tke_diss_coeff
                                *np.sqrt(np.fmax(self.EnvVar.TKE.values[k],0))/np.fmax(self.mixing_length[k],1.0) )
            c[kk] = (tmp['ρ_0', k+1] * ae[k+1] * whalf[k+1] * dzi - rho_ae_K_m[k] * dzi * dzi)
            x[kk] = (tmp['ρ_0', k] * ae_old[k] * Covar.values[k] * dti
                     + Covar.press[k] + Covar.buoy[k] + Covar.shear[k] + Covar.entr_gain[k] +  Covar.rain_src[k]) #

            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0
            x[0] = Covar_surf

            b[nz-1] += c[nz-1]
            c[nz-1] = 0.0
        tridiag_solve(self.grid.nz, x, a, b, c)

        for kk in range(nz):
            k = kk + gw
            if Covar.name == 'thetal_qt_covar':
                Covar.values[k] = np.fmax(x[kk], - np.sqrt(self.EnvVar.Hvar.values[k]*self.EnvVar.QTvar.values[k]))
                Covar.values[k] = np.fmin(x[kk],   np.sqrt(self.EnvVar.Hvar.values[k]*self.EnvVar.QTvar.values[k]))
            else:
                Covar.values[k] = np.fmax(x[kk],0.0)

        self.get_GMV_CoVar(self.UpdVar.Area, UpdVar1, UpdVar2, EnvVar1, EnvVar2, Covar, GmvVar1.values, GmvVar2.values, GmvCovar.values)

        return
