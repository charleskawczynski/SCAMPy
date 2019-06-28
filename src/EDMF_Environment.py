import numpy as np
import sys
from parameters import *
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
from TimeStepping import TimeStepping
from ReferenceState import ReferenceState
from Variables import VariableDiagnostic, GridMeanVariables
from thermodynamic_functions import  *
from microphysics_functions import *

class EnvironmentVariable:
    def __init__(self, Gr, loc, bc, name, units):
        self.values = Field.field(Gr, loc)
        self.bc = bc
        self.name = name
        self.units = units

    def set_bcs(self, Gr):
        n_updrafts = np.shape(self.values)[0]
        for i in range(n_updrafts):
            self.values[i].apply_bc(Gr, self.bc, 0.0)
        return

class EnvironmentVariable_2m:
    def __init__(self, Gr, loc, bc, name, units):
        self.values      = Field.field(Gr, loc)
        self.dissipation = Field.field(Gr, loc)
        self.entr_gain   = Field.field(Gr, loc)
        self.detr_loss   = Field.field(Gr, loc)
        self.buoy        = Field.field(Gr, loc)
        self.press       = Field.field(Gr, loc)
        self.shear       = Field.field(Gr, loc)
        self.interdomain = Field.field(Gr, loc)
        self.rain_src    = Field.field(Gr, loc)
        self.bc = bc
        self.name = name
        self.units = units

    def set_bcs(self, Gr):
        n_updrafts = np.shape(self.values)[0]
        for i in range(n_updrafts):
            self.values[i].apply_bc(Gr, self.bc, 0.0)
        return

class EnvironmentVariables:
    def __init__(self,  namelist, Gr  ):
        self.grid = Gr

        self.W   = EnvironmentVariable(Gr, Node(), Dirichlet(), 'w','m/s' )
        self.QT  = EnvironmentVariable(Gr, Center(), Neumann(), 'qt','kg/kg' )
        self.QL  = EnvironmentVariable(Gr, Center(), Neumann(), 'ql','kg/kg' )
        self.QR  = EnvironmentVariable(Gr, Center(), Neumann(), 'qr','kg/kg' )
        self.THL = EnvironmentVariable(Gr, Center(), Neumann(), 'thetal', 'K')
        self.T   = EnvironmentVariable(Gr, Center(), Neumann(), 'temperature','K' )
        self.B   = EnvironmentVariable(Gr, Center(), Neumann(), 'buoyancy','m^2/s^3' )
        self.CF  = EnvironmentVariable(Gr, Center(), Neumann(),'cloud_fraction', '-')
        self.H = EnvironmentVariable(Gr, Center(), Neumann(), 'thetal','K' )

        try:
            self.EnvThermo_scheme = str(namelist['thermodynamics']['saturation'])
        except:
            self.EnvThermo_scheme = 'sa_mean'
            print('Defaulting to saturation adjustment with respect to environmental means')

        self.TKE = EnvironmentVariable_2m(Gr, Center(), Neumann(), 'tke','m^2/s^2' )

        self.QTvar = EnvironmentVariable_2m(Gr, Center(), Neumann(), 'qt_var','kg^2/kg^2' )
        self.Hvar = EnvironmentVariable_2m(Gr, Center(), Neumann(), 'thetal_var', 'K^2')
        self.HQTcov = EnvironmentVariable_2m(Gr, Center(), Neumann(), 'thetal_qt_covar', 'K(kg/kg)' )

        if self.EnvThermo_scheme == 'sommeria_deardorff':
            self.THVvar = EnvironmentVariable(Gr, Center(), Neumann(), 'thetav_var', 'K^2' )

        #TODO  - most likely a temporary solution (unless it could be useful for testing)
        try:
            self.use_prescribed_scalar_var = namelist['turbulence']['sgs']['use_prescribed_scalar_var']
        except:
            self.use_prescribed_scalar_var = False
        if self.use_prescribed_scalar_var:
            self.prescribed_QTvar  = namelist['turbulence']['sgs']['prescribed_QTvar']
            self.prescribed_Hvar   = namelist['turbulence']['sgs']['prescribed_Hvar']
            self.prescribed_HQTcov = namelist['turbulence']['sgs']['prescribed_HQTcov']

        return

    def initialize_io(self, Stats):
        Stats.add_profile('env_w')
        Stats.add_profile('env_qt')
        Stats.add_profile('env_ql')
        Stats.add_profile('env_qr')
        Stats.add_profile('env_thetal')
        Stats.add_profile('env_temperature')
        Stats.add_profile('env_tke')
        Stats.add_profile('env_Hvar')
        Stats.add_profile('env_QTvar')
        Stats.add_profile('env_HQTcov')
        if self.EnvThermo_scheme == 'sommeria_deardorff':
            Stats.add_profile('env_THVvar')
        return

    def io(self, Stats):
        Stats.write_profile_new('env_w'          , self.grid, self.W.values)
        Stats.write_profile_new('env_qt'         , self.grid, self.QT.values)
        Stats.write_profile_new('env_ql'         , self.grid, self.QL.values)
        Stats.write_profile_new('env_qr'         , self.grid, self.QR.values)
        Stats.write_profile_new('env_thetal' , self.grid, self.H.values)

        Stats.write_profile_new('env_temperature', self.grid, self.T.values)
        Stats.write_profile_new('env_tke'    , self.grid, self.TKE.values)
        Stats.write_profile_new('env_Hvar'   , self.grid, self.Hvar.values)
        Stats.write_profile_new('env_QTvar'  , self.grid, self.QTvar.values)
        Stats.write_profile_new('env_HQTcov' , self.grid, self.HQTcov.values)
        if self.EnvThermo_scheme  == 'sommeria_deardorff':
            Stats.write_profile_new('env_THVvar' , self.grid, self.THVvar.values)

        #ToDo [suggested by CK for AJ ;]
        # Add output of environmental cloud fraction, cloud base, cloud top (while the latter can be gleaned from ql profiles
        # it is more convenient to simply have them in the stats files!
        # Add the same with respect to the grid mean
        return

class EnvironmentThermodynamics:
    def __init__(self, namelist, paramlist, Gr, Ref, EnvVar):
        self.grid = Gr
        self.Ref = Ref
        try:
            self.quadrature_order = namelist['condensation']['quadrature_order']
        except:
            self.quadrature_order = 5

        self.t_to_prog_fp = t_to_thetali_c
        self.prog_to_t_fp = eos_first_guess_thetal

        self.qt_dry         = Half(Gr)
        self.th_dry         = Half(Gr)

        self.t_cloudy       = Half(Gr)
        self.qv_cloudy      = Half(Gr)
        self.qt_cloudy      = Half(Gr)
        self.th_cloudy      = Half(Gr)

        self.Hvar_rain_dt   = Half(Gr)
        self.QTvar_rain_dt  = Half(Gr)
        self.HQTcov_rain_dt = Half(Gr)

        self.max_supersaturation = paramlist['turbulence']['updraft_microphysics']['max_supersaturation']

        return

    def update_EnvVar(self, tmp, k, EnvVar, T, H, qt, ql, qr, alpha):

        EnvVar.T.values[k]   = T
        EnvVar.THL.values[k] = H
        EnvVar.H.values[k]   = H
        EnvVar.QT.values[k]  = qt
        EnvVar.QL.values[k]  = ql
        EnvVar.QR.values[k] += qr
        EnvVar.B.values[k]   = buoyancy_c(tmp['α_0_half'][k], alpha)
        return

    def update_cloud_dry(self, k, EnvVar, T, th, qt, ql, qv):

        if ql > 0.0:
            EnvVar.CF.values[k] = 1.
            self.th_cloudy[k]   = th
            self.t_cloudy[k]    = T
            self.qt_cloudy[k]   = qt
            self.qv_cloudy[k]   = qv
        else:
            EnvVar.CF.values[k] = 0.
            self.th_dry[k]      = th
            self.qt_dry[k]      = qt
        return

    def eos_update_SA_mean(self, EnvVar, in_Env, tmp):

        for k in self.grid.over_elems_real(Center()):
            # condensation + autoconversion
            T, ql  = eos(self.t_to_prog_fp, self.prog_to_t_fp, tmp['p_0_half'][k], EnvVar.QT.values[k], EnvVar.H.values[k])
            mph = microphysics(T, ql, tmp['p_0_half'][k], EnvVar.QT.values[k], self.max_supersaturation, in_Env)

            self.update_EnvVar(tmp,   k, EnvVar, mph.T, mph.thl, mph.qt, mph.ql, mph.qr, mph.alpha)
            self.update_cloud_dry(k, EnvVar, mph.T, mph.th,  mph.qt, mph.ql, mph.qv)
        return

    def eos_update_SA_sgs(self, EnvVar, in_Env, tmp):
        a, w = np.polynomial.hermite.hermgauss(self.quadrature_order)

        #TODO - remember you output source terms multipierd by dt (bec. of instanteneous autoconcv)
        #TODO - read prescribed var/covar from file to compare with LES data
        #TODO - add tendencies for GMV H, QT and QR due to rain

        abscissas = a
        weights = w
        # arrays for storing quadarature points and ints for labeling items in the arrays
        # a python dict would be nicer, but its 30% slower than this (for python 2.7. It might not be the case for python 3)
        env_len = 10
        src_len = 6

        sqpi_inv = 1.0/np.sqrt(pi)
        sqrt2 = np.sqrt(2.0)

        # for testing (to be removed)
        if EnvVar.use_prescribed_scalar_var:
            for k in self.grid.over_elems_real(Center()):
                if k * self.grid.dz <= 1500:
                    EnvVar.QTvar.values[k]  = EnvVar.prescribed_QTvar
                else:
                    EnvVar.QTvar.values[k]  = 0.
                if k * self.grid.dz <= 1500 and k * self.grid.dz > 500:
                    EnvVar.Hvar.values[k]   = EnvVar.prescribed_Hvar
                else:
                    EnvVar.Hvar.values[k]   = 0.
                if k * self.grid.dz <= 1500 and k * self.grid.dz > 200:
                    EnvVar.HQTcov.values[k] = EnvVar.prescribed_HQTcov
                else:
                    EnvVar.HQTcov.values[k] = 0.

        # initialize the quadrature points and their labels
        inner_env = np.zeros(env_len, dtype=np.double, order='c')
        outer_env = np.zeros(env_len, dtype=np.double, order='c')
        inner_src = np.zeros(src_len, dtype=np.double, order='c')
        outer_src = np.zeros(src_len, dtype=np.double, order='c')
        i_ql, i_T, i_thl, i_alpha, i_cf, i_qr, i_qt_cld, i_qt_dry, i_T_cld, i_T_dry = range(env_len)
        i_SH_qt, i_Sqt_H, i_SH_H, i_Sqt_qt, i_Sqt, i_SH = range(src_len)

        for k in self.grid.over_elems_real(Center()):
            if EnvVar.QTvar.values[k] != 0.0 and EnvVar.Hvar.values[k] != 0.0 and EnvVar.HQTcov.values[k] != 0.0:
                sd_q = np.sqrt(EnvVar.QTvar.values[k])
                sd_h = np.sqrt(EnvVar.Hvar.values[k])
                corr = np.fmax(np.fmin(EnvVar.HQTcov.values[k]/np.fmax(sd_h*sd_q, 1e-13),1.0),-1.0)

                # limit sd_q to prevent negative qt_hat
                sd_q_lim = (1e-10 - EnvVar.QT.values[k])/(sqrt2 * abscissas[0])
                sd_q = np.fmin(sd_q, sd_q_lim)
                qt_var = sd_q * sd_q
                sigma_h_star = np.sqrt(np.fmax(1.0-corr*corr,0.0)) * sd_h

                # zero outer quadrature points
                for idx in range(env_len):
                    outer_env[idx] = 0.0
                if in_Env:
                    for idx in range(src_len):
                        outer_src[idx] = 0.0

                for m_q in range(self.quadrature_order):
                    qt_hat    = EnvVar.QT.values[k] + sqrt2 * sd_q * abscissas[m_q]
                    mu_h_star = EnvVar.H.values[k]  + sqrt2 * corr * sd_h * abscissas[m_q]

                    # zero inner quadrature points
                    for idx in range(env_len):
                        inner_env[idx] = 0.0
                    if in_Env:
                        for idx in range(src_len):
                            inner_src[idx] = 0.0

                    for m_h in range(self.quadrature_order):
                        h_hat = sqrt2 * sigma_h_star * abscissas[m_h] + mu_h_star

                        # condensation + autoconversion
                        T, q_l  = eos(self.t_to_prog_fp, self.prog_to_t_fp, tmp['p_0_half'][k], qt_hat, h_hat)
                        mph = microphysics(T, ql, tmp['p_0_half'][k], qt_hat, self.max_supersaturation, in_Env)

                        # environmental variables
                        inner_env[i_ql]    += mph.ql    * weights[m_h] * sqpi_inv
                        inner_env[i_qr]    += mph.qr    * weights[m_h] * sqpi_inv
                        inner_env[i_T]     += mph.T     * weights[m_h] * sqpi_inv
                        inner_env[i_thl]   += mph.thl   * weights[m_h] * sqpi_inv
                        inner_env[i_alpha] += mph.alpha * weights[m_h] * sqpi_inv
                        # cloudy/dry categories for buoyancy in TKE
                        if mph.ql  > 0.0:
                            inner_env[i_cf]     +=          weights[m_h] * sqpi_inv
                            inner_env[i_qt_cld] += mph.qt * weights[m_h] * sqpi_inv
                            inner_env[i_T_cld]  += mph.T  * weights[m_h] * sqpi_inv
                        else:
                            inner_env[i_qt_dry] += mph.qt * weights[m_h] * sqpi_inv
                            inner_env[i_T_dry]  += mph.T  * weights[m_h] * sqpi_inv
                        # products for variance and covariance source terms
                        if in_Env:
                            inner_src[i_Sqt]    += -mph.qr                     * weights[m_h] * sqpi_inv
                            inner_src[i_SH]     +=  mph.thl_rain_src           * weights[m_h] * sqpi_inv
                            inner_src[i_Sqt_H]  += -mph.qr           * mph.thl * weights[m_h] * sqpi_inv
                            inner_src[i_Sqt_qt] += -mph.qr           * mph.qt  * weights[m_h] * sqpi_inv
                            inner_src[i_SH_H]   +=  mph.thl_rain_src * mph.thl * weights[m_h] * sqpi_inv
                            inner_src[i_SH_qt]  +=  mph.thl_rain_src * mph.qt  * weights[m_h] * sqpi_inv

                    for idx in range(env_len):
                        outer_env[idx] += inner_env[idx] * weights[m_q] * sqpi_inv
                    if in_Env:
                        for idx in range(src_len):
                            outer_src[idx] += inner_src[idx] * weights[m_q] * sqpi_inv

                # update environmental variables
                self.update_EnvVar(tmp,k, EnvVar, outer_env[i_T], outer_env[i_thl],\
                                   outer_env[i_qt_cld]+outer_env[i_qt_dry], outer_env[i_ql],\
                                   outer_env[i_qr], outer_env[i_alpha])
                # update cloudy/dry variables for buoyancy in TKE
                EnvVar.CF.values[k]  = outer_env[i_cf]
                self.qt_dry[k]    = outer_env[i_qt_dry]
                self.th_dry[k]    = theta_c(tmp['p_0_half'][k], outer_env[i_T_dry])
                self.t_cloudy[k]  = outer_env[i_T_cld]
                self.qv_cloudy[k] = outer_env[i_qt_cld] - outer_env[i_ql]
                self.qt_cloudy[k] = outer_env[i_qt_cld]
                self.th_cloudy[k] = theta_c(tmp['p_0_half'][k], outer_env[i_T_cld])
                # update var/covar rain sources
                if in_Env:
                    self.Hvar_rain_dt[k]   = outer_src[i_SH_H]   - outer_src[i_SH]  * EnvVar.H.values[k]
                    self.QTvar_rain_dt[k]  = outer_src[i_Sqt_qt] - outer_src[i_Sqt] * EnvVar.QT.values[k]
                    self.HQTcov_rain_dt[k] = outer_src[i_SH_qt]  - outer_src[i_SH]  * EnvVar.QT.values[k] + \
                                             outer_src[i_Sqt_H]  - outer_src[i_Sqt] * EnvVar.H.values[k]

            else:
                # the same as in SA_mean
                T, ql  = eos(self.t_to_prog_fp, self.prog_to_t_fp, tmp['p_0_half'][k], EnvVar.QT.values[k], EnvVar.H.values[k])
                mph = microphysics(T, ql, tmp['p_0_half'][k], EnvVar.QT.values[k], self.max_supersaturation, in_Env)

                self.update_EnvVar(tmp,   k, EnvVar, mph.T, mph.thl, mph.qt, mph.ql, mph.qr, mph.alpha)
                self.update_cloud_dry(k, EnvVar, mph.T, mph.th,  mph.qt, mph.ql, mph.qv)

                if in_Env:
                    self.Hvar_rain_dt[k]   = 0.
                    self.QTvar_rain_dt[k]  = 0.
                    self.HQTcov_rain_dt[k] = 0.

        return

    def sommeria_deardorff(self, EnvVar, tmp):
        # this function follows the derivation in
        # Sommeria and Deardorff 1977: Sub grid scale condensation in models of non-precipitating clouds.
        # J. Atmos. Sci., 34, 344-355.

        for k in self.grid.over_elems_real(Center()):
            Lv = latent_heat(EnvVar.T.values[k])
            cp = cpd
            # paper notation used below
            Tl = EnvVar.H.values[k]*exner_c(tmp['p_0_half'][k])
            q_sl = qv_star_t(self.Ref.p0[k], Tl) # using the qv_star_c function instead of the approximation in eq. (4) in SD
            beta1 = 0.622*Lv**2/(Rd*cp*Tl**2) # eq. (8) in SD
            #q_s = q_sl*(1+beta1*EnvVar.QT.values[k])/(1+beta1*q_sl) # eq. (7) in SD
            lambda1 = 1/(1+beta1*q_sl) # text under eq. (20) in SD
            # check the pressure units - mb vs pa
            alpha1 = (self.Ref.p0[k]/100000.0)**0.286*0.622*Lv*q_sl/Rd/Tl**2 # eq. (14) and eq. (6) in SD
            # see if there is another way to calculate dq/dT from scmapy
            sigma1 = EnvVar.QTvar.values[k]-2*alpha1*EnvVar.HQTcov.values[k]+alpha1**2*EnvVar.Hvar.values[k] # eq. (18) in SD , with r from (11)
            Q1 = (EnvVar.QT.values[k]-q_sl)/sigma1 # eq. (17) in SD
            R = 0.5*(1+np.erf(Q1/np.sqrt(2.0))) # approximation in eq. (16) in SD
            #R1 = 0.5*(1+Q1/1.6) # approximation in eq. (22) in SD
            C0 = 1.0+0.61*q_sl-alpha1*lambda1*EnvVar.THL.values[k]*(Lv/cp/Tl*(1.0+0.61*q_sl)-1.61) # eq. (37) in SD
            C1 = (1.0-R)*(1+0.61*q_sl)+R*C0 # eq. (42a) in SD
            C2 = (1.0-R)*0.61+R*(C0*Lv/cp/Tl-1.0) # eq. (42b) in SD
            C2_THL = C2*EnvVar.THL.values[k] # defacto the coefficient in eq(41) is C2*THL
            # the THVvar is given as a function of THVTHLcov and THVQTcov from eq. (41) in SD.
            # these covariances with THL are obtained by substituting w for THL or QT in eq. (41),
            # i.e. applying eq. (41) twice. The resulting expression yields: C1^2*THL_var+2*C1*C2*THL_var*QT_var+C2^2**QT_var
            EnvVar.THVvar.values[k] = C1**2*EnvVar.Hvar.values[k] + 2*C1*C2_THL*EnvVar.HQTcov.values[k]+ C2_THL**2*EnvVar.QTvar.values[k]
            # equation (19) exact form for QL
            EnvVar.QL.values[k] = 1.0/(1.0+beta1*q_sl)*(R*(EnvVar.QT.values[k]-q_sl)+sigma1/np.sqrt(6.14)*exp(-((EnvVar.QT.values[k]-q_sl)*(EnvVar.QT.values[k]-q_sl)/(2.0*sigma1*sigma1))))
            EnvVar.T.values[k] = Tl + Lv/cp*EnvVar.QL.values[k] # should this be the differnece in ql - would it work for evaporation as well ?
            EnvVar.CF.values[k] = R
            qv = EnvVar.QT.values[k] - EnvVar.QL.values[k]
            alpha = alpha_c(tmp['p_0_half'][k], EnvVar.T.values[k], EnvVar.QT.values[k], qv)
            EnvVar.B.values[k] = buoyancy_c(tmp['α_0_half'][k], alpha)
            EnvVar.THL.values[k] = t_to_thetali_c(tmp['p_0_half'][k], EnvVar.T.values[k], EnvVar.QT.values[k],
                                                  EnvVar.QL.values[k], 0.0)

            self.qt_dry[k] = EnvVar.QT.values[k]
            self.th_dry[k] = EnvVar.T.values[k]/exner_c(tmp['p_0_half'][k])
            self.t_cloudy[k] = EnvVar.T.values[k]
            self.qv_cloudy[k] = EnvVar.QT.values[k] - EnvVar.QL.values[k]
            self.qt_cloudy[k] = EnvVar.QT.values[k]
            self.th_cloudy[k] = EnvVar.T.values[k]/exner_c(tmp['p_0_half'][k])

        return

    def satadjust(self, EnvVar, in_Env, tmp):

        if EnvVar.EnvThermo_scheme == 'sa_mean':
            self.eos_update_SA_mean(EnvVar, in_Env, tmp)
        elif EnvVar.EnvThermo_scheme == 'sa_quadrature':
            self.eos_update_SA_sgs(EnvVar, in_Env, tmp)#, TS)
        elif EnvVar.EnvThermo_scheme == 'sommeria_deardorff':
            self.sommeria_deardorff(EnvVar, tmp)
        else:
            sys.exit('EDMF_Environment: Unrecognized EnvThermo_scheme. Possible options: sa_mean, sa_quadrature, sommeria_deardorff')

        return
