import numpy as np
from parameters import *
from Grid import Grid, Zmin, Zmax, Center, Node
from thermodynamic_functions import *
from surface_functions import entropy_flux, compute_ustar, buoyancy_flux, exchange_coefficients_byun
from turbulence_functions import get_wstar, get_inversion
from Variables import GridMeanVariables
from Field import Field, Full, Half, Dirichlet, Neumann

class SurfaceBase:
    def __init__(self, paramlist):
        self.ustar = None
        self.lhf = None
        self.shf = None
        self.Ri_bulk_crit = paramlist['turbulence']['Ri_bulk_crit']
        return
    def initialize(self):
        return

    def update(self, GMV):
        return
    def free_convection_windspeed(self, GMV, tmp):
        theta_rho = Half(self.grid)

        # Need to get theta_rho
        for k in self.grid.over_elems(Center()):
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            theta_rho[k] = theta_rho_c(tmp['p_0', k], GMV.T.values[k], GMV.QT.values[k], qv)
        zi = get_inversion(theta_rho, GMV.U.values, GMV.V.values, self.grid, self.Ri_bulk_crit)
        wstar = get_wstar(self.bflux, zi) # yair here zi in TRMM should be adjusted
        self.windspeed = np.sqrt(self.windspeed*self.windspeed  + (1.2 *wstar)*(1.2 * wstar) )
        return


class SurfaceFixedFlux(SurfaceBase):
    def __init__(self,paramlist):
        SurfaceBase.__init__(self, paramlist)
        return
    def initialize(self):
        return

    def update(self, GMV, tmp):
        gw = self.grid.gw
        rho_tflux =  self.shf /(cpm_c(self.qsurface))

        self.windspeed = np.sqrt(GMV.U.values[gw]*GMV.U.values[gw] + GMV.V.values[gw] * GMV.V.values[gw])
        self.rho_qtflux = self.lhf/(latent_heat(self.Tsurface))

        ρ_0_surf = tmp.surface(self.grid, 'ρ_0')
        α_0_surf = tmp.surface(self.grid, 'α_0')

        if GMV.H.name == 'thetal':
            self.rho_hflux = rho_tflux / exner_c(self.Ref.Pg)
        elif GMV.H.name == 's':
            self.rho_hflux = entropy_flux(rho_tflux/ρ_0_surf,self.rho_qtflux/ρ_0_surf,
                                          tmp['p_0', gw], GMV.T.values[gw], GMV.QT.values[gw])
        self.bflux = buoyancy_flux(self.shf, self.lhf, GMV.T.values[gw], GMV.QT.values[gw], α_0_surf)

        if not self.ustar_fixed:
            # Correction to windspeed for free convective cases (Beljaars, QJRMS (1994), 121, pp. 255-270)
            # Value 1.2 is empirical, but should be O(1)
            if self.windspeed < 0.1:  # Limit here is heuristic
                if self.bflux > 0.0:
                   self.free_convection_windspeed(GMV, tmp)
                else:
                    print('WARNING: Low windspeed + stable conditions, need to check ustar computation')
                    print('self.bflux ==>',self.bflux )
                    print('self.shf ==>',self.shf)
                    print('self.lhf ==>',self.lhf)
                    print('GMV.U.values[gw] ==>',GMV.U.values[gw])
                    print('GMV.v.values[gw] ==>',GMV.V.values[gw])
                    print('GMV.QT.values[gw] ==>',GMV.QT.values[gw])
                    print('α_0_surf ==>', α_0_surf)

            self.ustar = compute_ustar(self.windspeed, self.bflux, self.zrough, self.grid.z_half[gw])

        self.obukhov_length = -self.ustar *self.ustar *self.ustar /self.bflux /vkb
        self.rho_uflux = - ρ_0_surf *  self.ustar * self.ustar / self.windspeed * GMV.U.values[gw]
        self.rho_vflux = - ρ_0_surf *  self.ustar * self.ustar / self.windspeed * GMV.V.values[gw]
        return
    def free_convection_windspeed(self, GMV, tmp):
        SurfaceBase.free_convection_windspeed(self, GMV, tmp)
        return


# Cases such as Rico which provide values of transfer coefficients
class SurfaceFixedCoeffs(SurfaceBase):
    def __init__(self, paramlist):
        SurfaceBase.__init__(self, paramlist)
        return
    def initialize(self):
        pvg = pv_star(self.Tsurface)
        pdg = self.Ref.Pg - pvg
        self.qsurface = qv_star_t(self.Ref.Pg, self.Tsurface)
        self.s_surface = (1.0-self.qsurface) * sd_c(pdg, self.Tsurface) + self.qsurface * sv_c(pvg,self.Tsurface)
        return

    def update(self, GMV, tmp):
        gw = self.grid.gw
        windspeed = np.maximum(np.sqrt(GMV.U.values[gw]*GMV.U.values[gw] + GMV.V.values[gw] * GMV.V.values[gw]), 0.01)
        cp_ = cpm_c(GMV.QT.values[gw])
        lv = latent_heat(GMV.T.values[gw])
        ρ_0_surf = tmp.surface(self.grid, 'ρ_0')
        α_0_surf = tmp.surface(self.grid, 'α_0')

        self.rho_qtflux = -self.cq * windspeed * (GMV.QT.values[gw] - self.qsurface) * ρ_0_surf
        self.lhf = lv * self.rho_qtflux

        if GMV.H.name == 'thetal':
            self.rho_hflux = -self.ch * windspeed * (GMV.H.values[gw] - self.Tsurface/exner_c(self.Ref.Pg)) * ρ_0_surf
            self.shf = cp_  * self.rho_hflux
        elif GMV.H.name == 's':
            self.rho_hflux =  -self.ch * windspeed * (GMV.H.values[gw] - self.s_surface) * ρ_0_surf
            pv = pv_star(GMV.T.values[gw])
            pd = tmp['p_0', gw] - pv
            sv = sv_c(pv,GMV.T.values[gw])
            sd = sd_c(pd, GMV.T.values[gw])
            self.shf = (self.rho_hflux - self.lhf/lv * (sv-sd)) * GMV.T.values[gw]

        self.bflux = buoyancy_flux(self.shf, self.lhf, GMV.T.values[gw], GMV.QT.values[gw], α_0_surf)

        self.ustar =  np.sqrt(self.cm) * windspeed
        # CK--testing this--EDMF scheme checks greater or less than zero,
        if np.fabs(self.bflux) < 1e-10:
            self.obukhov_length = 0.0
        else:
            self.obukhov_length = -self.ustar *self.ustar *self.ustar /self.bflux /vkb

        self.rho_uflux = - ρ_0_surf *  self.ustar * self.ustar / windspeed * GMV.U.values[gw]
        self.rho_vflux = - ρ_0_surf *  self.ustar * self.ustar / windspeed * GMV.V.values[gw]
        return
    def free_convection_windspeed(self, GMV):
        SurfaceBase.free_convection_windspeed(self, GMV)
        return

class SurfaceMoninObukhov(SurfaceBase):
    def __init__(self, paramlist):
        SurfaceBase.__init__(self, paramlist)
        return
    def initialize(self):
        return
    def update(self, GMV, tmp):
        self.qsurface = qv_star_t(self.Ref.Pg, self.Tsurface)
        gw = self.grid.gw
        zb = self.grid.z_half[gw]
        theta_rho_g = theta_rho_c(self.Ref.Pg, self.Tsurface, self.qsurface, self.qsurface)
        theta_rho_b = theta_rho_c(tmp['p_0', gw], GMV.T.values[gw], self.qsurface, self.qsurface)
        lv = latent_heat(GMV.T.values[gw])

        if GMV.H.name == 'thetal':
            h_star = t_to_thetali_c(self.Ref.Pg, self.Tsurface, self.qsurface, 0.0, 0.0)
        elif GMV.H.name == 's':
            h_star = t_to_entropy_c(self.Ref.Pg, self.Tsurface, self.qsurface, 0.0, 0.0)


        self.windspeed = np.sqrt(GMV.U.values[gw]*GMV.U.values[gw] + GMV.V.values[gw] * GMV.V.values[gw])
        Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/zb
        Ri = Nb2 * zb * zb/(self.windspeed * self.windspeed)

        exchange_coefficients_byun(Ri, self.grid.z_half[gw], self.zrough, self.cm, self.ch, self.obukhov_length)

        ρ_0_surf = tmp.surface(self.grid, 'ρ_0')
        α_0_surf = tmp.surface(self.grid, 'α_0')

        self.rho_uflux = -self.cm * self.windspeed * (GMV.U.values[gw] ) * ρ_0_surf
        self.rho_vflux = -self.cm * self.windspeed * (GMV.V.values[gw] ) * ρ_0_surf

        self.rho_hflux =  -self.ch * self.windspeed * (GMV.H.values[gw] - h_star) * ρ_0_surf
        self.rho_qtflux = -self.ch * self.windspeed * (GMV.QT.values[gw] - self.qsurface) * ρ_0_surf
        self.lhf = lv * self.rho_qtflux

        if GMV.H.name == 'thetal':
            self.shf = cpm_c(GMV.QT.values[gw])  * self.rho_hflux

        elif GMV.H.name == 's':
            pv = pv_star(GMV.T.values[gw])
            pd = tmp['p_0', gw] - pv
            sv = sv_c(pv,GMV.T.values[gw])
            sd = sd_c(pd, GMV.T.values[gw])
            self.shf = (self.rho_hflux - self.lhf/lv * (sv-sd)) * GMV.T.values[gw]

        self.bflux = buoyancy_flux(self.shf, self.lhf, GMV.T.values[gw], GMV.QT.values[gw], α_0_surf)
        self.ustar =  sqrt(self.cm) * self.windspeed
        # CK--testing this--EDMF scheme checks greater or less than zero,
        if np.fabs(self.bflux) < 1e-10:
            self.obukhov_length = 0.0
        else:
            self.obukhov_length = -self.ustar *self.ustar *self.ustar /self.bflux /vkb

        return

    def free_convection_windspeed(self, GMV):
        SurfaceBase.free_convection_windspeed(self, GMV)
        return

# Not fully implemented yet. Maybe not needed - Ignacio
class SurfaceSullivanPatton(SurfaceBase):
    def __init__(self, paramlist):
        SurfaceBase.__init__(self, paramlist)
        return
    def initialize(self):
        return
    def update(self, GMV, tmp):
        gw = self.grid.gw
        zb = self.grid.z_half[gw]
        theta_rho_g = theta_rho_c(self.Ref.Pg, self.Tsurface, self.qsurface, self.qsurface)
        theta_rho_b = theta_rho_c(tmp['p_0', gw], GMV.T.values[gw], self.qsurface, self.qsurface)
        lv = latent_heat(GMV.T.values[gw])
        g=9.81
        T0 = tmp['p_0', gw] * tmp['α_0', gw]/Rd

        theta_flux = 0.24
        self.bflux = g * theta_flux * exner_c(tmp['p_0', gw]) / T0

        self.qsurface = qv_star_t(self.Ref.Pg, self.Tsurface)
        if GMV.H.name == 'thetal':
            h_star = t_to_thetali_c(self.Ref.Pg, self.Tsurface, self.qsurface, 0.0, 0.0)
        elif GMV.H.name == 's':
            h_star = t_to_entropy_c(self.Ref.Pg, self.Tsurface, self.qsurface, 0.0, 0.0)


        self.windspeed = np.sqrt(GMV.U.values[gw]*GMV.U.values[gw] + GMV.V.values[gw] * GMV.V.values[gw])
        Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/zb
        Ri = Nb2 * zb * zb/(self.windspeed * self.windspeed)

        exchange_coefficients_byun(Ri, self.grid.z_half[gw], self.zrough, self.cm, self.ch, self.obukhov_length)

        ρ_0_surf = tmp.surface(self.grid, 'ρ_0')

        self.rho_uflux = -self.cm * self.windspeed * (GMV.U.values[gw] ) * ρ_0_surf
        self.rho_vflux = -self.cm * self.windspeed * (GMV.V.values[gw] ) * ρ_0_surf

        self.rho_hflux =  -self.ch * self.windspeed * (GMV.H.values[gw] - h_star) * ρ_0_surf
        self.rho_qtflux = -self.ch * self.windspeed * (GMV.QT.values[gw] - self.qsurface) * ρ_0_surf
        self.lhf = lv * self.rho_qtflux

        if GMV.H.name == 'thetal':
            self.shf = cpm_c(GMV.QT.values[gw])  * self.rho_hflux

        elif GMV.H.name == 's':
            pv = pv_star(GMV.T.values[gw])
            pd = tmp['p_0', gw] - pv
            sv = sv_c(pv,GMV.T.values[gw])
            sd = sd_c(pd, GMV.T.values[gw])
            self.shf = (self.rho_hflux - self.lhf/lv * (sv-sd)) * GMV.T.values[gw]


        self.ustar =  sqrt(self.cm) * self.windspeed
        # CK--testing this--EDMF scheme checks greater or less than zero,
        if np.fabs(self.bflux) < 1e-10:
            self.obukhov_length = 0.0
        else:
            self.obukhov_length = -self.ustar *self.ustar *self.ustar /self.bflux /vkb

        return

    def free_convection_windspeed(self, GMV):
        SurfaceBase.free_convection_windspeed(self, GMV)
        return
