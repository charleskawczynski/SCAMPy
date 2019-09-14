import numpy as np
from parameters import *
from Grid import Grid, Zmin, Zmax, Center, Node
from MoistThermodynamics import *
from funcs_surface import compute_ustar, buoyancy_flux, exchange_coefficients_byun
from funcs_turbulence import *
from funcs_EDMF import *
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

    def update(self):
        return
    def free_convection_windspeed(self, grid, q, tmp):
        gm, en, ud, sd, al = tmp.idx.allcombinations()
        theta_rho = Half(grid)
        for k in grid.over_elems(Center()):
            q_pt = PhasePartitionRaw(q['q_tot', gm][k], tmp['q_liq', gm][k])
            theta_rho[k] = virtual_pottemp_raw(tmp['T', gm][k], tmp['p_0'][k], q_pt)


        zi = compute_inversion_height(theta_rho, q['u', gm], q['v', gm], grid, self.Ri_bulk_crit)
        wstar = compute_convective_velocity(self.bflux, zi) # yair here zi in TRMM should be adjusted
        self.windspeed = np.sqrt(self.windspeed*self.windspeed  + (1.2 *wstar)*(1.2 * wstar) )
        return

def compute_windspeed(grid, q, windspeed_min):
    gm, en, ud, sd, al = q.idx.allcombinations()
    k_1 = grid.first_interior(Zmin())
    return np.maximum(np.sqrt(q['u', gm][k_1]**2.0 + q['v', gm][k_1]**2.0), windspeed_min)

def compute_MO_len(ustar, bflux):
    if np.fabs(bflux) < 1e-10:
        return 0.0
    else:
        return -ustar * ustar * ustar / bflux / vkb

class SurfaceFixedFlux(SurfaceBase):
    def __init__(self,paramlist):
        SurfaceBase.__init__(self, paramlist)
        return
    def initialize(self):
        return

    def update(self, grid, q, tmp):
        gm, en, ud, sd, al = tmp.idx.allcombinations()
        k_1 = grid.first_interior(Zmin())
        z_1 = grid.z_half[k_1]
        ρ_0_surf = tmp.surface(grid, 'ρ_0')
        α_0_surf = tmp.surface(grid, 'α_0')
        T_1 = tmp['T', gm][k_1]
        θ_liq_1 = q['θ_liq', gm][k_1]
        q_tot_1 = q['q_tot', gm][k_1]
        V_1 = q['v', gm][k_1]
        U_1 = q['u', gm][k_1]

        rho_tflux =  self.shf /(cp_m(self.qsurface))
        self.windspeed = compute_windspeed(grid, q, 0.0)
        self.rho_q_tot_flux = self.lhf/(latent_heat_vapor_raw(self.Tsurface))
        self.rho_θ_liq_flux = rho_tflux / exner(self.Ref.Pg)
        self.bflux = buoyancy_flux(self.shf, self.lhf, T_1, q_tot_1, α_0_surf)

        if not self.ustar_fixed:
            # Correction to windspeed for free convective cases (Beljaars, QJRMS (1994), 121, pp. 255-270)
            # Value 1.2 is empirical, but should be O(1)
            if self.windspeed < 0.1:  # Limit here is heuristic
                if self.bflux > 0.0:
                   self.free_convection_windspeed(grid, q, tmp)
                else:
                    print('WARNING: Low windspeed + stable conditions, need to check ustar computation')
                    print('self.bflux ==>', self.bflux)
                    print('self.shf ==>', self.shf)
                    print('self.lhf ==>', self.lhf)
                    print('U_1  ==>', U_1)
                    print('V_1  ==>', V_1)
                    print('q_tot_1 ==>', q_tot_1)
                    print('α_0_surf ==>', α_0_surf)

            self.ustar = compute_ustar(self.windspeed, self.bflux, self.zrough, z_1)

        self.obukhov_length = compute_MO_len(self.ustar, self.bflux)
        self.rho_uflux = - ρ_0_surf *  self.ustar * self.ustar / self.windspeed * U_1
        self.rho_vflux = - ρ_0_surf *  self.ustar * self.ustar / self.windspeed * V_1
        return

# Cases such as Rico which provide values of transfer coefficients
class SurfaceFixedCoeffs(SurfaceBase):
    def __init__(self, paramlist):
        SurfaceBase.__init__(self, paramlist)
        return
    def initialize(self):
        pvg = saturation_vapor_pressure_raw(self.Tsurface, Liquid())
        pdg = self.Ref.Pg - pvg

        q_pt = PhasePartitionRaw(0.0)
        ρ = air_density_raw(self.Tsurface, self.Ref.Pg, q_pt)
        self.qsurface = q_vap_saturation_raw(self.Tsurface, ρ, q_pt)
        self.s_surface = (1.0-self.qsurface) * sd_c(pdg, self.Tsurface) + self.qsurface * sv_c(pvg,self.Tsurface)
        return

    def update(self, grid, q, tmp):
        gm, en, ud, sd, al = tmp.idx.allcombinations()
        k_1 = grid.first_interior(Zmin())
        ρ_0_surf = tmp.surface(grid, 'ρ_0')
        α_0_surf = tmp.surface(grid, 'α_0')
        T_1 = tmp['T', gm][k_1]
        θ_liq_1 = q['θ_liq', gm][k_1]
        q_tot_1 = q['q_tot', gm][k_1]
        V_1 = q['v', gm][k_1]
        U_1 = q['u', gm][k_1]

        cp_ = cp_m(q_tot_1)
        lv = latent_heat_vapor_raw(T_1)
        windspeed = compute_windspeed(grid, q, 0.01)
        self.rho_q_tot_flux = -self.cq * windspeed * (q_tot_1 - self.qsurface) * ρ_0_surf
        self.rho_θ_liq_flux = -self.ch * windspeed * (θ_liq_1 - self.Tsurface/exner(self.Ref.Pg)) * ρ_0_surf

        self.lhf = lv * self.rho_q_tot_flux
        self.shf = cp_  * self.rho_θ_liq_flux

        self.bflux = buoyancy_flux(self.shf, self.lhf, T_1, q_tot_1, α_0_surf)
        self.ustar =  np.sqrt(self.cm) * windspeed
        self.obukhov_length = compute_MO_len(self.ustar, self.bflux)

        self.rho_uflux = - ρ_0_surf *  self.ustar * self.ustar / windspeed * U_1
        self.rho_vflux = - ρ_0_surf *  self.ustar * self.ustar / windspeed * V_1
        return

class SurfaceMoninObukhov(SurfaceBase):
    def __init__(self, paramlist):
        SurfaceBase.__init__(self, paramlist)
        return
    def initialize(self):
        return
    def update(self, grid, q, tmp):
        gm, en, ud, sd, al = tmp.idx.allcombinations()
        k_1 = grid.first_interior(Zmin())
        z_1 = grid.z_half[k_1]
        ρ_0_surf = tmp.surface(grid, 'ρ_0')
        α_0_surf = tmp.surface(grid, 'α_0')
        p_1 = tmp['p_0'][k_1]
        T_1 = tmp['T', gm][k_1]
        θ_liq_1 = q['θ_liq', gm][k_1]
        q_tot_1 = q['q_tot', gm][k_1]
        V_1 = q['v', gm][k_1]
        U_1 = q['u', gm][k_1]

        q_pt = PhasePartitionRaw(0.0)
        ρ = air_density_raw(self.Tsurface, self.Ref.Pg, q_pt)
        self.qsurface = q_vap_saturation_raw(self.Tsurface, ρ, q_pt)

        q_pt = PhasePartitionRaw(self.qsurface)
        theta_rho_g = virtual_pottemp_raw(self.Tsurface, self.Ref.Pg, q_pt)
        theta_rho_b = virtual_pottemp_raw(T_1, p_1, q_pt)

        lv = latent_heat_vapor_raw(T_1)

        θ_liq_star = thetali_c(self.Ref.Pg, self.Tsurface, self.qsurface, 0.0, 0.0)

        self.windspeed = compute_windspeed(grid, q, 0.0)
        Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/z_1
        Ri = Nb2 * z_1 * z_1/(self.windspeed * self.windspeed)

        self.cm, self.ch, self.obukhov_length = exchange_coefficients_byun(Ri, z_1, self.zrough)

        self.rho_uflux = -self.cm * self.windspeed * U_1 * ρ_0_surf
        self.rho_vflux = -self.cm * self.windspeed * V_1 * ρ_0_surf
        self.rho_θ_liq_flux = -self.ch * self.windspeed * (θ_liq_1  - θ_liq_star) * ρ_0_surf
        self.rho_q_tot_flux = -self.ch * self.windspeed * (q_tot_1 - self.qsurface) * ρ_0_surf

        self.lhf = lv * self.rho_q_tot_flux
        self.shf = cp_m(q_tot_1) * self.rho_θ_liq_flux

        self.bflux = buoyancy_flux(self.shf, self.lhf, T_1, q_tot_1, α_0_surf)
        self.ustar =  np.sqrt(self.cm) * self.windspeed
        self.obukhov_length = compute_MO_len(self.ustar, self.bflux)

        return


# Not fully implemented yet. Maybe not needed - Ignacio
class SurfaceSullivanPatton(SurfaceBase):
    def __init__(self, paramlist):
        SurfaceBase.__init__(self, paramlist)
        return
    def initialize(self):
        return
    def update(self, grid, q, tmp):
        gm, en, ud, sd, al = tmp.idx.allcombinations()
        k_1 = grid.first_interior(Zmin())
        z_1 = grid.z_half[k_1]
        p_0_1 = tmp['p_0'][k_1]
        α_0_1 = tmp['α_0'][k_1]
        ρ_0_surf = tmp.surface(grid, 'ρ_0')
        T_1 = tmp['T', gm][k_1]
        θ_liq_1 = q['θ_liq', gm][k_1]
        q_tot_1 = q['q_tot', gm][k_1]
        V_1 = q['v', gm][k_1]
        U_1 = q['u', gm][k_1]

        q_pt = PhasePartitionRaw(0.0)
        ρ = air_density_raw(self.Tsurface, self.Ref.Pg, q_pt)
        self.qsurface = q_vap_saturation_raw(self.Tsurface, ρ, q_pt)

        q_pt = PhasePartitionRaw(self.qsurface)
        theta_rho_g = virtual_pottemp_raw(self.Tsurface, self.Ref.Pg, q_pt)
        theta_rho_b = virtual_pottemp_raw(T_1, p_1, q_pt)
        lv = latent_heat_vapor_raw(T_1)
        T0 = p_0_1 * α_0_1/Rd

        theta_flux = 0.24

        θ_liq_star = thetali_c(self.Ref.Pg, self.Tsurface, self.qsurface, 0.0, 0.0)

        self.windspeed = compute_windspeed(grid, q, 0.0)
        Nb2 = g/theta_rho_g*(theta_rho_b-theta_rho_g)/z_1
        Ri = Nb2 * z_1 * z_1/(self.windspeed * self.windspeed)

        self.cm, self.ch, self.obukhov_length = exchange_coefficients_byun(Ri, z_1, self.zrough)

        self.rho_uflux = -self.cm * self.windspeed * U_1 * ρ_0_surf
        self.rho_vflux = -self.cm * self.windspeed * V_1 * ρ_0_surf
        self.rho_θ_liq_flux =  -self.ch * self.windspeed * (θ_liq_1 - θ_liq_star) * ρ_0_surf
        self.rho_q_tot_flux = -self.ch * self.windspeed * (q_tot_1 - self.qsurface) * ρ_0_surf
        self.lhf = lv * self.rho_q_tot_flux
        self.shf = cp_m(q_tot_1)  * self.rho_θ_liq_flux

        self.bflux = g * theta_flux * exner(p_0_1) / T0
        self.ustar =  sqrt(self.cm) * self.windspeed
        self.obukhov_length = compute_MO_len(self.ustar, self.bflux)
        return
