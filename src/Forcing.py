import numpy as np
from parameters import *
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid
from Field import Field, Full, Half, Dirichlet, Neumann, nice_name
from Operators import advect, grad, Laplacian, grad_pos, grad_neg

from funcs_EDMF import *
from funcs_forcing import  convert_forcing_thetal

class ForcingBase:
    def __init__(self):
        return
    def initialize(self, grid):
        self.subsidence = Half(grid)
        self.dTdt       = Half(grid)
        self.dqtdt      = Half(grid)
        self.ug         = Half(grid)
        self.vg         = Half(grid)
        return
    def update(self, grid, q, q_tendencies, tmp):
        return
    def coriolis_force(self, grid, q, q_tendencies):
        return
    def initialize_io(self, Stats):
        return
    def export_data(self, grid, Stats):
        return


class ForcingNone(ForcingBase):
    def __init__(self):
        ForcingBase.__init__(self)
        return
    def initialize(self, grid):
        ForcingBase.initialize(self, grid)
        return
    def update(self, grid, q, q_tendencies, tmp):
        return
    def coriolis_force(self, grid, q, q_tendencies):
        return
    def initialize_io(self, Stats):
        return
    def export_data(self, grid, Stats):
        return

class ForcingStandard(ForcingBase):
    def __init__(self):
        ForcingBase.__init__(self)
        return
    def initialize(self, grid):
        ForcingBase.initialize(self, grid)
        return
    def update(self, grid, q, q_tendencies, tmp):
        i_gm, i_env, i_uds, i_sd = q_tendencies.domain_idx()
        return
    def initialize_io(self, Stats):
        return
    def export_data(self, grid, Stats):
        return

# class ForcingRadiative(ForcingBase): # yair - added to avoid zero subsidence
#     def __init__(self):
#         ForcingBase.__init__(self)
#         return
#     def initialize(self, grid):
#         ForcingBase.initialize(self, grid, q_tendencies)
#         return
#     def update(self):
#
#         i_gm, i_env, i_uds, i_sd = q_tendencies.domain_idx()
#         for k in grid.over_elems_real(Center()):
#             # Apply large-scale horizontal advection tendencies
#             q_vap = q['q_tot', i_gm][k] - tmp['q_liq', i_gm][k]
#             q_tendencies['θ_liq', i_gm][k] += convert_forcing_thetal(tmp['p_0'][k],q['q_tot', i_gm][k], q_vap,
#                                                                 tmp['T', i_gm][k], self.dqtdt[k], self.dTdt[k])
#             q_tendencies['q_tot', i_gm][k] += self.dqtdt[k]
#
#
#         return
#
#     def coriolis_force(self, grid, q, q_tendencies):
#         ForcingBase.coriolis_force(self, grid, q, q_tendencies)
#         return
#     def initialize_io(self, Stats):
#         return
#     def export_data(self, grid, Stats):
#         return


class ForcingDYCOMS_RF01(ForcingBase):

    def __init__(self):
        ForcingBase.__init__(self)
        return

    def initialize(self, grid):
        ForcingBase.initialize(self, grid)

        self.alpha_z    = 1.
        self.kappa      = 85.
        self.F0         = 70.
        self.F1         = 22.
        self.divergence = 3.75e-6  # divergence is defined twice: here and in initialize_forcing method of DYCOMS_RF01 case class
                                   # where it is used to initialize large scale subsidence

        self.f_rad = Full(grid)
        return

    def calculate_radiation(self, tmp):
        """
        see eq. 3 in Stevens et. al. 2005 DYCOMS paper
        """

        # find z_i (level of 8.0 g/kg isoline of q_tot)
        k_1 = grid.first_interior(Zmin())
        z_i = grid.z[k_1]
        for k in grid.over_elems_real(Center()):
            if (q['q_tot', i_gm][k] < 8.0 / 1000):
                idx_zi = k
                # will be used at cell edges
                z_i  = grid.z[idx_zi]
                rhoi = tmp['ρ_0'][idx_zi]
                break

        self.f_rad = Full(grid)
        k_2 = grid.boundary(Zmax())
        k_1 = 0
        k_2 = grid.nzg-1

        # cloud-top cooling
        q_0 = 0.0

        self.f_rad[k_2] = self.F0 * np.exp(-q_0)
        for k in range(k_2 - 1, -1, -1):
            q_0           += self.kappa * tmp['ρ_0'][k] * tmp['q_liq', i_gm][k] * grid.dz
            self.f_rad[k]  = self.F0 * np.exp(-q_0)

        # cloud-base warming
        q_1 = 0.0
        self.f_rad[k_1] += self.F1 * np.exp(-q_1)
        for k in range(1, k_2 + 1):
            q_1           += self.kappa * tmp['ρ_0'][k - 1] * tmp['q_liq', i_gm][k - 1] * grid.dz
            self.f_rad[k] += self.F1 * np.exp(-q_1)

        # cooling in free troposphere
        for k in range(k_1, k_2):
            if grid.z[k] > z_i:
                cbrt_z         = np.cbrt(grid.z[k] - z_i)
                self.f_rad[k] += rhoi * dycoms_cp * self.divergence * self.alpha_z * (np.power(cbrt_z, 4) / 4.0 + z_i * cbrt_z)
        # condition at the top
        cbrt_z                   = np.cbrt(grid.z[k] + grid.dz - z_i)
        self.f_rad[k_2] += rhoi * dycoms_cp * self.divergence * self.alpha_z * (np.power(cbrt_z, 4) / 4.0 + z_i * cbrt_z)

        for k in grid.over_elems_real(Center()):
            self.dTdt[k] = - (self.f_rad[k + 1] - self.f_rad[k]) / grid.dz / tmp['ρ_0'][k] / dycoms_cp

        return

    def coriolis_force(self, grid, q, q_tendencies):
        ForcingBase.coriolis_force(self, grid, q, q_tendencies)
        return

    def update(self, grid, q, q_tendencies, tmp):
        self.calculate_radiation(tmp)
        i_gm, i_env, i_uds, i_sd = q_tendencies.domain_idx()

        for k in grid.over_elems_real(Center()):
            # Apply large-scale horizontal advection tendencies
            q_tot = q['q_tot', i_gm][k]
            q_vap = q_tot - tmp['q_liq', i_gm][k]
            q_tendencies['θ_liq', i_gm][k]  += convert_forcing_thetal(tmp['p_0'][k],
                                                                 q_tot,
                                                                 q_vap,
                                                                 tmp['T', i_gm][k],
                                                                 self.dqtdt[k],
                                                                 self.dTdt[k])
            q_tendencies['q_tot', i_gm][k] += self.dqtdt[k]
            # Apply large-scale subsidence tendencies
            q_tendencies['θ_liq', i_gm][k] -= grad_pos(q['θ_liq', i_gm].Cut(k), grid) * self.subsidence[k]
            q_tendencies['q_tot', i_gm][k] -= grad_pos(q['q_tot', i_gm].Cut(k), grid) * self.subsidence[k]

        if self.apply_coriolis:
            self.coriolis_force(grid, q, q_tendencies)

        return

    def initialize_io(self, Stats):
        Stats.add_profile('rad_dTdt')
        Stats.add_profile('rad_flux')
        return

    def export_data(self, grid, Stats):
        Stats.write_profile_new('rad_dTdt', grid, self.dTdt.values)
        Stats.write_profile_new('rad_flux', grid, self.f_rad.values)
        return
