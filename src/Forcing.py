import numpy as np
from parameters import *
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
from Operators import advect, grad, Laplacian, grad_pos, grad_neg

from Variables import GridMeanVariables, VariablePrognostic
from forcing_functions import  convert_forcing_entropy, convert_forcing_thetal

class ForcingBase:
    def __init__(self):
        return
    def initialize(self, GMV):
        self.subsidence = Half(self.grid)
        self.dTdt       = Half(self.grid)
        self.dqtdt      = Half(self.grid)
        self.ug         = Half(self.grid)
        self.vg         = Half(self.grid)
        self.convert_forcing_prog_fp = convert_forcing_thetal
        return
    def update(self, GMV, tmp):
        return
    def coriolis_force(self, U, V):
        for k in self.grid.over_elems_real(Center()):
            U.tendencies[k] -= self.coriolis_param * (self.vg[k] - V.values[k])
            V.tendencies[k] += self.coriolis_param * (self.ug[k] - U.values[k])
        return
    def initialize_io(self, Stats):
        return
    def io(self, Stats):
        return


class ForcingNone(ForcingBase):
    def __init__(self):
        ForcingBase.__init__(self)
        return
    def initialize(self, GMV):
        ForcingBase.initialize(self, GMV)
        return
    def update(self, GMV, tmp):
        return
    def coriolis_force(self, U, V):
        return
    def initialize_io(self, Stats):
        return
    def io(self, Stats):
        return

class ForcingStandard(ForcingBase):
    def __init__(self):
        ForcingBase.__init__(self)
        return
    def initialize(self, GMV):
        ForcingBase.initialize(self, GMV)
        return
    def update(self, GMV, tmp):

        for k in self.grid.over_elems_real(Center()):
            # Apply large-scale horizontal advection tendencies
            qv = GMV.q_tot.values[k] - GMV.q_liq.values[k]
            GMV.H.tendencies[k] += self.convert_forcing_prog_fp(tmp['p_0_half'][k], GMV.q_tot.values[k],
                                                                qv, GMV.T.values[k], self.dqtdt[k], self.dTdt[k])
            GMV.q_tot.tendencies[k] += self.dqtdt[k]
        if self.apply_subsidence:
            for k in self.grid.over_elems_real(Center()):
                # Apply large-scale subsidence tendencies
                GMV.H.tendencies[k] -= grad(GMV.H.values.Dual(k), self.grid) * self.subsidence[k]
                GMV.q_tot.tendencies[k] -= grad(GMV.q_tot.values.Dual(k), self.grid) * self.subsidence[k]

        if self.apply_coriolis:
            self.coriolis_force(GMV.U, GMV.V)

        return
    def initialize_io(self, Stats):
        return
    def io(self, Stats):
        return

# class ForcingRadiative(ForcingBase): # yair - added to avoid zero subsidence
#     def __init__(self):
#         ForcingBase.__init__(self)
#         return
#     def initialize(self, GMV):
#         ForcingBase.initialize(self, GMV)
#         return
#     def update(self, GMV):
#
#         for k in self.grid.over_elems_real(Center()):
#             # Apply large-scale horizontal advection tendencies
#             qv = GMV.q_tot.values[k] - GMV.q_liq.values[k]
#             GMV.H.tendencies[k] += self.convert_forcing_prog_fp(tmp['p_0_half'][k],GMV.q_tot.values[k], qv,
#                                                                 GMV.T.values[k], self.dqtdt[k], self.dTdt[k])
#             GMV.q_tot.tendencies[k] += self.dqtdt[k]
#
#
#         return
#
#     def coriolis_force(self, U, V):
#         ForcingBase.coriolis_force(self, U, V)
#         return
#     def initialize_io(self, Stats):
#         return
#     def io(self, Stats):
#         return


class ForcingDYCOMS_RF01(ForcingBase):

    def __init__(self):
        ForcingBase.__init__(self)
        return

    def initialize(self, GMV):
        ForcingBase.initialize(self, GMV)

        self.alpha_z    = 1.
        self.kappa      = 85.
        self.F0         = 70.
        self.F1         = 22.
        self.divergence = 3.75e-6  # divergence is defined twice: here and in initialize_forcing method of DYCOMS_RF01 case class
                                   # where it is used to initialize large scale subsidence

        self.f_rad = Full(self.grid)
        return

    def calculate_radiation(self, GMV, tmp):
        """
        see eq. 3 in Stevens et. al. 2005 DYCOMS paper
        """

        # find zi (level of 8.0 g/kg isoline of qt)
        k_1 = self.grid.first_interior(Zmin())
        zi     = self.grid.z[k_1]
        for k in self.grid.over_elems_real(Center()):
            if (GMV.q_tot.values[k] < 8.0 / 1000):
                idx_zi = k
                # will be used at cell edges
                zi     = self.grid.z[idx_zi]
                rhoi   = tmp['ρ_0_half'][idx_zi]
                break

        self.f_rad = Full(self.grid)
        k_2 = self.grid.boundary(Zmax())
        k_1 = 0
        k_2 = self.grid.nzg-1

        # cloud-top cooling
        q_0 = 0.0

        self.f_rad[k_2] = self.F0 * np.exp(-q_0)
        for k in range(k_2 - 1, -1, -1):
            q_0           += self.kappa * self.Ref.rho0_half[k] * GMV.q_liq.values[k] * self.grid.dz
            self.f_rad[k]  = self.F0 * np.exp(-q_0)

        # cloud-base warming
        q_1 = 0.0
        self.f_rad[k_1] += self.F1 * np.exp(-q_1)
        for k in range(1, k_2 + 1):
            q_1           += self.kappa * self.Ref.rho0_half[k - 1] * GMV.q_liq.values[k - 1] * self.grid.dz
            self.f_rad[k] += self.F1 * np.exp(-q_1)

        # cooling in free troposphere
        for k in range(k_1, k_2):
            if self.grid.z[k] > zi:
                cbrt_z         = np.cbrt(self.grid.z[k] - zi)
                self.f_rad[k] += rhoi * dycoms_cp * self.divergence * self.alpha_z * (np.power(cbrt_z, 4) / 4.0 + zi * cbrt_z)
        # condition at the top
        cbrt_z                   = np.cbrt(self.grid.z[k] + self.grid.dz - zi)
        self.f_rad[k_2] += rhoi * dycoms_cp * self.divergence * self.alpha_z * (np.power(cbrt_z, 4) / 4.0 + zi * cbrt_z)

        for k in self.grid.over_elems_real(Center()):
            self.dTdt[k] = - (self.f_rad[k + 1] - self.f_rad[k]) / self.grid.dz / self.Ref.rho0_half[k] / dycoms_cp

        return

    def coriolis_force(self, U, V):
        ForcingBase.coriolis_force(self, U, V)
        return

    def update(self, GMV, tmp):
        self.calculate_radiation(GMV, tmp)

        for k in self.grid.over_elems_real(Center()):
            # Apply large-scale horizontal advection tendencies
            qv = GMV.q_tot.values[k] - GMV.q_liq.values[k]
            GMV.H.tendencies[k]  += self.convert_forcing_prog_fp(tmp['p_0_half'][k],GMV.q_tot.values[k], qv, GMV.T.values[k], self.dqtdt[k], self.dTdt[k])
            GMV.q_tot.tendencies[k] += self.dqtdt[k]
            # Apply large-scale subsidence tendencies
            GMV.H.tendencies[k]  -= grad_pos(GMV.H.values.Cut(k), self.grid) * self.subsidence[k]
            GMV.q_tot.tendencies[k] -= grad_pos(GMV.q_tot.values.Cut(k), self.grid) * self.subsidence[k]

        if self.apply_coriolis:
            self.coriolis_force(GMV.U, GMV.V)

        return

    def initialize_io(self, Stats):
        Stats.add_profile('rad_dTdt')
        Stats.add_profile('rad_flux')
        return

    def io(self, Stats):
        k_1 = self.grid.boundary(Zmin())
        k_2 = self.grid.boundary(Zmax())
        Stats.write_profile('rad_dTdt', self.dTdt[k_1:k_2])
        Stats.write_profile('rad_flux', self.f_rad[k_1:k_2])
        return
