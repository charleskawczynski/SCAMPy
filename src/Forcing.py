import numpy as np
from parameters import *
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
from Operators import advect, grad, Laplacian

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

        if GMV.H.name == 's':
            self.convert_forcing_prog_fp = convert_forcing_entropy
        elif GMV.H.name == 'thetal':
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
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.H.tendencies[k] += self.convert_forcing_prog_fp(tmp['p_0'][k],GMV.QT.values[k],
                                                                qv, GMV.T.values[k], self.dqtdt[k], self.dTdt[k])
            GMV.QT.tendencies[k] += self.dqtdt[k]
        if self.apply_subsidence:
            for k in self.grid.over_elems_real(Center()):
                # Apply large-scale subsidence tendencies
                GMV.H.tendencies[k] -= grad(GMV.H.values.Dual(k), self.grid) * self.subsidence[k]
                GMV.QT.tendencies[k] -= grad(GMV.QT.values.Dual(k), self.grid) * self.subsidence[k]

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
#             qv = GMV.QT.values[k] - GMV.QL.values[k]
#             GMV.H.tendencies[k] += self.convert_forcing_prog_fp(tmp['p_0'][k],GMV.QT.values[k], qv,
#                                                                 GMV.T.values[k], self.dqtdt[k], self.dTdt[k])
#             GMV.QT.tendencies[k] += self.dqtdt[k]
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
        for k in self.grid.over_elems_real(Center()):
            if (GMV.QT.values[k] < 8.0 / 1000):
                idx_zi = k
                # will be used at cell edges
                zi     = self.grid.z[idx_zi]
                rhoi   = tmp['ρ_0', idx_zi]
                break

        # cloud-top cooling
        q_0 = 0.0

        self.f_rad = Full(self.grid)
        k_2 = self.grid.boundary(Zmax())
        self.f_rad[k_2] = self.F0 * np.exp(-q_0)

        # Verify correctness of this to original definition
        for k in reversed(self.grid.over_elems(Node())[1:-1]):
            q_0           += self.kappa * tmp['ρ_0'][k] * GMV.QL.values[k] * self.grid.dz
            self.f_rad[k]  = self.F0 * np.exp(-q_0)

        # cloud-base warming
        q_1 = 0.0
        self.f_rad[0] += self.F1 * np.exp(-q_1)
        for k in self.grid.over_elems_real(Node()):
            q_1           += self.kappa * tmp['ρ_0'][k - 1] * GMV.QL.values[k - 1] * self.grid.dz
            self.f_rad[k] += self.F1 * np.exp(-q_1)

        # cooling in free troposphere
        for k in self.grid.over_elems(Node())[1:-1]:
            if self.grid.z[k] > zi:
                cbrt_z         = cbrt(self.grid.z[k] - zi)
                self.f_rad[k] += rhoi * dycoms_cp * self.divergence * self.alpha_z * (np.power(cbrt_z, 4) / 4.0 + zi * cbrt_z)
        # condition at the top
        cbrt_z                   = cbrt(self.grid.z[k] + self.grid.dz - zi)
        self.f_rad[k_2] += rhoi * dycoms_cp * self.divergence * self.alpha_z * (np.power(cbrt_z, 4) / 4.0 + zi * cbrt_z)

        for k in self.grid.over_elems_real(Center()):
            self.dTdt[k] = - (self.f_rad[k + 1] - self.f_rad[k]) / self.grid.dz / tmp['ρ_0'][k] / dycoms_cp

        return

    def coriolis_force(self, U, V):
        ForcingBase.coriolis_force(self, U, V)
        return

    def update(self, GMV, tmp):
        self.calculate_radiation(GMV)

        for k in self.grid.over_elems_real(Center()):
            # Apply large-scale horizontal advection tendencies
            qv = GMV.QT.values[k] - GMV.QL.values[k]
            GMV.H.tendencies[k]  += self.convert_forcing_prog_fp(tmp['p_0'][k],GMV.QT.values[k], qv, GMV.T.values[k], self.dqtdt[k], self.dTdt[k])
            GMV.QT.tendencies[k] += self.dqtdt[k]
            # Apply large-scale subsidence tendencies
            GMV.H.tendencies[k]  -= (GMV.H.values[k+1]-GMV.H.values[k]) * self.grid.dzi * self.subsidence[k]
            GMV.QT.tendencies[k] -= (GMV.QT.values[k+1]-GMV.QT.values[k]) * self.grid.dzi * self.subsidence[k]

        if self.apply_coriolis:
            self.coriolis_force(GMV.U, GMV.V)

        return

    def initialize_io(self, Stats):
        Stats.add_profile('rad_dTdt')
        Stats.add_profile('rad_flux')
        return

    def io(self, Stats):
        Stats.write_profile('rad_dTdt', self.dTdt[ self.grid.gw     : self.grid.nzg - self.grid.gw])
        Stats.write_profile('rad_flux', self.f_rad[self.grid.gw + 1 : self.grid.nzg - self.grid.gw + 1])
        return
