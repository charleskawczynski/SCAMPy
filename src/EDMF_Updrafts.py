import numpy as np
from parameters import *
from thermodynamic_functions import  *
from microphysics_functions import  *
from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann
import ReferenceState
from Variables import GridMeanVariables
from NetCDFIO import NetCDFIO_Stats
from EDMF_Environment import EnvironmentVariables
import pylab as plt


class UpdraftVariable:
    def __init__(self, Gr, nu, loc, bc, name, units):
        self.values     = [Field.field(Gr, loc, bc) for i in range(nu)]
        self.old        = [Field.field(Gr, loc, bc) for i in range(nu)]
        self.new        = [Field.field(Gr, loc, bc) for i in range(nu)]
        self.tendencies = [Field.field(Gr, loc, bc) for i in range(nu)]
        self.bulkvalues = Field.field(Gr, loc, bc)
        self.name = name
        self.units = units

    def set_bcs(self, Gr):
        n_updrafts = np.shape(self.values)[0]
        for i in range(n_updrafts):
            self.values[i].apply_bc(Gr, 0.0)
        return

class UpdraftVariables:
    def __init__(self, nu, namelist, paramlist, Gr):
        self.grid = Gr
        self.n_updrafts = nu

        self.W    = UpdraftVariable(Gr, nu, Node(), Dirichlet(), 'w','m/s' )
        self.Area = UpdraftVariable(Gr, nu, Center(), Neumann(), 'area_fraction','[-]' )
        self.QT   = UpdraftVariable(Gr, nu, Center(), Neumann(), 'qt','kg/kg' )
        self.QL   = UpdraftVariable(Gr, nu, Center(), Neumann(), 'ql','kg/kg' )
        self.QR   = UpdraftVariable(Gr, nu, Center(), Neumann(), 'qr','kg/kg' )
        self.THL  = UpdraftVariable(Gr, nu, Center(), Neumann(), 'thetal', 'K')
        self.T    = UpdraftVariable(Gr, nu, Center(), Neumann(), 'temperature','K' )
        self.B    = UpdraftVariable(Gr, nu, Center(), Neumann(), 'buoyancy','m^2/s^3' )
        self.H = UpdraftVariable(Gr, nu, Center(), Neumann(), 'thetal','K' )

        if namelist['turbulence']['scheme'] == 'EDMF_PrognosticTKE':
            try:
                use_steady_updrafts = namelist['turbulence']['EDMF_PrognosticTKE']['use_steady_updrafts']
            except:
                use_steady_updrafts = False
            if use_steady_updrafts:
                self.prognostic = False
            else:
                self.prognostic = True
            self.updraft_fraction = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']
        else:
            self.prognostic = False
            self.updraft_fraction = paramlist['turbulence']['EDMF_BulkSteady']['surface_area']

        self.cloud_base = np.zeros((nu,), dtype=np.double, order='c')
        self.cloud_top = np.zeros((nu,), dtype=np.double, order='c')
        self.cloud_cover = np.zeros((nu,), dtype=np.double, order='c')


        return

    def initialize(self, GMV, tmp, q):
        k_1 = self.grid.first_interior(Zmin())

        for i in range(self.n_updrafts):
            for k in self.grid.over_elems(Center()):

                self.W.values[i][k] = 0.0
                # Simple treatment for now, revise when multiple updraft closures
                # become more well defined
                if self.prognostic:
                    self.Area.values[i][k] = 0.0 #self.updraft_fraction/self.n_updrafts
                else:
                    self.Area.values[i][k] = self.updraft_fraction/self.n_updrafts
                self.QT.values[i][k] = GMV.QT.values[k]
                self.QL.values[i][k] = GMV.QL.values[k]
                self.QR.values[i][k] = GMV.QR.values[k]
                self.H.values[i][k] = GMV.H.values[k]
                self.T.values[i][k] = GMV.T.values[k]
                self.B.values[i][k] = 0.0
            self.Area.values[i][k_1] = self.updraft_fraction/self.n_updrafts

        self.QT.set_bcs(self.grid)
        self.QR.set_bcs(self.grid)
        self.H.set_bcs(self.grid)

        return

    def initialize_io(self, Stats):
        Stats.add_profile('updraft_area')
        Stats.add_profile('updraft_w')
        Stats.add_profile('updraft_qt')
        Stats.add_profile('updraft_ql')
        Stats.add_profile('updraft_qr')
        Stats.add_profile('updraft_thetal')
        Stats.add_profile('updraft_temperature')
        Stats.add_profile('updraft_buoyancy')

        Stats.add_ts('updraft_cloud_cover')
        Stats.add_ts('updraft_cloud_base')
        Stats.add_ts('updraft_cloud_top')

        return

    def set_means(self, GMV):

        self.Area.bulkvalues[:] = np.sum(self.Area.values,axis=0)
        self.W.bulkvalues[:] = 0.0
        self.QT.bulkvalues[:] = 0.0
        self.QL.bulkvalues[:] = 0.0
        self.QR.bulkvalues[:] = 0.0
        self.H.bulkvalues[:] = 0.0
        self.T.bulkvalues[:] = 0.0
        self.B.bulkvalues[:] = 0.0

        for k in self.grid.over_elems_real(Center()):
            if self.Area.bulkvalues[k] > 1.0e-20:
                for i in range(self.n_updrafts):
                    self.QT.bulkvalues[k] += self.Area.values[i][k] * self.QT.values[i][k]/self.Area.bulkvalues[k]
                    self.QL.bulkvalues[k] += self.Area.values[i][k] * self.QL.values[i][k]/self.Area.bulkvalues[k]
                    self.QR.bulkvalues[k] += self.Area.values[i][k] * self.QR.values[i][k]/self.Area.bulkvalues[k]
                    self.H.bulkvalues[k] += self.Area.values[i][k] * self.H.values[i][k]/self.Area.bulkvalues[k]
                    self.T.bulkvalues[k] += self.Area.values[i][k] * self.T.values[i][k]/self.Area.bulkvalues[k]
                    self.B.bulkvalues[k] += self.Area.values[i][k] * self.B.values[i][k]/self.Area.bulkvalues[k]
                    self.W.bulkvalues[k] += self.Area.values[i].Mid(k) * self.W.values[i][k]/self.Area.bulkvalues.Mid(k)
            else:
                self.QT.bulkvalues[k] = GMV.QT.values[k]
                self.QR.bulkvalues[k] = GMV.QR.values[k]
                self.QL.bulkvalues[k] = 0.0
                self.H.bulkvalues[k] = GMV.H.values[k]
                self.T.bulkvalues[k] = GMV.T.values[k]
                self.B.bulkvalues[k] = 0.0
                self.W.bulkvalues[k] = 0.0

        return
    # quick utility to set "new" arrays with values in the "values" arrays
    def set_new_with_values(self):
        for i in range(self.n_updrafts):
            for k in self.grid.over_elems(Center()):
                self.W.new[i][k] = self.W.values[i][k]
                self.Area.new[i][k] = self.Area.values[i][k]
                self.QT.new[i][k] = self.QT.values[i][k]
                self.QL.new[i][k] = self.QL.values[i][k]
                self.QR.new[i][k] = self.QR.values[i][k]
                self.H.new[i][k] = self.H.values[i][k]
                self.THL.new[i][k] = self.THL.values[i][k]
                self.T.new[i][k] = self.T.values[i][k]
                self.B.new[i][k] = self.B.values[i][k]
        return

    # quick utility to set "new" arrays with values in the "values" arrays
    def set_old_with_values(self):
        for i in range(self.n_updrafts):
            for k in self.grid.over_elems(Center()):
                self.W.old[i][k] = self.W.values[i][k]
                self.Area.old[i][k] = self.Area.values[i][k]
                self.QT.old[i][k] = self.QT.values[i][k]
                self.QL.old[i][k] = self.QL.values[i][k]
                self.QR.old[i][k] = self.QR.values[i][k]
                self.H.old[i][k] = self.H.values[i][k]
                self.THL.old[i][k] = self.THL.values[i][k]
                self.T.old[i][k] = self.T.values[i][k]
                self.B.old[i][k] = self.B.values[i][k]
        return
    # quick utility to set "tmp" arrays with values in the "new" arrays
    def set_values_with_new(self):
        for i in range(self.n_updrafts):
            for k in self.grid.over_elems(Center()):
                self.W.values[i][k] = self.W.new[i][k]
                self.Area.values[i][k] = self.Area.new[i][k]
                self.QT.values[i][k] = self.QT.new[i][k]
                self.QL.values[i][k] = self.QL.new[i][k]
                self.QR.values[i][k] = self.QR.new[i][k]
                self.H.values[i][k] = self.H.new[i][k]
                self.THL.values[i][k] = self.THL.new[i][k]
                self.T.values[i][k] = self.T.new[i][k]
                self.B.values[i][k] = self.B.new[i][k]
        return


    def io(self, Stats):

        Stats.write_profile_new('updraft_area'       , self.grid, self.Area.bulkvalues)
        Stats.write_profile_new('updraft_w'          , self.grid, self.W.bulkvalues)
        Stats.write_profile_new('updraft_qt'         , self.grid, self.QT.bulkvalues)
        Stats.write_profile_new('updraft_ql'         , self.grid, self.QL.bulkvalues)
        Stats.write_profile_new('updraft_qr'         , self.grid, self.QR.bulkvalues)
        Stats.write_profile_new('updraft_thetal' , self.grid, self.H.bulkvalues)
        Stats.write_profile_new('updraft_temperature', self.grid, self.T.bulkvalues)
        Stats.write_profile_new('updraft_buoyancy'   , self.grid, self.B.bulkvalues)
        self.get_cloud_base_top_cover()
        # Note definition of cloud cover : each updraft is associated with a cloud cover equal to the maximum
        # area fraction of the updraft where ql > 0. Each updraft is assumed to have maximum overlap with respect to
        # itself (i.e. no consideration of tilting due to shear) while the updraft classes are assumed to have no overlap
        # at all. Thus total updraft cover is the sum of each updraft's cover
        Stats.write_ts('updraft_cloud_cover', np.sum(self.cloud_cover))
        Stats.write_ts('updraft_cloud_base', np.amin(self.cloud_base))
        Stats.write_ts('updraft_cloud_top', np.amax(self.cloud_top))

        return

    def get_cloud_base_top_cover(self):
        for i in range(self.n_updrafts):
            # Todo check the setting of ghost point z_half
            self.cloud_base[i] = self.grid.z_half[self.grid.nzg-self.grid.gw-1]
            self.cloud_top[i] = 0.0
            self.cloud_cover[i] = 0.0
            for k in self.grid.over_elems_real(Center()):
                if self.QL.values[i][k] > 1e-8 and self.Area.values[i][k] > 1e-3:
                    self.cloud_base[i] = np.fmin(self.cloud_base[i], self.grid.z_half[k])
                    self.cloud_top[i] = np.fmax(self.cloud_top[i], self.grid.z_half[k])
                    self.cloud_cover[i] = np.fmax(self.cloud_cover[i], self.Area.values[i][k])
        return

class UpdraftThermodynamics:
    def __init__(self, n_updrafts, Gr, Ref, UpdVar):
        self.grid = Gr
        self.Ref = Ref
        self.n_updrafts = n_updrafts
        self.t_to_prog_fp = t_to_thetali_c
        self.prog_to_t_fp = eos_first_guess_thetal

        return
    def satadjust(self, UpdVar, tmp):
        for i in range(self.n_updrafts):
            for k in self.grid.over_elems(Center()):
                T, ql = eos(self.t_to_prog_fp, self.prog_to_t_fp, tmp['p_0_half'][k], UpdVar.QT.values[i][k], UpdVar.H.values[i][k])
                UpdVar.QL.values[i][k] = ql
                UpdVar.T.values[i][k] = T
        return

    def buoyancy(self,  UpdVar, EnvVar, GMV, extrap, tmp):
        UpdVar.Area.bulkvalues[:] = np.sum(UpdVar.Area.values,axis=0)
        for i in range(self.n_updrafts):
            for k in self.grid.over_elems_real(Center()):
                if UpdVar.Area.values[i][k] > 1e-3:
                    q_tot = UpdVar.QT.values[i][k]
                    q_vap = UpdVar.QT.values[i][k] - UpdVar.QL.values[i][k]
                    T = UpdVar.T.values[i][k]
                    α_i = alpha_c(tmp['p_0_half'][k], T, q_tot, q_vap)
                    UpdVar.B.values[i][k] = buoyancy_c(tmp['α_0_half'][k], α_i)
                else:
                    UpdVar.B.values[i][k] = EnvVar.B.values[k]
        # Subtract grid mean buoyancy
        for k in self.grid.over_elems_real(Center()):
            GMV.B.values[k] = (1.0 - UpdVar.Area.bulkvalues[k]) * EnvVar.B.values[k]
            for i in range(self.n_updrafts):
                GMV.B.values[k] += UpdVar.Area.values[i][k] * UpdVar.B.values[i][k]
            for i in range(self.n_updrafts):
                UpdVar.B.values[i][k] -= GMV.B.values[k]
            EnvVar.B.values[k] -= GMV.B.values[k]
        return

#Implements a simple "microphysics" that clips excess humidity above a user-specified level
class UpdraftMicrophysics:
    def __init__(self, paramlist, n_updrafts, Gr, Ref):
        self.grid = Gr
        self.Ref = Ref
        self.n_updrafts = n_updrafts
        self.max_supersaturation = paramlist['turbulence']['updraft_microphysics']['max_supersaturation']
        self.prec_source_h = [Half(Gr) for i in range(n_updrafts)]
        self.prec_source_qt = [Half(Gr) for i in range(n_updrafts)]
        self.prec_source_h_tot  = Half(Gr)
        self.prec_source_qt_tot = Half(Gr)
        return

    def compute_sources(self, UpdVar, tmp):
        """
        Compute precipitation source terms for QT, QR and H
        """
        for i in range(self.n_updrafts):
            for k in self.grid.over_elems(Center()):
                tmp_qr = acnv_instant(UpdVar.QL.values[i][k], UpdVar.QT.values[i][k], self.max_supersaturation,\
                                      UpdVar.T.values[i][k], tmp['p_0_half'][k])
                self.prec_source_qt[i][k] = -tmp_qr
                self.prec_source_h[i][k]  = rain_source_to_thetal(tmp['p_0_half'][k], UpdVar.T.values[i][k],\
                                             UpdVar.QT.values[i][k], UpdVar.QL.values[i][k], 0.0, tmp_qr)
        for k in self.grid.over_elems(Center()):
            self.prec_source_h_tot[k]  = np.sum([self.prec_source_h[i][k] * UpdVar.Area.values[i][k] for i in range(self.n_updrafts)])
            self.prec_source_qt_tot[k] = np.sum([self.prec_source_qt[i][k]* UpdVar.Area.values[i][k] for i in range(self.n_updrafts)])

        return

    def update_updraftvars(self, UpdVar):
        """
        Apply precipitation source terms to QL, QR and H
        """
        for i in range(self.n_updrafts):
            for k in self.grid.over_elems(Center()):
                UpdVar.QT.values[i][k] += self.prec_source_qt[i][k]
                UpdVar.QL.values[i][k] += self.prec_source_qt[i][k]
                UpdVar.QR.values[i][k] -= self.prec_source_qt[i][k]
                UpdVar.H.values[i][k] += self.prec_source_h[i][k]
        return

    def compute_update_combined_local_thetal(self, p0, T, qt, ql, qr, h, i, k):

        tmp_qr = acnv_instant(ql[i][k], qt[i][k], self.max_supersaturation, T[i][k], p0[k])
        self.prec_source_qt[i][k] = -tmp_qr
        self.prec_source_h[i][k]  = rain_source_to_thetal(p0[k], T[i][k], qt[i][k], ql[i][k], 0.0, tmp_qr)
        qt[i][k] += self.prec_source_qt[i][k]
        ql[i][k] += self.prec_source_qt[i][k]
        qr[i][k] -= self.prec_source_qt[i][k]
        h[i][k]  += self.prec_source_h[i][k]

        return
