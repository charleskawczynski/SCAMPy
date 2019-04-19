import numpy as np
from parameters import *
from thermodynamic_functions import  *
from microphysics_functions import  *
import Grid as Grid
from Field import Field
import ReferenceState
from Variables import GridMeanVariables
from NetCDFIO import NetCDFIO_Stats
from EDMF_Environment import EnvironmentVariables
import pylab as plt


class UpdraftVariable:
    def __init__(self, nu, nz, loc, kind, name, units):
        self.values     = np.zeros((nu,nz),dtype=np.double, order='c')
        self.old        = np.zeros((nu,nz),dtype=np.double, order='c') # needed for prognostic updrafts
        self.new        = np.zeros((nu,nz),dtype=np.double, order='c') # needed for prognostic updrafts
        self.tendencies = np.zeros((nu,nz),dtype=np.double, order='c')
        self.flux       = np.zeros((nu,nz),dtype=np.double, order='c')
        self.bulkvalues = np.zeros((nz,)  ,dtype=np.double, order='c')
        if loc != 'half' and loc != 'full':
            print('Invalid location setting for variable! Must be half or full')
        self.loc = loc
        if kind != 'scalar' and kind != 'velocity':
            print ('Invalid kind setting for variable! Must be scalar or velocity')
        self.kind = kind
        self.name = name
        self.units = units

    def set_bcs(self,Gr):
        start_low = Gr.gw - 1
        start_high = Gr.nzg - Gr.gw - 1

        n_updrafts = np.shape(self.values)[0]

        if self.name == 'w':
            for i in range(n_updrafts):
                self.values[i,start_high] = 0.0
                self.values[i,start_low] = 0.0
                for k in range(1,Gr.gw):
                    self.values[i,start_high+ k] = -self.values[i,start_high - k ]
                    self.values[i,start_low- k] = -self.values[i,start_low + k  ]
        else:
            for k in range(Gr.gw):
                for i in range(n_updrafts):
                    self.values[i,start_high + k +1] = self.values[i,start_high  - k]
                    self.values[i,start_low - k] = self.values[i,start_low + 1 + k]
        return


class UpdraftVariables:
    def __init__(self, nu, namelist, paramlist, Gr):
        self.Gr = Gr
        self.n_updrafts = nu
        nzg = Gr.nzg

        self.W = UpdraftVariable(nu, nzg, 'full', 'velocity', 'w','m/s' )
        self.Area = UpdraftVariable(nu, nzg, 'full', 'scalar', 'area_fraction','[-]' )
        self.QT = UpdraftVariable(nu, nzg, 'half', 'scalar', 'qt','kg/kg' )
        self.QL = UpdraftVariable(nu, nzg, 'half', 'scalar', 'ql','kg/kg' )
        self.QR = UpdraftVariable(nu, nzg, 'half', 'scalar', 'qr','kg/kg' )
        self.THL = UpdraftVariable(nu, nzg, 'half', 'scalar', 'thetal', 'K')
        self.T = UpdraftVariable(nu, nzg, 'half', 'scalar', 'temperature','K' )
        self.B = UpdraftVariable(nu, nzg, 'half', 'scalar', 'buoyancy','m^2/s^3' )
        if namelist['thermodynamics']['thermal_variable'] == 'entropy':
            self.H = UpdraftVariable(nu, nzg, 'half', 'scalar', 's','J/kg/K' )
        elif namelist['thermodynamics']['thermal_variable'] == 'thetal':
            self.H = UpdraftVariable(nu, nzg, 'half', 'scalar', 'thetal','K' )


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

    def initialize(self, GMV):
        gw = self.Gr.gw
        dz = self.Gr.dz

        for i in range(self.n_updrafts):
            for k in range(self.Gr.nzg):

                self.W.values[i,k] = 0.0
                # Simple treatment for now, revise when multiple updraft closures
                # become more well defined
                if self.prognostic:
                    self.Area.values[i,k] = 0.0 #self.updraft_fraction/self.n_updrafts
                else:
                    self.Area.values[i,k] = self.updraft_fraction/self.n_updrafts
                self.QT.values[i,k] = GMV.QT.values[k]
                self.QL.values[i,k] = GMV.QL.values[k]
                self.QR.values[i,k] = GMV.QR.values[k]
                self.H.values[i,k] = GMV.H.values[k]
                self.T.values[i,k] = GMV.T.values[k]
                self.B.values[i,k] = 0.0
            self.Area.values[i,gw] = self.updraft_fraction/self.n_updrafts

        self.QT.set_bcs(self.Gr)
        self.QR.set_bcs(self.Gr)
        self.H.set_bcs(self.Gr)

        return

    def initialize_io(self, Stats):
        Stats.add_profile('updraft_area')
        Stats.add_profile('updraft_w')
        Stats.add_profile('updraft_qt')
        Stats.add_profile('updraft_ql')
        Stats.add_profile('updraft_qr')
        if self.H.name == 'thetal':
            Stats.add_profile('updraft_thetal')
        else:
            # Stats.add_profile('updraft_thetal')
            Stats.add_profile('updraft_s')
        Stats.add_profile('updraft_temperature')
        Stats.add_profile('updraft_buoyancy')

        Stats.add_ts('updraft_cloud_cover')
        Stats.add_ts('updraft_cloud_base')
        Stats.add_ts('updraft_cloud_top')

        return

    def set_means(self, GMV):

        self.Area.bulkvalues = np.sum(self.Area.values,axis=0)
        self.W.bulkvalues[:] = 0.0
        self.QT.bulkvalues[:] = 0.0
        self.QL.bulkvalues[:] = 0.0
        self.QR.bulkvalues[:] = 0.0
        self.H.bulkvalues[:] = 0.0
        self.T.bulkvalues[:] = 0.0
        self.B.bulkvalues[:] = 0.0


        for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            if self.Area.bulkvalues[k] > 1.0e-20:
                for i in range(self.n_updrafts):
                    self.QT.bulkvalues[k] += self.Area.values[i,k] * self.QT.values[i,k]/self.Area.bulkvalues[k]
                    self.QL.bulkvalues[k] += self.Area.values[i,k] * self.QL.values[i,k]/self.Area.bulkvalues[k]
                    self.QR.bulkvalues[k] += self.Area.values[i,k] * self.QR.values[i,k]/self.Area.bulkvalues[k]
                    self.H.bulkvalues[k] += self.Area.values[i,k] * self.H.values[i,k]/self.Area.bulkvalues[k]
                    self.T.bulkvalues[k] += self.Area.values[i,k] * self.T.values[i,k]/self.Area.bulkvalues[k]
                    self.B.bulkvalues[k] += self.Area.values[i,k] * self.B.values[i,k]/self.Area.bulkvalues[k]
                    self.W.bulkvalues[k] += ((self.Area.values[i,k] + self.Area.values[i,k+1]) * self.W.values[i,k]
                                        /(self.Area.bulkvalues[k] + self.Area.bulkvalues[k+1]))
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
            for k in range(self.Gr.nzg):
                self.W.new[i,k] = self.W.values[i,k]
                self.Area.new[i,k] = self.Area.values[i,k]
                self.QT.new[i,k] = self.QT.values[i,k]
                self.QL.new[i,k] = self.QL.values[i,k]
                self.QR.new[i,k] = self.QR.values[i,k]
                self.H.new[i,k] = self.H.values[i,k]
                self.THL.new[i,k] = self.THL.values[i,k]
                self.T.new[i,k] = self.T.values[i,k]
                self.B.new[i,k] = self.B.values[i,k]
        return

    # quick utility to set "new" arrays with values in the "values" arrays
    def set_old_with_values(self):
        for i in range(self.n_updrafts):
            for k in range(self.Gr.nzg):
                self.W.old[i,k] = self.W.values[i,k]
                self.Area.old[i,k] = self.Area.values[i,k]
                self.QT.old[i,k] = self.QT.values[i,k]
                self.QL.old[i,k] = self.QL.values[i,k]
                self.QR.old[i,k] = self.QR.values[i,k]
                self.H.old[i,k] = self.H.values[i,k]
                self.THL.old[i,k] = self.THL.values[i,k]
                self.T.old[i,k] = self.T.values[i,k]
                self.B.old[i,k] = self.B.values[i,k]
        return
    # quick utility to set "tmp" arrays with values in the "new" arrays
    def set_values_with_new(self):
        for i in range(self.n_updrafts):
            for k in range(self.Gr.nzg):
                self.W.values[i,k] = self.W.new[i,k]
                self.Area.values[i,k] = self.Area.new[i,k]
                self.QT.values[i,k] = self.QT.new[i,k]
                self.QL.values[i,k] = self.QL.new[i,k]
                self.QR.values[i,k] = self.QR.new[i,k]
                self.H.values[i,k] = self.H.new[i,k]
                self.THL.values[i,k] = self.THL.new[i,k]
                self.T.values[i,k] = self.T.new[i,k]
                self.B.values[i,k] = self.B.new[i,k]
        return


    def io(self, Stats):

        Stats.write_profile_new('updraft_area'       , self.Gr, self.Area.bulkvalues)
        Stats.write_profile_new('updraft_w'          , self.Gr, self.W.bulkvalues)
        Stats.write_profile_new('updraft_qt'         , self.Gr, self.QT.bulkvalues)
        Stats.write_profile_new('updraft_ql'         , self.Gr, self.QL.bulkvalues)
        Stats.write_profile_new('updraft_qr'         , self.Gr, self.QR.bulkvalues)
        if self.H.name == 'thetal':
            Stats.write_profile_new('updraft_thetal' , self.Gr, self.H.bulkvalues)
        else:
            Stats.write_profile_new('updraft_s'      , self.Gr, self.H.bulkvalues)
            #Stats.write_profile_new('updraft_thetal', self.Gr, self.THL.bulkvalues)
        Stats.write_profile_new('updraft_temperature', self.Gr, self.T.bulkvalues)
        Stats.write_profile_new('updraft_buoyancy'   , self.Gr, self.B.bulkvalues)
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
            self.cloud_base[i] = self.Gr.z_half[self.Gr.nzg-self.Gr.gw-1]
            self.cloud_top[i] = 0.0
            self.cloud_cover[i] = 0.0
            for k in range(self.Gr.gw,self.Gr.nzg-self.Gr.gw):
                if self.QL.values[i,k] > 1e-8 and self.Area.values[i,k] > 1e-3:
                    self.cloud_base[i] = np.fmin(self.cloud_base[i], self.Gr.z_half[k])
                    self.cloud_top[i] = np.fmax(self.cloud_top[i], self.Gr.z_half[k])
                    self.cloud_cover[i] = np.fmax(self.cloud_cover[i], self.Area.values[i,k])
        return

class UpdraftThermodynamics:
    def __init__(self, n_updraft, Gr, Ref, UpdVar):
        self.Gr = Gr
        self.Ref = Ref
        self.n_updraft = n_updraft
        if UpdVar.H.name == 's':
            self.t_to_prog_fp = t_to_entropy_c
            self.prog_to_t_fp = eos_first_guess_entropy
        elif UpdVar.H.name == 'thetal':
            self.t_to_prog_fp = t_to_thetali_c
            self.prog_to_t_fp = eos_first_guess_thetal

        return
    def satadjust(self, UpdVar):
        #Update T, QL
        for i in range(self.n_updraft):
            for k in range(self.Gr.nzg):
                T, ql = eos(self.t_to_prog_fp, self.prog_to_t_fp, self.Ref.p0_half[k], UpdVar.QT.values[i,k], UpdVar.H.values[i,k])
                UpdVar.QL.values[i,k] = ql
                UpdVar.T.values[i,k] = T
        return

    def buoyancy(self,  UpdVar, EnvVar,GMV, extrap):
        gw = self.Gr.gw
        UpdVar.Area.bulkvalues = np.sum(UpdVar.Area.values,axis=0)
        if not extrap:
            for i in range(self.n_updraft):
                for k in range(self.Gr.nzg):
                    qv = UpdVar.QT.values[i,k] - UpdVar.QL.values[i,k]
                    alpha = alpha_c(self.Ref.p0_half[k], UpdVar.T.values[i,k], UpdVar.QT.values[i,k], qv)
                    UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha) #- GMV.B.values[k]
        else:
            for i in range(self.n_updraft):
                for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
                    if UpdVar.Area.values[i,k] > 1e-3:
                        qt = UpdVar.QT.values[i,k]
                        qv = UpdVar.QT.values[i,k] - UpdVar.QL.values[i,k]
                        h = UpdVar.H.values[i,k]
                        t = UpdVar.T.values[i,k]
                        alpha = alpha_c(self.Ref.p0_half[k], t, qt, qv)
                        UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)

                    else:
                        T, q_l = eos(self.t_to_prog_fp,self.prog_to_t_fp, self.Ref.p0_half[k], qt, h)
                        qt -= q_l
                        qv = qt
                        t = T
                        alpha = alpha_c(self.Ref.p0_half[k], t, qt, qv)
                        UpdVar.B.values[i,k] = buoyancy_c(self.Ref.alpha0_half[k], alpha)
        for k in range(self.Gr.gw, self.Gr.nzg-self.Gr.gw):
            GMV.B.values[k] = (1.0 - UpdVar.Area.bulkvalues[k]) * EnvVar.B.values[k]
            for i in range(self.n_updraft):
                GMV.B.values[k] += UpdVar.Area.values[i,k] * UpdVar.B.values[i,k]
            for i in range(self.n_updraft):
                UpdVar.B.values[i,k] -= GMV.B.values[k]
            EnvVar.B.values[k] -= GMV.B.values[k]

        return


#Implements a simple "microphysics" that clips excess humidity above a user-specified level
class UpdraftMicrophysics:
    def __init__(self, paramlist, n_updraft, Gr, Ref):
        self.Gr = Gr
        self.Ref = Ref
        self.n_updraft = n_updraft
        self.max_supersaturation = paramlist['turbulence']['updraft_microphysics']['max_supersaturation']
        self.prec_source_h = np.zeros((n_updraft, Gr.nzg), dtype=np.double, order='c')
        self.prec_source_qt = np.zeros((n_updraft, Gr.nzg), dtype=np.double, order='c')
        self.prec_source_h_tot  = Field.half(Gr)
        self.prec_source_qt_tot = Field.half(Gr)
        return

    def compute_sources(self, UpdVar):
        """
        Compute precipitation source terms for QT, QR and H
        """
        for i in range(self.n_updraft):
            for k in range(self.Gr.nzg):
                tmp_qr = acnv_instant(UpdVar.QL.values[i,k], UpdVar.QT.values[i,k], self.max_supersaturation,\
                                      UpdVar.T.values[i,k], self.Ref.p0_half[k])
                self.prec_source_qt[i,k] = -tmp_qr
                self.prec_source_h[i,k]  = rain_source_to_thetal(self.Ref.p0_half[k], UpdVar.T.values[i,k],\
                                             UpdVar.QT.values[i,k], UpdVar.QL.values[i,k], 0.0, tmp_qr)
                                                                                          #TODO assumes no ice
        self.prec_source_h_tot  = np.sum(np.multiply(self.prec_source_h,  UpdVar.Area.values), axis=0)
        self.prec_source_qt_tot = np.sum(np.multiply(self.prec_source_qt, UpdVar.Area.values), axis=0)

        return

    def update_updraftvars(self, UpdVar):
        """
        Apply precipitation source terms to QL, QR and H
        """
        for i in range(self.n_updraft):
            for k in range(self.Gr.nzg):
                UpdVar.QT.values[i,k] += self.prec_source_qt[i,k]
                UpdVar.QL.values[i,k] += self.prec_source_qt[i,k]
                UpdVar.QR.values[i,k] -= self.prec_source_qt[i,k]
                UpdVar.H.values[i,k] += self.prec_source_h[i,k]
        return

    def compute_update_combined_local_thetal(self, p0, T, qt, ql, qr, h, i, k):

        # Language note: array indexing must be used to dereference pointers in Cython. * notation (C-style dereferencing)
        # is reserved for packing tuples

        tmp_qr = acnv_instant(ql[i,k], qt[i,k], self.max_supersaturation, T[i,k], p0[k])
        self.prec_source_qt[i,k] = -tmp_qr
        self.prec_source_h[i,k]  = rain_source_to_thetal(p0[k], T[i,k], qt[i,k], ql[i,k], 0.0, tmp_qr)
                                                                             #TODO - assumes no ice
        qt[i,k] += self.prec_source_qt[i,k]
        ql[i,k] += self.prec_source_qt[i,k]
        qr[i,k] -= self.prec_source_qt[i,k]
        h[i,k]  += self.prec_source_h[i,k]

        return
