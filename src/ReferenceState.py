from Grid import Grid
from Field import Field
from NetCDFIO import NetCDFIO_Stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from thermodynamic_functions import t_to_entropy_c, eos_first_guess_entropy, eos, alpha_c
from parameters import *

class ReferenceState:
    def __init__(self, Gr ):
        self.p0          = Field.full(Gr)
        self.p0_half     = Field.half(Gr)
        self.alpha0      = Field.full(Gr)
        self.alpha0_half = Field.half(Gr)
        self.rho0        = Field.full(Gr)
        self.rho0_half   = Field.half(Gr)
        return

    def initialize(self, Gr, Stats):
        '''
        Initilize the reference profiles. The function is typically called from the case
        specific initialization fucntion defined in Initialization.pyx
        :param Gr: Grid class
        :param Thermodynamics: Thermodynamics class
        :param NS: StatsIO class
        :param Pa:  ParallelMPI class
        :return:
        '''

        self.sg = t_to_entropy_c(self.Pg, self.Tg, self.qtg, 0.0, 0.0)


        # Form a right hand side for integrating the hydrostatic equation to
        # determine the reference pressure
        def rhs(p, z):
            T, q_l = eos(t_to_entropy_c, eos_first_guess_entropy, np.exp(p),  self.qtg, self.sg)
            q_i = 0.0
            return -g / (Rd * T * (1.0 - self.qtg + eps_vi * (self.qtg - q_l - q_i)))

        # Construct arrays for integration points

        z = Gr.z_full_real()
        z_half = Gr.z_half_real()

        # We are integrating the log pressure so need to take the log of the
        # surface pressure
        p0 = np.log(self.Pg)

        p = Field.full(Gr)
        p_half = Field.half(Gr)

        # Perform the integration
        p[Gr.k_full_real()] = odeint(rhs, p0, z, hmax=1.0)[:, 0]
        p_half[Gr.k_half_real()] = odeint(rhs, p0, z_half, hmax=1.0)[:, 0]

        p_half.apply_Neumann(Gr, 0.0)
        p.apply_Neumann(Gr, 0.0)

        # Set boundary conditions
        p[:] = np.exp(p[:])
        p_half[:] = np.exp(p_half[:])

        p_ = p
        p_half_ = p_half
        temperature = Field.full(Gr)
        temperature_half = Field.half(Gr)
        alpha = Field.full(Gr)
        alpha_half = Field.half(Gr)

        ql = Field.full(Gr)
        qi = Field.full(Gr)
        qv = Field.full(Gr)

        ql_half = Field.half(Gr)
        qi_half = Field.half(Gr)
        qv_half = Field.half(Gr)

        # Compute reference state thermodynamic profiles
        # for k in Gr.over_points_full():
        for k in Gr.over_points_full_real():
            temperature[k], ql[k] = eos(t_to_entropy_c, eos_first_guess_entropy, p_[k], self.qtg, self.sg)
            qv[k] = self.qtg - (ql[k] + qi[k])
            alpha[k] = alpha_c(p_[k], temperature[k], self.qtg, qv[k])

        # for k in Gr.over_points_half():
        for k in Gr.over_points_half_real():
            temperature_half[k], ql_half[k] = eos(t_to_entropy_c, eos_first_guess_entropy, p_half_[k], self.qtg, self.sg)
            qv_half[k] = self.qtg - (ql_half[k] + qi_half[k])
            alpha_half[k] = alpha_c(p_half_[k], temperature_half[k], self.qtg, qv_half[k])

        # Now do a sanity check to make sure that the Reference State entropy profile is uniform following
        # saturation adjustment
        for k in Gr.over_points_half():
            s = t_to_entropy_c(p_half[k],temperature_half[k],self.qtg,ql_half[k],qi_half[k])
            if np.abs(s - self.sg)/self.sg > 0.01:
                print('Error in reference profiles entropy not constant !')
                print('Likely error in saturation adjustment')

        self.alpha0_half[:] = alpha_half[:]
        self.alpha0[:] = alpha[:]
        self.p0[:] = p_[:]
        self.p0_half[:] = p_half[:]
        self.rho0[:] = 1.0 / np.array(self.alpha0[:])
        self.rho0_half[:] = 1.0 / np.array(self.alpha0_half[:])

        self.alpha0_half.extrap(Gr)
        self.alpha0.extrap(Gr)
        self.p0.extrap(Gr)
        self.p0_half.extrap(Gr)
        self.rho0.extrap(Gr)
        self.rho0_half.extrap(Gr)

        plt.plot(self.p0.values         , Gr.z); plt.savefig(Stats.outpath+'p0.png'         ); plt.close()
        plt.plot(self.p0_half.values    , Gr.z); plt.savefig(Stats.outpath+'p0_half.png'    ); plt.close()
        plt.plot(self.rho0.values       , Gr.z); plt.savefig(Stats.outpath+'rho0.png'       ); plt.close()
        plt.plot(self.rho0_half.values  , Gr.z); plt.savefig(Stats.outpath+'rho0_half.png'  ); plt.close()
        plt.plot(self.alpha0.values     , Gr.z); plt.savefig(Stats.outpath+'alpha0.png'     ); plt.close()
        plt.plot(self.alpha0_half.values, Gr.z); plt.savefig(Stats.outpath+'alpha0_half.png'); plt.close()

        Stats.add_reference_profile('alpha0')
        Stats.write_reference_profile('alpha0', alpha[Gr.gw:-Gr.gw])
        Stats.add_reference_profile('alpha0_half')
        Stats.write_reference_profile('alpha0_half', alpha_half[Gr.gw:-Gr.gw])

        Stats.add_reference_profile('p0')
        Stats.write_reference_profile('p0', p_[Gr.gw:-Gr.gw])
        Stats.add_reference_profile('p0_half')
        Stats.write_reference_profile('p0_half', p_half[Gr.gw:-Gr.gw])

        Stats.add_reference_profile('rho0')
        Stats.write_reference_profile('rho0', 1.0 / np.array(alpha[Gr.gw:-Gr.gw]))
        Stats.add_reference_profile('rho0_half')
        Stats.write_reference_profile('rho0_half', 1.0 / np.array(alpha_half[Gr.gw:-Gr.gw]))

        return

