from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid, DualCut
from Field import Field, Full, Half, Dirichlet, Neumann, nice_name
from NetCDFIO import NetCDFIO_Stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from funcs_thermo import t_to_entropy_c, eos, eos_entropy, alpha_c
from parameters import *

def initialize_ref_state(grid, Stats, p_0, ρ_0, α_0, loc, sg, Pg, Tg, qtg):
    sg = t_to_entropy_c(Pg, Tg, qtg, 0.0, 0.0)
    # Form a right hand side for integrating the hydrostatic equation to
    # determine the reference pressure
    def rhs(p, z):
        T, q_l = eos_entropy(np.exp(p),  qtg, sg)
        q_i = 0.0
        R_m = Rd * (1.0 - qtg + eps_vi * (qtg - q_l - q_i))
        return -g / (R_m * T)

    # Construct arrays for integration points
    z_full = [grid.z[k] for k in grid.over_elems_real(Node())]
    z_half = [grid.z_half[k] for k in grid.over_elems_real(Center())]
    z = z_full if isinstance(loc, Node) else z_half

    # We are integrating the log pressure so need to take the log of the
    # surface pressure
    q_liq = Field.field(grid, loc)
    q_ice = Field.field(grid, loc)
    q_vap = Field.field(grid, loc)
    temperature = Field.field(grid, loc)

    p0 = np.log(Pg)
    p_0[grid.slice_real(loc)] = odeint(rhs, p0, z, hmax=1.0)[:, 0]
    p_0.apply_Neumann(grid, 0.0)
    p_0[:] = np.exp(p_0[:])

    # Compute reference state thermodynamic profiles
    for k in grid.over_elems_real(loc):
        temperature[k], q_liq[k] = eos_entropy(p_0[k], qtg, sg)
        q_vap[k] = qtg - (q_liq[k] + q_ice[k])
        α_0[k] = alpha_c(p_0[k], temperature[k], qtg, q_vap[k])
        ρ_0[k] = 1.0/α_0[k]

    # Sanity check: make sure Reference State entropy is uniform
    for k in grid.over_elems(loc):
        s = t_to_entropy_c(p_0[k], temperature[k], qtg, q_liq[k], q_ice[k])
        if np.abs(s - sg)/sg > 0.01:
            print('Error in reference profiles entropy not constant !')
            print('Likely error in saturation adjustment')

    α_0.extrap(grid)
    p_0.extrap(grid)
    ρ_0.extrap(grid)

    p_0_name = nice_name('p_0')+str(loc.__class__.__name__)
    ρ_0_name = nice_name('ρ_0')+str(loc.__class__.__name__)
    α_0_name = nice_name('α_0')+str(loc.__class__.__name__)

    plt.plot(p_0.values, grid.z); plt.title(p_0_name+' vs z')
    plt.xlabel(p_0_name); plt.ylabel('z')
    plt.savefig(Stats.figpath+p_0_name+'.png'); plt.close()

    plt.plot(ρ_0.values, grid.z); plt.title(ρ_0_name+' vs z')
    plt.xlabel(ρ_0_name); plt.ylabel('z')
    plt.savefig(Stats.figpath+ρ_0_name+'.png'); plt.close()

    plt.plot(α_0.values, grid.z); plt.title(α_0_name+' vs z')
    plt.xlabel(α_0_name); plt.ylabel('z')
    plt.savefig(Stats.figpath+α_0_name+'.png'); plt.close()

    p_0.export_data(grid, Stats.outpath+p_0_name+'.dat')
    ρ_0.export_data(grid, Stats.outpath+ρ_0_name+'.dat')
    α_0.export_data(grid, Stats.outpath+α_0_name+'.dat')

    k_1 = grid.boundary(Zmin())
    k_2 = grid.boundary(Zmax())

    Stats.add_reference_profile(p_0_name)
    Stats.write_reference_profile(p_0_name, α_0[k_1:k_2])
    Stats.add_reference_profile(ρ_0_name)
    Stats.write_reference_profile(ρ_0_name, p_0[k_1:k_2])
    Stats.add_reference_profile(α_0_name)
    Stats.write_reference_profile(α_0_name, ρ_0[k_1:k_2])
    return


class ReferenceState:
    def __init__(self, grid):
        return

    def initialize(self, grid, Stats, tmp):
        self.sg = t_to_entropy_c(self.Pg, self.Tg, self.qtg, 0.0, 0.0)
        initialize_ref_state(grid, Stats, tmp['p_0']     , tmp['ρ_0']     , tmp['α_0']     , Node()  , self.sg, self.Pg, self.Tg, self.qtg)
        initialize_ref_state(grid, Stats, tmp['p_0_half'], tmp['ρ_0_half'], tmp['α_0_half'], Center(), self.sg, self.Pg, self.Tg, self.qtg)
        return
