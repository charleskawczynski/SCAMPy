from Grid import Grid, Zmin, Zmax, Center, Node, Cut, Dual, Mid
from Field import Field, Full, Half, Dirichlet, Neumann, nice_name
from NetCDFIO import NetCDFIO_Stats
import numpy as np
from PlanetParameters import *
from MoistThermodynamics import *
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def initialize_ref_state(grid, Stats, p_0, ρ_0, α_0, loc, Pg, Tg, qtg):

    q_pt_g = PhasePartitionRaw(qtg)
    θ_liq_ice_g = liquid_ice_pottemp_raw(Tg, Pg, q_pt_g)
    logp = np.log(Pg)

    def tendencies(p, z):
        expp_arr = np.exp(p)
        expp = expp_arr[0]
        ρ = air_density_raw(Tg, expp, q_pt_g)
        ts = LiquidIcePotTempSHumEquil(θ_liq_ice_g, qtg, ρ, expp)
        # print(ts)
        R_m = gas_constant_air(ts)
        T = air_temperature(ts)
        return - grav / (T * R_m)

    # Construct arrays for integration points
    z_full = [grid.z[k] for k in grid.over_elems_real(Node())]
    z_half = [grid.z_half[k] for k in grid.over_elems_real(Center())]
    z = z_full if isinstance(loc, Node) else z_half

    # p_0[grid.slice_real(loc)] = odeint(tendencies, logp, z, rtol=1e-12, atol=1e-12, printmessg=True)[:, 0]
    p_0[grid.slice_real(loc)] = odeint(tendencies, logp, z, rtol=1e-12, atol=1e-12)[:, 0]
    p_0.apply_Neumann(grid, 0.0)
    p_0[:] = np.exp(p_0[:])

    # Compute reference state thermodynamic profiles
    for k in grid.over_elems_real(loc):
        ts = TemperatureSHumEquil(Tg, qtg, p_0[k])
        ρ_0[k] = air_density(ts)
        α_0[k] = 1.0/ρ_0[k]

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
        initialize_ref_state(grid, Stats, tmp['p_0'], tmp['ρ_0'], tmp['α_0'], Center(), self.Pg, self.Tg, self.qtg)
        return
