from RootSolvers import *
import numpy as np
from PlanetParameters import *

class PhasePartition:
    def __init__(self, tot, liq=0.0, ice=0.0):
        self.tot = tot
        self.liq = liq
        self.ice = ice
        return
q_pt0=PhasePartition(0.0,0.0,0.0)

class ThermodynamicState:
    def __init__(self, e_int, q_tot, ρ, T):
        self.e_int = e_int
        self.q_tot = q_tot
        self.ρ = ρ
        self.T = T
        return

    def __str__(self):
        s = ''
        s+='e_int = '+str(self.e_int)+'\n'
        s+='q_tot = '+str(self.q_tot)+'\n'
        s+='ρ     = '+str(self.ρ)+'\n'
        s+='T     = '+str(self.T)+'\n'
        return s

class PhaseEquil(ThermodynamicState):
    def __init__(self, e_int, q_tot, ρ, T):
        self.e_int = e_int
        self.q_tot = q_tot
        self.ρ = ρ
        self.T = T
        return

def GetPhaseEquil(e_int, q_tot, ρ):
    return PhaseEquil(e_int, q_tot, ρ, saturation_adjustment(e_int, ρ, q_tot))

def LiquidIcePotTempSHumEquil(θ_liq_ice, q_tot, ρ, p):
    T = saturation_adjustment_q_tot_θ_liq_ice(θ_liq_ice, q_tot, ρ, p)
    print('T = ',T)
    q = PhasePartition_equil(T, ρ, q_tot)
    e_int = internal_energy_raw(T, q)
    return PhaseEquil(e_int, q_tot, ρ, T)

def LiquidIcePotTempSHumEquil_no_ρ(θ_liq_ice, q_tot, p):
    q_pt_dry = PhasePartition(q_tot)
    T_dry = θ_liq_ice * exner_raw(p, q_pt_dry)
    ρ_dry = air_density_raw(T_dry, p, q_pt_dry)
    T = saturation_adjustment_q_tot_θ_liq_ice(θ_liq_ice, q_tot, ρ_dry, p)
    ρ = air_density_raw(T, p, q_pt_dry)
    q_pt = PhasePartition_equil(T, ρ, q_tot)
    e_int = internal_energy_raw(T, q_pt)
    return PhaseEquil(e_int, q_tot, ρ, T)

def LiquidIcePotTempSHumEquil_no_ρ_pt(θ_liq_ice, q_pt, p):
    T = θ_liq_ice * exner_raw(p, q_pt)
    ρ = air_density_raw(T, p, q_pt)
    e_int = internal_energy_raw(T, q_pt)
    return PhaseEquil(e_int, q_pt.tot, ρ, T)

def TemperatureSHumEquil(T, q_tot, p):
    ρ = air_density_raw(T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(T, ρ, q_tot)
    e_int = internal_energy_raw(T, q)
    return PhaseEquil(e_int, q_tot, ρ, T)

##### Functions
def gas_constant_air_raw(q):
    return R_d * ( 1.0 +  (molmass_ratio - 1.0)*q.tot - molmass_ratio*(q.liq + q.ice) )

def gas_constant_air(ts):
    return gas_constant_air_raw(PhasePartition(ts))

def air_pressure_raw(T, ρ, q):
    return gas_constant_air_raw(q) * ρ * T

def air_pressure(ts):
    return air_pressure_raw(air_temperature(ts), air_density(ts), PhasePartition(ts))

def air_density_raw(T, p, q):
    return p / (gas_constant_air_raw(q) * T)

def air_density(ts):
    return ts.ρ

def specific_volume_raw(T, p, q):
    return (gas_constant_air_raw(q) * T) / p

def specific_volume(ts):
    return specific_volume_raw(air_temperature(ts), air_pressure(ts), PhasePartition(ts))

def cp_m_raw(q=q_pt0):
    return cp_d + (cp_v - cp_d)*q.tot + (cp_l - cp_v)*q.liq + (cp_i - cp_v)*q.ice

def cp_m(ts):
    return cp_m_raw(PhasePartition(ts))

def cv_m_raw(q=q_pt0):
    return cv_d + (cv_v - cv_d)*q.tot + (cv_l - cv_v)*q.liq + (cv_i - cv_v)*q.ice

def cv_m(ts):
    return cv_m_raw(PhasePartition(ts))

def moist_gas_constants_raw(q=q_pt0):
    R_gas  = gas_constant_air_raw(q)
    cp = cp_m_raw(q)
    cv = cv_m_raw(q)
    γ = cp/cv
    return (R_gas, cp, cv, γ)

def moist_gas_constants(ts):
    return moist_gas_constants_raw(PhasePartition(ts))

def air_temperature_e(e_int, q=q_pt0):
    return T_0 + (e_int - (q.tot - q.liq) * e_int_v0 + q.ice * (e_int_v0 + e_int_i0)) / cv_m_raw(q)

def air_temperature(ts):
    if isinstance(ts, PhaseEquil): return ts.T
    else: return air_temperature_e(ts.e_int, PhasePartition(ts))

def internal_energy_raw(T, q):
    return cv_m_raw(q) * (T - T_0) + (q.tot - q.liq) * e_int_v0 - q.ice * (e_int_v0 + e_int_i0)

def internal_energy(ts):
    return ts.e_int

def internal_energy_sat_raw(T, ρ, q_tot):
    return internal_energy_raw(T, PhasePartition_equil(T, ρ, q_tot))

def internal_energy_sat(ts):
    return internal_energy_sat_raw(air_temperature(ts), air_density(ts), ts.q_tot)

def total_energy_raw(e_kin, e_pot, T, q):
    return e_kin + e_pot + internal_energy_raw(T, q)

def total_energy(e_kin, e_pot, ts):
    return internal_energy(ts) + e_kin + e_pot

def soundspeed_air_raw(T, q=q_pt0):
    γ   = cp_m_raw(q) / cv_m_raw(q)
    R_m = gas_constant_air_raw(q)
    return np.sqrt(γ*R_m*T)

def soundspeed_air(ts):
    return soundspeed_air_raw(air_temperature(ts), PhasePartition(ts))

def latent_heat_vapor_raw(T):
    return latent_heat_generic(T, LH_v0, cp_v - cp_l)

def latent_heat_vapor(ts):
    return latent_heat_vapor_raw(air_temperature(ts))

def latent_heat_sublim_raw(T):
    return latent_heat_generic(T, LH_s0, cp_v - cp_i)

def latent_heat_sublim(ts):
    return latent_heat_sublim_raw(air_temperature(ts))

def latent_heat_fusion_raw(T):
    return latent_heat_generic(T, LH_f0, cp_l - cp_i)

def latent_heat_fusion(ts):
    return latent_heat_fusion_raw(air_temperature(ts))

def latent_heat_generic(T, LH_0, Δcp):
    return LH_0 + Δcp * (T - T_0)

def saturation_vapor_pressure_raw_liq(T):
    return saturation_vapor_pressure_raw_liq(T, LH_v0, cp_v - cp_l)

def saturation_vapor_pressure(ts):
    return saturation_vapor_pressure_raw_liq(air_temperature(ts), LH_v0, cp_v - cp_l)

def saturation_vapor_pressure_raw_ice(T):
    return saturation_vapor_pressure_raw_ice(T, LH_s0, cp_v - cp_i)

def saturation_vapor_pressure(ts):
    return saturation_vapor_pressure_raw_ice(air_temperature(ts), LH_s0, cp_v - cp_i)

def saturation_vapor_pressure(T, LH_0, Δcp):
    return press_triple * (T/T_triple)**(Δcp/R_v) * np.exp( (LH_0 - Δcp*T_0)/R_v * (1.0 / T_triple - 1.0 / T) )

# def q_vap_saturation_generic(T, ρ; phase::Phase=Liquid()):
#     p_v_sat = saturation_vapor_pressure(T, phase)
#     return q_vap_saturation_from_pressure(T, ρ, p_v_sat)

def q_vap_saturation_raw(T, ρ, q=q_pt0):

    # get phase partitioning
    _liquid_frac = liquid_fraction_equil_raw(T, q)
    _ice_frac    = 1.0 - _liquid_frac

    # effective latent heat at T_0 and effective difference in isobaric specific
    # heats of the mixture
    LH_0    = _liquid_frac * LH_v0 + _ice_frac * LH_s0
    Δcp     = _liquid_frac * (cp_v - cp_l) + _ice_frac * (cp_v - cp_i)

    # saturation vapor pressure over possible mixture of liquid and ice
    p_v_sat = saturation_vapor_pressure(T, LH_0, Δcp)

    return q_vap_saturation_from_pressure(T, ρ, p_v_sat)

def q_vap_saturation(ts):
    return q_vap_saturation_raw(air_temperature(ts), air_density(ts), PhasePartition(ts))

def q_vap_saturation_from_pressure(T, ρ, p_v_sat):
    return np.min([1.0, p_v_sat / (ρ * R_v * T)])

def saturation_excess_raw(T, ρ, q=q_pt0):
    return np.max([0.0, q.tot - q_vap_saturation_raw(T, ρ, q)])

def saturation_excess(ts):
    return saturation_excess_raw(air_temperature(ts), air_density(ts), PhasePartition(ts))

def liquid_fraction_equil_raw(T, q=q_pt0):
    q_c = q.liq + q.ice     # condensate specific humidity
    if q_c > 0.0:
        return q.liq / q_c
    else:
        # For now: Heaviside def for partitioning into liquid and ice: all liquid
        # for T > T_freeze; all ice for T <= T_freeze
        return T > T_freeze

def liquid_fraction_equil(ts):
    return liquid_fraction_equil_raw(air_temperature(ts), PhasePartition(ts))

def PhasePartition_equil(T, ρ, q_tot):
    _liquid_frac = liquid_fraction_equil_raw(T)   # fraction of condensate that is liquid
    q_c   = saturation_excess_raw(T, ρ, PhasePartition(q_tot))   # condensate specific humidity
    q_liq = _liquid_frac * q_c  # liquid specific humidity
    q_ice = (1.0 - _liquid_frac) * q_c # ice specific humidity
    return PhasePartition(q_tot, q_liq, q_ice)

def GetPhasePartition(ts):
    return PhasePartition_equil(air_temperature(ts), air_density(ts), ts.q_tot)

def saturation_adjustment(e_int, ρ, q_tot):
    T_1 = np.max([T_min, air_temperature(e_int, PhasePartition(q_tot))]) # Assume all vapor
    q_v_sat = q_vap_saturation_raw(T_1, ρ)
    if q_tot <= q_v_sat: # If not saturated return T_1
        return T_1
    else: # If saturated, iterate
        # FIXME here: need to revisit bounds for saturation adjustment to guarantee bracketing of zero.
        T_2 = air_temperature(e_int, PhasePartition(q_tot, 0.0, q_tot)) # Assume all ice
        def eos(T):
            return internal_energy_sat_raw(T, ρ, q_tot) - e_int
        T, converged = find_zero(eos, T_1, T_2, SecantMethod(), 1e-3, 10)
        if not converged:
            raise ValueErorr('saturation adjustment did not converge')
        return T

def saturation_adjustment_q_tot_θ_liq_ice(θ_liq_ice, q_tot, ρ, p):
    T_1 = air_temperature_from_liquid_ice_pottemp(θ_liq_ice, p) # Assume all vapor
    q_v_sat = q_vap_saturation_raw(T_1, ρ)
    if q_tot <= q_v_sat: # If not saturated
        return T_1
    else:  # If saturated, iterate
        T_2 = air_temperature_from_liquid_ice_pottemp(θ_liq_ice, p, PhasePartition(q_tot, 0.0, q_tot)) # Assume all ice
        def eos(T):
            return θ_liq_ice - liquid_ice_pottemp_sat_raw(T, p, PhasePartition_equil(T, ρ, q_tot))
        T, converged = find_zero(eos, T_1, T_2, SecantMethod(), 1e-3, 10)
        if not converged:
            raise ValueErorr('saturation_adjustment_q_tot_θ_liq_ice did not converge')
        return T

def liquid_ice_pottemp_raw(T, p, q=q_pt0):
    # liquid-ice potential temperature, approximating latent heats
    # of phase transitions as constants
    return dry_pottemp_raw(T, p, q) * (1.0 - (LH_v0*q.liq + LH_s0*q.ice)/(cp_m_raw(q)*T))

def liquid_ice_pottemp(ts):
    return liquid_ice_pottemp_raw(air_temperature(ts), air_pressure(ts), PhasePartition(ts))

def dry_pottemp_raw(T, p, q):
    return T / exner_raw(p, q)

def dry_pottemp(ts):
    return dry_pottemp_raw(air_temperature(ts), air_pressure(ts), PhasePartition(ts))

def air_temperature_from_liquid_ice_pottemp(θ_liq_ice, p, q=q_pt0):
    return θ_liq_ice*exner_raw(p, q) + (LH_v0*q.liq + LH_s0*q.ice) / cp_m_raw(q)

def virtual_pottemp_raw(T, p, q=q_pt0):
    return gas_constant_air_raw(q) / R_d * dry_pottemp_raw(T, p, q)

def virtual_pottemp(ts):
    return virtual_pottemp_raw(air_temperature(ts), air_pressure(ts), PhasePartition(ts))

def liquid_ice_pottemp_sat_raw(T, p, q=q_pt0):
    ρ = air_density_raw(T, p, q)
    q_v_sat = q_vap_saturation_raw(T, ρ, q)
    return liquid_ice_pottemp_raw(T, p, PhasePartition(q_v_sat))

def liquid_ice_pottemp_sat(ts):
    return liquid_ice_pottemp_sat_raw(air_temperature(ts), air_pressure(ts), PhasePartition(ts))

def exner_raw(p, q=q_pt0):
    _R_m    = gas_constant_air_raw(q)
    _cp_m   = cp_m_raw(q)
    return (p/MSLP)**(_R_m/_cp_m)

def exner(ts):
    return exner_raw(air_pressure(ts), PhasePartition(ts))

def relative_humidity(ts):
    return air_pressure(ts)/saturation_vapor_pressure(ts, Liquid())
