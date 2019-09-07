
from PlanetParameters import *
from MoistThermodynamics import *

θ_liq_ice = 300.0
q_tot = 0.1
ρ = 1.0
p = MSLP*1.01

ts = LiquidIcePotTempSHumEquil(θ_liq_ice, q_tot, ρ, p)

print(ts)
