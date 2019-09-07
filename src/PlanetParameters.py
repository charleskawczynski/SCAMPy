
# Physical constants
gas_constant      = 8.3144598                    #"Universal gas constant (J/mol/K)"
light_speed       = 2.99792458e8                 #"Speed of light in vacuum (m/s)"
h_Planck          = 6.626e-34                    #"Planck constant (m^2 kg/s)"
k_Boltzmann       = 1.381e-23                    #"Boltzmann constant (m^2 kg/s^2/K)"
Stefan            = 5.670e-8                     #"Stefan-Boltzmann constant (W/m^2/K^4)"
astro_unit        = 1.4959787e11                 #"Astronomical unit (m)"
k_Karman          = 0.4                          #"Von Karman constant (1)"

# Properties of dry air
molmass_dryair    = 28.97e-3                     #"Molecular weight dry air (kg/mol)"
R_d               = gas_constant/molmass_dryair  #"Gas constant dry air (J/kg/K)"
kappa_d           = 2/7                          #"Adiabatic exponent dry air"
cp_d              = R_d/kappa_d                  #"Isobaric specific heat dry air"
cv_d              = cp_d - R_d                   #"Isochoric specific heat dry air"

# Properties of water
ρ_cloud_liq       = 1e3                          #"Density of liquid water (kg/m^3)"
ρ_cloud_ice       = 9167                         #"Density of ice water (kg/m^3)"
molmass_water     = 18.01528e-3                  #"Molecular weight (kg/mol)"
molmass_ratio     = molmass_dryair/molmass_water #"Molar mass ratio dry air/water"
R_v               = gas_constant/molmass_water   #"Gas constant water vapor (J/kg/K)"
cp_v              = 1859                         #"Isobaric specific heat vapor (J/kg/K)"
cp_l              = 4181                         #"Isobaric specific heat liquid (J/kg/K)"
cp_i              = 2100                         #"Isobaric specific heat ice (J/kg/K)"
cv_v              = cp_v - R_v                   #"Isochoric specific heat vapor (J/kg/K)"
cv_l              = cp_l                         #"Isochoric specific heat liquid (J/kg/K)"
cv_i              = cp_i                         #"Isochoric specific heat ice (J/kg/K)"
T_freeze          = 273.15                       #"Freezing point temperature (K)"
T_min             = 150.0                        #"Minimum temperature guess in saturation adjustment (K)"
T_icenuc          = 233.00                       #"Homogeneous nucleation temperature (K)"
T_triple          = 273.16                       #"Triple point temperature (K)"
T_0               = T_triple                     #"Reference temperature (K)"
LH_v0             = 2.5008e6                     #"Latent heat vaporization at T_0 (J/kg)"
LH_s0             = 2.8344e6                     #"Latent heat sublimation at T_0 (J/kg)"
LH_f0             = LH_s0 - LH_v0                #"Latent heat of fusion at T_0 (J/kg)"
e_int_v0          = LH_v0 - R_v*T_0              #"Specific internal energy of vapor at T_0 (J/kg)"
e_int_i0          = LH_f0                        #"Specific internal energy of ice at T_0 (J/kg)"
press_triple      = 611.657                      #"Triple point vapor pressure (Pa)"

# Properties of sea water
ρ_ocean           = 1.035e3                      #"Reference density sea water (kg/m^3)"
cp_ocean          = 3989.25                      #"Specific heat sea water (J/kg/K)"

# Planetary parameters
planet_radius     = 6.371e6                      #"Mean planetary radius (m)"
day               = 86400                        #"Length of day (s)"
Omega             = 7.2921159e-5                 #"Ang. velocity planetary rotation (1/s)"
grav              = 9.81                         #"Gravitational acceleration (m/s^2)"
year_anom         = 365.26*day                   #"Length of anomalistic year (s)"
orbit_semimaj     = 1*astro_unit                 #"Length of semimajor orbital axis (m)"
TSI               = 1362                         #"Total solar irradiance (W/m^2)"
MSLP              = 1.01325e5                    #"Mean sea level pressure (Pa)"

