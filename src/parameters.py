from PlanetParameters import *

#Adapated from PyCLES: https://github.com/pressel/pycles
pi = 3.14159265359
g = grav
Rd = R_d
Rv = R_v
eps_v = Rd / Rv
eps_vi = Rv / Rd
cpd = cp_d
cpv = cp_v
cl = 4218.0
ci = 2106.0
kappa = kappa_d
Tf = 273.15
Tt = 273.16
T_tilde = 298.15
p_tilde = 100000.0
sd_tilde = 6864.8
sv_tilde = 10513.6
omega = 7.29211514671e-05
vkb = 0.4
Pr0 = 1.0
beta_m = 4.8
beta_h = 7.8
gamma_m = 15.0
gamma_h = 9.0
# constants defined in Stevens et al 2005 (that are different from scampy)
# needed for DYCOMS case setup
dycoms_cp = 1015.
dycoms_L = 2.47 * 1e6
dycoms_Rd = 287.
