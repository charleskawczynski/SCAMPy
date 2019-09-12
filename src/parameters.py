from PlanetParameters import *

#Adapated from PyCLES: https://github.com/pressel/pycles
pi = 3.14159265359
g = 9.80665
Rd = 287.1
Rv = 461.5
eps_v = 0.62210184182   # Rd / Rv
eps_vi = 1.60745384883  # Rv / Rd
cpd = 1004.0
cpv = 1859.0
cl = 4218.0
ci = 2106.0
kappa = 0.285956175299
Tf = 273.15
Tt = 273.16
T_tilde = 298.15
p_tilde = 100000.0
pv_star_t = 611.7
sd_tilde = 6864.8
sv_tilde = 10513.6
omega = 7.29211514671e-05
ql_threshold = 1e-08
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
