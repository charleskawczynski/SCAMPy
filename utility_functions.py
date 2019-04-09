import numpy as np
import scipy.special as sp
# from libc.math import exp, log
from scipy.stats import norm

# compute the mean of the values above a given percentile (0 to 1) for a standard normal distribution
# this gives the surface scalar coefficient for a single updraft or nth updraft of n updrafts
def percentile_mean_norm(percentile, nsamples):
    x = norm.rvs(size=nsamples)
    xp = norm.ppf(percentile)
    return np.ma.mean(np.ma.masked_less(x,xp))

# compute the mean of the values between two percentiles (0 to 1) for a standard normal distribution
# this gives the surface scalar coefficients for 1 to n-1 updrafts when using n updrafts
def percentile_bounds_mean_norm(low_percentile, high_percentile, nsamples):
    x = norm.rvs(size=nsamples)
    xp_low = norm.ppf(low_percentile)
    xp_high = norm.ppf(high_percentile)
    return np.ma.mean(np.ma.masked_greater(np.ma.masked_less(x,xp_low),xp_high))

def interp2pt(val1, val2):
    return 0.5*(val1 + val2)

def logistic(x, slope, mid):
    return 1.0/(1.0 + exp( -slope * (x-mid)))
