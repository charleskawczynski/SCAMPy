import numpy as np
import scipy.special as sp
from scipy.stats import norm

# compute the mean of the values between two percentiles (0 to 1) for a standard normal distribution
# this gives the surface scalar coefficients for 1 to n-1 updrafts when using n updrafts
def percentile_bounds_mean_norm(low_percentile, high_percentile, nsamples):
    x = norm.rvs(size=nsamples)
    xp_low = norm.ppf(low_percentile)
    xp_high = norm.ppf(high_percentile)
    x = np.ma.masked_less(x,xp_low)
    x = np.ma.masked_greater(x,xp_high)
    return np.ma.mean(x)
