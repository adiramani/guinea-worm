from math import inf, log, pi, sqrt
import numpy as np

from scipy.special import (
    logsumexp,
    beta as beta_func
)


def calc_log_likelihood(sses, sigma, total_data_points):
    """
    @param sses: list of dictionaries containing sse summaries
    @param sigma: noise scale (std dev, > 0)
    @param total_data_points: number of time points we are comparing to for sse calculation
    """
    loglikes = []
    num_runs = len(sses)
    for sse in sses:            
        # Sum of squared errors (SSE)
        sse_val = sse["sse"]
        
        # Gaussian log-likelihood for this run
        ll = -(total_data_points/2) * log(2 * pi * sigma**2) - (sse_val / (2 * sigma**2))
        loglikes.append(ll)
    
    return logsumexp(loglikes) - log(num_runs)


def log_prior_gamma(value, median, sigma):
    if value <= 0:
        return -np.inf
    mu = np.log(median) # median of ~ 2
    return -0.5*((np.log(value)-mu)/sigma)**2 - np.log(value*sigma*np.sqrt(2*np.pi))

def log_prior_beta(value, alpha=2, beta=2):
    if value <= 0 or value >= 1:
        return -np.inf
    return (alpha-1)*np.log(value) + (beta-1)*np.log(1-value) - np.log(beta_func(alpha, beta))

def log_prior(thetas, sigma):
    r0, r0_s_w, nh_nc, exp_het = thetas
    lp = 0.0
    
    lp += log_prior_gamma(r0, median=2.0, sigma=0.7)
    lp += log_prior_beta(r0_s_w)
    lp += log_prior_beta(nh_nc)
    lp += log_prior_beta(exp_het)

    # Prior on sigma (Half-Normal(1.0))
    if sigma > 0:
        lp += -0.5 * (sigma**2) - log(sqrt(2*pi))  # Approx
    else:
        return -inf  # sigma must be positive
    
    return lp
