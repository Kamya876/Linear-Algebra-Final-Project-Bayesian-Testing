"""Shared calculations for Bayesian A/B testing (Beta-Binomial model).

Only numpy and scipy.stats are used. The z statistic, the posterior
parameters, the posterior moments and the Monte Carlo estimator are
implemented here from the derivations in docs/main.tex; scipy.stats is
used only for tail probabilities, quantiles and Beta densities.
"""
import numpy as np
from scipy import stats


def beta_posterior_parameters(a, b, successes, trials):
    """Posterior Beta shape parameters: (a + x, b + n - x)."""
    if not (a > 0 and b > 0):
        raise ValueError("Beta prior parameters a and b must be positive.")
    if not (0 <= successes <= trials):
        raise ValueError("Require 0 <= successes <= trials.")
    return a + successes, b + trials - successes


def beta_posterior_mean(a_post, b_post):
    """E[p | data] = alpha / (alpha + beta)."""
    return a_post / (a_post + b_post)


def beta_posterior_variance(a_post, b_post):
    """Var[p | data] = alpha*beta / ((alpha+beta)^2 (alpha+beta+1))."""
    total = a_post + b_post
    return (a_post * b_post) / (total**2 * (total + 1))


def weighted_posterior_mean(a, b, successes, trials):
    """Prior-mean / sample-rate weighted average (Question 3)."""
    w = trials / (a + b + trials)
    return (1 - w) * a / (a + b) + w * successes / trials, w


def exact_prob_b_greater_a(a_a, b_a, a_b, b_b, grid_size=200_001):
    """Numerical value of q = P(p_B > p_A | data) = int f_A(t) (1 - F_B(t)) dt.

    Used only as a deterministic reference for the Monte Carlo estimate.
    """
    t = np.linspace(0.0, 1.0, grid_size)
    integrand = stats.beta.pdf(t, a_a, b_a) * stats.beta.sf(t, a_b, b_b)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(integrand, t))


def two_proportion_z_test(x_a, n_a, x_b, n_b):
    """Two-sided pooled two-proportion z-test of H0: p_A = p_B."""
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Sample sizes must be positive.")
    if not (0 <= x_a <= n_a and 0 <= x_b <= n_b):
        raise ValueError("Success counts must be between zero and sample size.")
    p_a = x_a / n_a
    p_b = x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return {"p_a": p_a, "p_b": p_b, "pooled": p_pool,
                "se": se, "z": np.nan, "p_value": np.nan}
    z = (p_b - p_a) / se
    p_value = 2 * stats.norm.sf(abs(z))
    return {"p_a": p_a, "p_b": p_b, "pooled": p_pool,
            "se": se, "z": z, "p_value": p_value}


def theoretical_power(p_a, p_b, n_a, n_b, alpha=0.05):
    """Large-sample power of the pooled two-sided z-test at (p_A, p_B)."""
    p_bar = (n_a * p_a + n_b * p_b) / (n_a + n_b)
    se0 = np.sqrt(p_bar * (1 - p_bar) * (1 / n_a + 1 / n_b))
    se1 = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    zc = stats.norm.isf(alpha / 2)
    shift = (p_b - p_a)
    return float(stats.norm.sf((zc * se0 - shift) / se1)
                 + stats.norm.cdf((-zc * se0 - shift) / se1))
