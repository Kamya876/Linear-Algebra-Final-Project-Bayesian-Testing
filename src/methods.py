"""From-scratch methods for Bayesian A/B testing (Project 7, Iteration 3).

Every function implements one numbered step of the Iteration 2 algorithm
(Section "Algorithm as mathematics") and uses NumPy / the standard library only.
SciPy is used elsewhere in the repo ONLY to check these functions.

Symbols follow the report: a, b = prior shapes; x, n = successes, trials;
alpha, beta = posterior shapes; M = number of Monte Carlo draws.
"""
from math import erfc, lgamma, sqrt

import numpy as np


# ---------------------------------------------------------------------------
# Step 2: conjugate update
# Equation:  alpha = a + x,   beta = b + n - x      (Beta-Binomial conjugacy)
# ---------------------------------------------------------------------------
def my_beta_posterior(x, n, a=1.0, b=1.0):
    """Return the Beta posterior parameters (alpha, beta).

    x: number of successes (clicks / returns), n: number of trials,
    a, b: Beta prior shapes.
    """
    if not (0 <= x <= n):
        raise ValueError("Successes must be between 0 and n.")
    if a <= 0 or b <= 0:
        raise ValueError("Prior parameters must be positive.")
    return a + x, b + n - x


# ---------------------------------------------------------------------------
# Step 3: posterior moments
# Equation:  E[p|x]   = alpha / s
#            Var[p|x] = alpha*beta / (s^2 (s+1)),   s = alpha + beta
# ---------------------------------------------------------------------------
def my_posterior_stats(a_post, b_post):
    """Mean and variance of a Beta(a_post, b_post) distribution."""
    s = a_post + b_post
    mean = a_post / s
    variance = (a_post * b_post) / (s ** 2 * (s + 1))
    return mean, variance


# ---------------------------------------------------------------------------
# Weighted-average identity (Question 3)
# Equation:  E[p|x] = (1 - w) * mu0 + w * (x/n),
#            w = n/(a+b+n),  mu0 = a/(a+b)
# ---------------------------------------------------------------------------
def my_weighted_posterior_mean(x, n, a=1.0, b=1.0):
    """Posterior mean written as prior mean and data mean with weights."""
    w = n / (a + b + n)
    return (1 - w) * (a / (a + b)) + w * (x / n)


# ---------------------------------------------------------------------------
# Step 5: Monte Carlo probability that B beats A
# Equation:  q_hat = (1/M) * sum_m 1{ pB^(m) > pA^(m) },
#            SE    = sqrt( q_hat (1 - q_hat) / M )
# ---------------------------------------------------------------------------
def my_mc_probability(a_A, b_A, a_B, b_B, M=100000, seed=0):
    """Estimate P(pB > pA | data); returns (q_hat, standard_error)."""
    rng = np.random.default_rng(seed)
    samples_A = rng.beta(a_A, b_A, size=M)
    samples_B = rng.beta(a_B, b_B, size=M)
    indicators = (samples_B > samples_A).astype(float)
    probability = float(np.mean(indicators))
    standard_error = float(np.sqrt(probability * (1 - probability) / M))
    return probability, standard_error


# ---------------------------------------------------------------------------
# Step 4: equal-tailed credible interval for the difference d = pB - pA
# Equation:  [ Q_{alpha/2}(d^(m)),  Q_{1-alpha/2}(d^(m)) ],  d^(m) = pB^(m) - pA^(m)
# ---------------------------------------------------------------------------
def my_credible_interval(a_A, b_A, a_B, b_B, M=200000, seed=0, level=0.95):
    """Equal-tailed credible interval for pB - pA from posterior draws."""
    if not 0 < level < 1:
        raise ValueError("Credible level must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    samples_A = rng.beta(a_A, b_A, size=M)
    samples_B = rng.beta(a_B, b_B, size=M)
    differences = samples_B - samples_A
    alpha = 1 - level
    lower = float(np.quantile(differences, alpha / 2))
    upper = float(np.quantile(differences, 1 - alpha / 2))
    return lower, upper


# ---------------------------------------------------------------------------
# Step 6: pooled two-proportion z-test
# Equation:  p_pool = (xA + xB)/(nA + nB)
#            SE0    = sqrt( p_pool (1 - p_pool) (1/nA + 1/nB) )
#            z      = (pB_hat - pA_hat) / SE0
#            p      = 2 (1 - Phi(|z|)) = erfc(|z| / sqrt 2)
# ---------------------------------------------------------------------------
def my_z_steps(x_A, n_A, x_B, n_B):
    """Return every intermediate quantity of the z-test as a dict."""
    if n_A <= 0 or n_B <= 0:
        raise ValueError("Group sizes must be positive.")
    if not (0 <= x_A <= n_A and 0 <= x_B <= n_B):
        raise ValueError("Invalid success counts.")
    p_A, p_B = x_A / n_A, x_B / n_B
    p_pool = (x_A + x_B) / (n_A + n_B)
    se0 = sqrt(p_pool * (1 - p_pool) * (1 / n_A + 1 / n_B))
    if se0 == 0:
        raise ValueError("Pooled standard error is zero.")
    z = (p_B - p_A) / se0
    p_value = erfc(abs(z) / sqrt(2))
    return {"p_A": p_A, "p_B": p_B, "p_pool": p_pool, "se0": se0,
            "z": z, "p_value": p_value}


def my_z_test(x_A, n_A, x_B, n_B):
    """Two-proportion z statistic and two-sided p-value: returns (z, p)."""
    steps = my_z_steps(x_A, n_A, x_B, n_B)
    return steps["z"], steps["p_value"]


# ---------------------------------------------------------------------------
# Beta density (used by the grid integration and the "integrates to 1" check)
# Equation:  f(p | a, b) = p^(a-1) (1-p)^(b-1) / B(a, b),
#            log B(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b)
# ---------------------------------------------------------------------------
def my_beta_pdf(x, a, b):
    """Beta(a, b) density evaluated on an array x in [0, 1]."""
    x = np.asarray(x, dtype=float)
    log_const = lgamma(a + b) - lgamma(a) - lgamma(b)
    out = np.zeros_like(x)
    interior = (x > 0) & (x < 1)
    t = x[interior]
    out[interior] = np.exp(log_const + (a - 1) * np.log(t) + (b - 1) * np.log1p(-t))
    # Boundary values (only non-zero / finite when the shape equals 1)
    if a == 1:
        out[x == 0] = np.exp(log_const) if b >= 1 else 0.0
    if b == 1:
        out[x == 1] = np.exp(log_const) if a >= 1 else 0.0
    return out


# ---------------------------------------------------------------------------
# Trapezoidal rule and the "posterior integrates to 1" second-way check
# Equation:  integral_0^1 f(p | alpha, beta) dp = 1
# ---------------------------------------------------------------------------
def my_trapezoid(y, x):
    """Trapezoidal rule for samples y on the grid x."""
    return float(np.sum((y[:-1] + y[1:]) * np.diff(x) / 2))


def my_posterior_total_mass(a_post, b_post, grid_size=10001):
    grid = np.linspace(0, 1, grid_size)
    return my_trapezoid(my_beta_pdf(grid, a_post, b_post), grid)


# ---------------------------------------------------------------------------
# Step 5 (deterministic version): numerical integration
# Equation:  P(pB > pA | data) = integral_0^1 fB(y) FA(y) dy,
#            FA(y) = integral_0^y fA(x) dx   (cumulative trapezoid)
# ---------------------------------------------------------------------------
def my_grid_probability(a_A, b_A, a_B, b_B, grid_size=10001, auto_window=False):
    """P(pB > pA | data) by cumulative trapezoidal integration on a grid.

    auto_window=False integrates over the whole interval [0, 1].
    auto_window=True integrates over mean +/- 12 posterior SDs of both arms
    (the mass outside is below 1e-30), which keeps the grid spacing small
    compared with the posterior width when n is large (e.g. Cookie Cats).
    """
    if min(a_A, b_A, a_B, b_B) <= 0:
        raise ValueError("Beta parameters must be positive.")
    if grid_size < 3:
        raise ValueError("Grid size must be at least 3.")
    lo, hi = 0.0, 1.0
    if auto_window:
        stats_ = [my_posterior_stats(a_A, b_A), my_posterior_stats(a_B, b_B)]
        lo = max(0.0, min(m - 12 * sqrt(v) for m, v in stats_))
        hi = min(1.0, max(m + 12 * sqrt(v) for m, v in stats_))
    grid = np.linspace(lo, hi, grid_size)
    dx = grid[1] - grid[0]
    pdf_A = my_beta_pdf(grid, a_A, b_A)
    pdf_B = my_beta_pdf(grid, a_B, b_B)
    cdf_A = np.zeros(grid_size)
    cdf_A[1:] = np.cumsum((pdf_A[:-1] + pdf_A[1:]) * dx / 2)
    return my_trapezoid(pdf_B * cdf_A, grid)


# ---------------------------------------------------------------------------
# Assumption A4: sample-ratio test for a planned 50/50 split
# Equation:  z = (nA - n/2) / sqrt(n/4),   p = erfc(|z| / sqrt 2)
# ---------------------------------------------------------------------------
def my_srm_test(n_A, n_B, share_A=0.5):
    """Sample-ratio-mismatch z-test: returns (z, two-sided p)."""
    n = n_A + n_B
    z = (n_A - n * share_A) / sqrt(n * share_A * (1 - share_A))
    return z, erfc(abs(z) / sqrt(2))


# ---------------------------------------------------------------------------
# Assumption A8: the normal approximation needs large expected counts
# Equation:  min_g { n_g p_pool, n_g (1 - p_pool) }  >= 10
# ---------------------------------------------------------------------------
def my_normal_approx_min_count(x_A, n_A, x_B, n_B):
    p_pool = (x_A + x_B) / (n_A + n_B)
    return min(n_A * p_pool, n_A * (1 - p_pool), n_B * p_pool, n_B * (1 - p_pool))


if __name__ == "__main__":
    a_A, b_A = my_beta_posterior(3, 10)
    a_B, b_B = my_beta_posterior(5, 10)
    mc_prob, mc_se = my_mc_probability(a_A, b_A, a_B, b_B, M=100000, seed=0)
    grid_prob = my_grid_probability(a_A, b_A, a_B, b_B)
    print("Monte Carlo probability:", round(mc_prob, 6))
    print("Numerical integration:  ", round(grid_prob, 6))
    print("Absolute difference:    ", round(abs(mc_prob - grid_prob), 6))
    print("Monte Carlo SE:         ", round(mc_se, 6))
