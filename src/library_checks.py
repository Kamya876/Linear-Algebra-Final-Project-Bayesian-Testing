"""SciPy reference implementations -- used ONLY to check the from-scratch methods."""
import numpy as np
from scipy import integrate, optimize, stats


def lib_posterior_stats(a_post, b_post):
    return stats.beta.mean(a_post, b_post), stats.beta.var(a_post, b_post)


def lib_prob_b_greater_a(a_A, b_A, a_B, b_B):
    """P(pB > pA) = integral f_B(y) F_A(y) dy by adaptive quadrature."""
    mA, mB = a_A / (a_A + b_A), a_B / (a_B + b_B)
    val, _ = integrate.quad(lambda y: stats.beta.pdf(y, a_B, b_B) * stats.beta.cdf(y, a_A, b_A),
                            0.0, 1.0, epsabs=1e-12, limit=400, points=sorted({mA, mB}))
    return val


def lib_z_test(x_A, n_A, x_B, n_B):
    """Pooled z-test via the chi-square test of independence (no continuity correction).

    For a 2x2 table chi2 = z^2, so |z| = sqrt(chi2) and the p-values coincide.
    The sign of z is taken from the difference of proportions.
    """
    table = np.array([[x_A, n_A - x_A], [x_B, n_B - x_B]])
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    z = np.sign(x_B / n_B - x_A / n_A) * np.sqrt(chi2)
    return z, p


def lib_diff_cdf(d, a_A, b_A, a_B, b_B):
    """P(pB - pA <= d) = integral f_A(x) F_B(x + d) dx (exact, by quadrature)."""
    mA = a_A / (a_A + b_A)
    sd = np.sqrt(stats.beta.var(a_A, b_A))
    lo, hi = max(0.0, mA - 12 * sd), min(1.0, mA + 12 * sd)
    val, _ = integrate.quad(lambda x: stats.beta.pdf(x, a_A, b_A) * stats.beta.cdf(x + d, a_B, b_B),
                            lo, hi, epsabs=1e-12, limit=400, points=[mA])
    return val


def lib_credible_interval(a_A, b_A, a_B, b_B, level=0.95):
    """Exact equal-tailed interval for pB - pA by root-finding on the exact CDF."""
    alpha = 1 - level
    mean_d = a_B / (a_B + b_B) - a_A / (a_A + b_A)
    sd_d = np.sqrt(stats.beta.var(a_A, b_A) + stats.beta.var(a_B, b_B))
    lo_b, hi_b = mean_d - 8 * sd_d, mean_d + 8 * sd_d
    f = lambda d, t: lib_diff_cdf(d, a_A, b_A, a_B, b_B) - t
    lower = optimize.brentq(f, lo_b, hi_b, args=(alpha / 2,), xtol=1e-12)
    upper = optimize.brentq(f, lo_b, hi_b, args=(1 - alpha / 2,), xtol=1e-12)
    return lower, upper


def lib_ci_endpoint_tolerance(a_A, b_A, a_B, b_B, endpoint, M, level=0.95, k=3):
    """k Monte Carlo standard errors of a sample quantile of pB - pA with M draws.

    SE(q_hat) = sqrt(t (1 - t) / M) / f_D(q), f_D = density of pB - pA at the endpoint,
    obtained by a central difference of the exact CDF.
    """
    alpha = 1 - level
    t = alpha / 2 if endpoint < (a_B / (a_B + b_B) - a_A / (a_A + b_A)) else 1 - alpha / 2
    sd_d = np.sqrt(stats.beta.var(a_A, b_A) + stats.beta.var(a_B, b_B))
    h = sd_d * 1e-2
    f = (lib_diff_cdf(endpoint + h, a_A, b_A, a_B, b_B)
         - lib_diff_cdf(endpoint - h, a_A, b_A, a_B, b_B)) / (2 * h)
    return k * np.sqrt(t * (1 - t) / M) / f
