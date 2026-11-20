"""Iteration 3 -- small worked example, step by step (Project 7).

A: 3 clicks in 10 visits.  B: 5 clicks in 10 visits.  Uniform Beta(1,1) priors.
Every intermediate quantity is printed, the exact value is computed with
fractions.Fraction, and the result is checked a second way (the posterior
integrates to 1; Monte Carlo vs numerical integration vs adaptive quadrature).

Run:  python src/small_example.py
"""
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from library_checks import lib_posterior_stats, lib_prob_b_greater_a, lib_z_test  # check only
from methods import (my_beta_posterior, my_grid_probability, my_mc_probability,
                     my_posterior_stats, my_posterior_total_mass, my_weighted_posterior_mean,
                     my_z_steps)
from verify import VerificationLog, report

DATA = "Small example"


def exact_beta_moments(alpha, beta):
    """Exact mean and variance as Fractions: alpha/s and alpha*beta/(s^2 (s+1))."""
    alpha, beta = Fraction(alpha), Fraction(beta)
    s = alpha + beta
    return alpha / s, alpha * beta / (s ** 2 * (s + 1))


def main():
    log = VerificationLog("verification_small_example.csv")
    x_A, n_A, x_B, n_B = 3, 10, 5, 10

    print("=== STEP 1: OBSERVED DATA ===")
    print(f"A: {x_A} successes out of {n_A}  (p_hat_A = {Fraction(x_A, n_A)})")
    print(f"B: {x_B} successes out of {n_B}  (p_hat_B = {Fraction(x_B, n_B)})")

    print("\n=== STEP 2: BETA POSTERIORS (alpha = a + x, beta = b + n - x) ===")
    a_A, b_A = my_beta_posterior(x_A, n_A)
    a_B, b_B = my_beta_posterior(x_B, n_B)
    fa_A, fb_A = Fraction(1 + x_A), Fraction(1 + n_A - x_A)
    fa_B, fb_B = Fraction(1 + x_B), Fraction(1 + n_B - x_B)
    print(f"A: Beta({int(a_A)}, {int(b_A)})   exact: Beta({fa_A}, {fb_A})")
    print(f"B: Beta({int(a_B)}, {int(b_B)})   exact: Beta({fa_B}, {fb_B})")

    print("\n=== STEP 3: POSTERIOR MEAN AND VARIANCE ===")
    mean_A, var_A = my_posterior_stats(a_A, b_A)
    mean_B, var_B = my_posterior_stats(a_B, b_B)
    ex_mean_A, ex_var_A = exact_beta_moments(fa_A, fb_A)
    ex_mean_B, ex_var_B = exact_beta_moments(fa_B, fb_B)
    print(f"A: mean = {mean_A:.6f}  exact {ex_mean_A};  variance = {var_A:.6f}  exact {ex_var_A}")
    print(f"B: mean = {mean_B:.6f}  exact {ex_mean_B};  variance = {var_B:.6f}  exact {ex_var_B}")
    w = Fraction(n_A, 2 + n_A)
    print(f"Weighted-average check (A): w = n/(a+b+n) = {w}; "
          f"(1-w)*mu0 + w*p_hat = {(1 - w) * Fraction(1, 2) + w * Fraction(x_A, n_A)}")

    print("\n=== STEP 4: MONTE CARLO  q_hat = mean(1{pB > pA}),  M = 100,000 ===")
    mc_prob, mc_se = my_mc_probability(a_A, b_A, a_B, b_B, M=100000, seed=0)
    print(f"q_hat = {mc_prob:.6f},  SE = sqrt(q_hat(1-q_hat)/M) = {mc_se:.6f}")

    print("\n=== STEP 5: POOLED TWO-PROPORTION Z-TEST (every intermediate value) ===")
    st = my_z_steps(x_A, n_A, x_B, n_B)
    ex_pool = Fraction(x_A + x_B, n_A + n_B)
    ex_se2 = ex_pool * (1 - ex_pool) * (Fraction(1, n_A) + Fraction(1, n_B))
    ex_z2 = (Fraction(x_B, n_B) - Fraction(x_A, n_A)) ** 2 / ex_se2
    print(f"p_pool = (xA+xB)/(nA+nB) = {st['p_pool']:.6f}   exact {ex_pool}")
    print(f"SE0^2  = p_pool(1-p_pool)(1/nA+1/nB) = {st['se0'] ** 2:.6f}   exact {ex_se2}")
    print(f"SE0    = {st['se0']:.6f}")
    print(f"z      = (pB_hat - pA_hat)/SE0 = {st['z']:.6f}   (exact z^2 = {ex_z2})")
    print(f"two-sided p = erfc(|z|/sqrt 2) = {st['p_value']:.6f}")

    print("\n=== STEP 6: SECOND-WAY CHECKS ===")
    grid_prob = my_grid_probability(a_A, b_A, a_B, b_B, grid_size=10001)
    quad_prob = lib_prob_b_greater_a(a_A, b_A, a_B, b_B)
    mass_A = my_posterior_total_mass(a_A, b_A)
    mass_B = my_posterior_total_mass(a_B, b_B)
    print(f"Grid (trapezoid) P(B>A)       = {grid_prob:.10f}")
    print(f"SciPy adaptive quadrature     = {quad_prob:.10f}")
    print(f"Monte Carlo                   = {mc_prob:.10f}  ({abs(mc_prob - quad_prob) / mc_se:.3f} SE from quad)")
    print(f"Posterior A integrates to     = {mass_A:.12f}")
    print(f"Posterior B integrates to     = {mass_B:.12f}")

    print("\n=== VERIFICATION ROWS ===")
    log.compare(DATA, "posterior means A,B", "exact fractions",
                [mean_A, mean_B], [float(ex_mean_A), float(ex_mean_B)], 1e-15)
    log.compare(DATA, "posterior variances A,B", "exact fractions",
                [var_A, var_B], [float(ex_var_A), float(ex_var_B)], 1e-15)
    log.compare(DATA, "posterior means/variances", "scipy.stats.beta",
                [mean_A, var_A, mean_B, var_B],
                [*lib_posterior_stats(a_A, b_A), *lib_posterior_stats(a_B, b_B)], 1e-12)
    log.compare(DATA, "weighted-average posterior mean A", "direct posterior mean",
                my_weighted_posterior_mean(x_A, n_A), mean_A, 1e-15)
    log.compare(DATA, "z^2", "exact fraction 5/6", st["z"] ** 2, float(ex_z2), 1e-12)
    z_lib, p_lib = lib_z_test(x_A, n_A, x_B, n_B)
    log.compare(DATA, "z and p-value", "scipy chi2_contingency",
                [st["z"], st["p_value"]], [z_lib, p_lib], 1e-10)
    log.compare(DATA, "posterior integrates to 1 (A, B)", "1", [mass_A, mass_B], [1.0, 1.0], 1e-6)
    log.compare(DATA, "P(B>A) grid", "scipy.integrate.quad", grid_prob, quad_prob, 1e-6)
    log.within_se(DATA, "P(B>A) Monte Carlo", "scipy.integrate.quad", mc_prob, quad_prob, mc_se)
    report("grid vs quad", grid_prob, quad_prob)
    log.save()
    return log


if __name__ == "__main__":
    main()
