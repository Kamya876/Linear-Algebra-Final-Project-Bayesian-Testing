"""Prescribed known-truth simulation for DS5020 Project 7."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from .common import (beta_posterior_parameters, beta_posterior_mean,
                     beta_posterior_variance, two_proportion_z_test,
                     weighted_posterior_mean, exact_prob_b_greater_a,
                     theoretical_power)

def run_simulation():
    Path("docs").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    # Instructor's specified known-truth setup
    p_a_true, p_b_true, n = 0.10, 0.12, 1000
    x_a = int(rng.binomial(n, p_a_true))
    x_b = int(rng.binomial(n, p_b_true))

    # Uniform Beta(1,1) priors
    a0, b0 = 1.0, 1.0
    a_a, b_a = beta_posterior_parameters(a0, b0, x_a, n)
    a_b, b_b = beta_posterior_parameters(a0, b0, x_b, n)

    mean_a = beta_posterior_mean(a_a, b_a)
    mean_b = beta_posterior_mean(a_b, b_b)
    var_a = beta_posterior_variance(a_a, b_a)
    var_b = beta_posterior_variance(a_b, b_b)

    # Verify posterior mean as a weighted average
    w = n / (a0 + b0 + n)
    weighted_a = (1-w) * (a0/(a0+b0)) + w * (x_a/n)
    weighted_b = (1-w) * (a0/(a0+b0)) + w * (x_b/n)

    # Monte Carlo estimate P(p_B > p_A | data)
    m = 100_000
    draws_a = rng.beta(a_a, b_a, size=m)
    draws_b = rng.beta(a_b, b_b, size=m)
    indicators = (draws_b > draws_a).astype(float)
    probability_hat = indicators.mean()
    mc_se = np.sqrt(probability_hat * (1-probability_hat) / m)

    # Equal-tail 95% credible intervals from Beta quantiles
    ci_a = stats.beta.ppf([0.025, 0.975], a_a, b_a)
    ci_b = stats.beta.ppf([0.025, 0.975], a_b, b_b)

    # Frequentist comparison
    z_result = two_proportion_z_test(x_a, n, x_b, n)
    scipy_z = stats.norm.isf(z_result["p_value"]/2) * np.sign(z_result["z"])

    print("\n=== Known-truth simulation ===")
    print(f"True p_A={p_a_true:.3f}, True p_B={p_b_true:.3f}; n_A=n_B={n}")
    print(f"Observed successes: x_A={x_a}, x_B={x_b}")
    print(f"Observed rates: A={x_a/n:.5f}, B={x_b/n:.5f}")
    print("\nPosterior results (Beta(1,1) prior)")
    print(f"A posterior: Beta({a_a:.0f}, {b_a:.0f}); mean={mean_a:.6f}; variance={var_a:.8f}")
    print(f"B posterior: Beta({a_b:.0f}, {b_b:.0f}); mean={mean_b:.6f}; variance={var_b:.8f}")
    print(f"Weighted-mean check A: direct={mean_a:.10f}, weighted={weighted_a:.10f}")
    print(f"Weighted-mean check B: direct={mean_b:.10f}, weighted={weighted_b:.10f}")
    print(f"95% credible interval A: [{ci_a[0]:.6f}, {ci_a[1]:.6f}]")
    print(f"95% credible interval B: [{ci_b[0]:.6f}, {ci_b[1]:.6f}]")
    print(f"Monte Carlo P(p_B > p_A | data)={probability_hat:.6f}")
    print(f"Monte Carlo standard error={mc_se:.6f} (M={m})")
    print("\nFrequentist comparison")
    print(f"Pooled proportion={z_result['pooled']:.6f}")
    print(f"Standard error={z_result['se']:.6f}")
    print(f"z={z_result['z']:.6f}; two-sided p-value={z_result['p_value']:.6f}")
    print(f"Normal-library z magnitude check={abs(scipy_z):.6f}")

    # Posterior density plot
    grid = np.linspace(0, 0.25, 600)
    plt.figure(figsize=(8, 5))
    plt.plot(grid, stats.beta.pdf(grid, a_a, b_a), label="Version A")
    plt.plot(grid, stats.beta.pdf(grid, a_b, b_b), label="Version B")
    plt.xlabel("Conversion probability")
    plt.ylabel("Posterior density")
    plt.title("Posterior distributions: known-truth simulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/simulation_posterior.png", dpi=160)
    plt.close()

    # Difference distribution
    diff = draws_b - draws_a
    plt.figure(figsize=(8, 5))
    plt.hist(diff, bins=60, density=True)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Difference p_B - p_A")
    plt.ylabel("Density")
    plt.title(f"Posterior difference; P(p_B > p_A)={probability_hat:.3f}")
    plt.tight_layout()
    plt.savefig("figures/simulation_difference.png", dpi=160)
    plt.close()

    # Monte Carlo convergence diagnostic
    checkpoints = np.array([100, 500, 1000, 5000, 10000, 50000, 100000])
    running = np.array([indicators[:k].mean() for k in checkpoints])
    plt.figure(figsize=(8, 5))
    plt.plot(checkpoints, running, marker="o")
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.xlabel("Number of Monte Carlo draws (log scale)")
    plt.ylabel("Estimated P(p_B > p_A | data)")
    plt.title("Monte Carlo estimate convergence")
    plt.tight_layout()
    plt.savefig("figures/monte_carlo_convergence.png", dpi=160)
    plt.close()

    # Repeated z-test experiment: empirical type-I error and power
    reps = 2000
    alpha = 0.05
    null_rejects = 0
    power_rejects = 0
    for _ in range(reps):
        xa0 = rng.binomial(n, 0.10)
        xb0 = rng.binomial(n, 0.10)
        r0 = two_proportion_z_test(xa0, n, xb0, n)
        null_rejects += int(r0["p_value"] < alpha)

        xa1 = rng.binomial(n, 0.10)
        xb1 = rng.binomial(n, 0.12)
        r1 = two_proportion_z_test(xa1, n, xb1, n)
        power_rejects += int(r1["p_value"] < alpha)

    type1 = null_rejects / reps
    power = power_rejects / reps
    print("\nRepeated frequentist experiment")
    print(f"Repetitions={reps}; significance level={alpha}")
    print(f"Estimated type-I error under p_A=p_B=0.10: {type1:.4f} (target approximately 0.05)")
    print(f"Estimated power under p_A=0.10, p_B=0.12: {power:.4f}")

    # ------------------------------------------------------------------
    # Extra checks (run AFTER the original random stream so earlier
    # numbers are unchanged for seed 0).
    # ------------------------------------------------------------------
    # (a) Posterior variance formula against the Monte Carlo draws
    var_a_mc, var_b_mc = draws_a.var(ddof=1), draws_b.var(ddof=1)

    # (b) Monte Carlo estimate vs deterministic numerical integral
    q_exact = exact_prob_b_greater_a(a_a, b_a, a_b, b_b)

    # (c) Monte Carlo standard error scales like 1/sqrt(M):
    #     repeat the estimator R times for several M, compare the empirical
    #     standard deviation with sqrt(q(1-q)/M).
    R = 300
    scaling_rows = []
    for M_ in [100, 1_000, 10_000]:
        est = np.array([np.mean(rng.beta(a_b, b_b, M_) > rng.beta(a_a, b_a, M_))
                        for _ in range(R)])
        scaling_rows.append([M_, est.mean(), est.std(ddof=1),
                             np.sqrt(q_exact * (1 - q_exact) / M_)])
    scaling = pd.DataFrame(scaling_rows, columns=[
        "M", "mean_of_estimates", "empirical_sd", "theory_sqrt_q1mq_over_M"])
    scaling.to_csv("docs/mc_se_scaling.csv", index=False)

    plt.figure(figsize=(6, 4.5))
    plt.loglog(scaling["M"], scaling["empirical_sd"], "o-", label="empirical SD")
    plt.loglog(scaling["M"], scaling["theory_sqrt_q1mq_over_M"], "s--",
               label=r"theory $\sqrt{q(1-q)/M}$")
    plt.xlabel("Monte Carlo draws M"); plt.ylabel("Std. dev. of $\\hat q$")
    plt.title("Monte Carlo standard error (log-log, slope -1/2)")
    plt.legend(); plt.tight_layout()
    plt.savefig("figures/simulation_mc_se_scaling.png", dpi=160); plt.close()

    # (d) credible intervals cover the truth; theoretical power
    covers_a = bool(ci_a[0] <= p_a_true <= ci_a[1])
    covers_b = bool(ci_b[0] <= p_b_true <= ci_b[1])
    power_theory = theoretical_power(p_a_true, p_b_true, n, n, alpha)
    power_se = np.sqrt(power * (1 - power) / reps)
    type1_se = np.sqrt(0.05 * 0.95 / reps)

    print("\nExtra verification checks")
    print(f"Posterior variance formula vs MC draws: A {var_a:.8f} vs {var_a_mc:.8f}; "
          f"B {var_b:.8f} vs {var_b_mc:.8f}")
    print(f"Numerical-integral q={q_exact:.6f}; MC q={probability_hat:.6f}; "
          f"|diff|/SE={abs(probability_hat-q_exact)/mc_se:.2f}")
    print(scaling.to_string(index=False))
    print(f"95% intervals contain truth: A={covers_a}, B={covers_b}")
    print(f"Theoretical power={power_theory:.4f}; simulated={power:.4f} (MC SE {power_se:.4f})")
    print(f"Type-I: simulated={type1:.4f}; nominal 0.05 (MC SE {type1_se:.4f})")
    assert abs(weighted_a - mean_a) < 1e-12 and abs(weighted_b - mean_b) < 1e-12
    assert abs(probability_hat - q_exact) < 4 * mc_se

    comparison = pd.DataFrame([
        ["p_A (posterior mean)", p_a_true, mean_a, f"[{ci_a[0]:.4f}, {ci_a[1]:.4f}]", covers_a],
        ["p_B (posterior mean)", p_b_true, mean_b, f"[{ci_b[0]:.4f}, {ci_b[1]:.4f}]", covers_b],
        ["P(p_B > p_A | data), Monte Carlo", q_exact, probability_hat, f"SE {mc_se:.6f}", ""],
        ["Posterior variance A (formula vs MC)", var_a, var_a_mc, "", ""],
        ["Posterior variance B (formula vs MC)", var_b, var_b_mc, "", ""],
        ["z-test Type-I error", 0.05, type1, f"MC SE {type1_se:.4f}", ""],
        ["z-test power at (0.10, 0.12)", power_theory, power, f"MC SE {power_se:.4f}", ""],
    ], columns=["quantity", "reference_value", "estimate", "note", "interval_covers_truth"])
    comparison.to_csv("docs/simulation_comparison.csv", index=False)

    # Save compact numerical results for reproducibility
    results = pd.DataFrame([
        ["A", p_a_true, n, x_a, x_a/n, a_a, b_a, mean_a, var_a, ci_a[0], ci_a[1]],
        ["B", p_b_true, n, x_b, x_b/n, a_b, b_b, mean_b, var_b, ci_b[0], ci_b[1]],
    ], columns=["group", "true_rate", "n", "successes", "observed_rate",
                "posterior_alpha", "posterior_beta", "posterior_mean",
                "posterior_variance", "credible_2.5%", "credible_97.5%"])
    results.to_csv("docs/simulation_results.csv", index=False)

    with open("docs/simulation_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"x_A={x_a}, x_B={x_b}, n={n}\n")
        f.write(f"posterior_probability_B_better={probability_hat:.8f}\n")
        f.write(f"monte_carlo_se={mc_se:.8f}\n")
        f.write(f"z={z_result['z']:.8f}, p_value={z_result['p_value']:.8f}\n")
        f.write(f"type1_error_estimate={type1:.8f}, power_estimate={power:.8f}\n")

if __name__ == "__main__":
    run_simulation()
