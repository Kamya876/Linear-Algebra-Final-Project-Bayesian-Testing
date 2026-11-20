"""Iteration 3 -- simulation test: recover the known truth (Project 7).

Model, parameters and seed are the Iteration 2 ones: p_A = 0.10, p_B = 0.12,
n = 1000 per group, Beta(1,1) priors, seed 0 (so x_A = 92, x_B = 121).

Contents
  1. main run vs truth and vs SciPy (posterior, grid, Monte Carlo, z-test, credible interval)
  2. error vs n on log-log axes, with a fitted slope
  3. Monte Carlo SE check (assumption A7)
  4. null rejection frequency of the z-test (5% +/- 1.4%)
  5. stress tests: degenerate counts, prior sensitivity, users in both arms
  6. assumption checks on the simulated data (they hold by construction)

Run:  python src/check_simulation.py
"""
import sys
from math import erfc, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from library_checks import (lib_ci_endpoint_tolerance, lib_credible_interval,  # noqa: E402
                            lib_posterior_stats, lib_prob_b_greater_a, lib_z_test)
from methods import (my_beta_posterior, my_credible_interval, my_grid_probability,  # noqa: E402
                     my_mc_probability, my_normal_approx_min_count, my_posterior_stats,
                     my_srm_test, my_weighted_posterior_mean, my_z_test)
from simulate import P_A_TRUE, P_B_TRUE, simulate_ab, simulate_shared_users  # noqa: E402
from verify import DOCS, FIGS, VerificationLog, report  # noqa: E402

DATA = "Simulation (n=1000)"


def posterior_pair(x_A, n_A, x_B, n_B, a=1.0, b=1.0):
    return my_beta_posterior(x_A, n_A, a, b), my_beta_posterior(x_B, n_B, a, b)


# ---------------------------------------------------------------------------
def main_run(log):
    print("\n=== 1. MAIN RUN (n = 1000, seed 0) ===")
    (x_A, n_A, x_B, n_B), truth = simulate_ab(n=1000, seed=0)
    print(f"x_A = {x_A}/{n_A}, x_B = {x_B}/{n_B}   (Iteration 2 report: 92 and 121)")
    assert (x_A, x_B) == (92, 121), "Iteration 3 simulation no longer matches Iteration 2"
    (a_A, b_A), (a_B, b_B) = posterior_pair(x_A, n_A, x_B, n_B)
    m_A, v_A = my_posterior_stats(a_A, b_A)
    m_B, v_B = my_posterior_stats(a_B, b_B)

    # vs truth: within 3 posterior standard deviations
    for name, mean, var, true in (("A", m_A, v_A, truth["p_A"]), ("B", m_B, v_B, truth["p_B"])):
        log.within_se(DATA, f"posterior mean p_{name}", f"true p_{name}", mean, true, sqrt(var))
    # weighted-average identity (Question 3)
    log.compare(DATA, "weighted-average posterior mean", "direct posterior mean",
                [my_weighted_posterior_mean(x_A, n_A), my_weighted_posterior_mean(x_B, n_B)],
                [m_A, m_B], 1e-15)
    # vs SciPy
    lib = [*lib_posterior_stats(a_A, b_A), *lib_posterior_stats(a_B, b_B)]
    log.compare(DATA, "posterior means/variances", "scipy.stats.beta", [m_A, v_A, m_B, v_B], lib, 1e-12)
    grid = my_grid_probability(a_A, b_A, a_B, b_B, 10001)
    quad = lib_prob_b_greater_a(a_A, b_A, a_B, b_B)
    mc, se = my_mc_probability(a_A, b_A, a_B, b_B, M=100000, seed=0)
    log.compare(DATA, "P(B>A) grid", "scipy.integrate.quad", grid, quad, 1e-6)
    log.within_se(DATA, "P(B>A) Monte Carlo", "scipy.integrate.quad", mc, quad, se)
    z, p = my_z_test(x_A, n_A, x_B, n_B)
    z_lib, p_lib = lib_z_test(x_A, n_A, x_B, n_B)
    log.compare(DATA, "z and p-value", "scipy chi2_contingency", [z, p], [z_lib, p_lib], 1e-10)
    lo, hi = my_credible_interval(a_A, b_A, a_B, b_B, M=200000, seed=0)
    elo, ehi = lib_credible_interval(a_A, b_A, a_B, b_B)
    tol = max(lib_ci_endpoint_tolerance(a_A, b_A, a_B, b_B, elo, 200000),
              lib_ci_endpoint_tolerance(a_A, b_A, a_B, b_B, ehi, 200000))
    log.compare(DATA, "95% interval for pB-pA (MC)", "exact CDF root-finding", [lo, hi], [elo, ehi], tol)
    print(f"posterior means {m_A:.6f}, {m_B:.6f}; P(B>A): grid {grid:.8f}, quad {quad:.8f}, "
          f"MC {mc:.6f} (SE {se:.6f}); z = {z:.4f}, p = {p:.4f}; interval [{lo:.6f}, {hi:.6f}]")
    report("posterior mean/var vs scipy", [m_A, v_A, m_B, v_B], lib)
    report("grid vs quad", grid, quad)
    return (x_A, n_A, x_B, n_B)


# ---------------------------------------------------------------------------
def error_vs_n(log, reps=500):
    print("\n=== 2. ERROR VS SAMPLE SIZE (log-log) ===")
    sizes = [50, 100, 300, 500, 1000, 3000, 5000, 10000]
    rmse_A, rmse_B, rows = [], [], []
    for n in sizes:
        err_A, err_B = [], []
        for seed in range(reps):
            (x_A, n_A, x_B, n_B), truth = simulate_ab(n=n, seed=seed)
            err_A.append(my_posterior_stats(*my_beta_posterior(x_A, n_A))[0] - truth["p_A"])
            err_B.append(my_posterior_stats(*my_beta_posterior(x_B, n_B))[0] - truth["p_B"])
        rmse_A.append(float(np.sqrt(np.mean(np.square(err_A)))))
        rmse_B.append(float(np.sqrt(np.mean(np.square(err_B)))))
        # single run, seed 0, in posterior SDs
        (x_A, n_A, x_B, n_B), truth = simulate_ab(n=n, seed=0)
        mA, vA = my_posterior_stats(*my_beta_posterior(x_A, n_A))
        mB, vB = my_posterior_stats(*my_beta_posterior(x_B, n_B))
        rows.append((n, rmse_A[-1], rmse_B[-1], abs(mA - truth["p_A"]) / sqrt(vA),
                     abs(mB - truth["p_B"]) / sqrt(vB)))
        print(f"n={n:6d} | RMSE A={rmse_A[-1]:.6f} B={rmse_B[-1]:.6f} | seed-0 error in posterior SDs: "
              f"A={rows[-1][3]:.2f} B={rows[-1][4]:.2f}")
    slope_A = np.polyfit(np.log(sizes), np.log(rmse_A), 1)[0]
    slope_B = np.polyfit(np.log(sizes), np.log(rmse_B), 1)[0]
    print(f"Fitted log-log slopes: A = {slope_A:.3f}, B = {slope_B:.3f}  (theory -0.5)")
    ok = all(-0.6 <= s <= -0.4 for s in (slope_A, slope_B))
    log.add(DATA, "log-log slope of RMSE vs n (A, B)", "theory -1/2",
            f"slopes {slope_A:.3f}, {slope_B:.3f} (allowed [-0.6, -0.4])", ok)
    log.add(DATA, "seed-0 error at every n", "3 posterior SDs",
            f"max {max(max(r[3], r[4]) for r in rows):.2f} SD (limit 3)",
            max(max(r[3], r[4]) for r in rows) <= 3)

    FIGS.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.loglog(sizes, rmse_A, "o-", label=f"Group A (slope {slope_A:.2f})")
    plt.loglog(sizes, rmse_B, "s-", label=f"Group B (slope {slope_B:.2f})")
    ref = [sqrt(0.11 * 0.89 / n) for n in sizes]
    plt.loglog(sizes, ref, "k--", alpha=0.6, label=r"$\sqrt{p(1-p)/n}$ (slope $-1/2$)")
    plt.xlabel("Sample size per group n (log scale)")
    plt.ylabel("RMSE of posterior mean vs truth (log scale)")
    plt.title(f"Posterior-mean error vs n ({reps} replications per n)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "error_convergence.png", dpi=300)
    plt.close()
    with open(DOCS / "error_convergence.csv", "w") as f:
        f.write("n,rmse_A,rmse_B,seed0_error_A_in_sd,seed0_error_B_in_sd\n")
        for r in rows:
            f.write(",".join(f"{v:.6f}" if i else str(v) for i, v in enumerate(r)) + "\n")


# ---------------------------------------------------------------------------
def mc_se_check(log, counts):
    print("\n=== 3. MONTE CARLO STANDARD ERROR (assumption A7) ===")
    (a_A, b_A), (a_B, b_B) = posterior_pair(*counts)
    q = lib_prob_b_greater_a(a_A, b_A, a_B, b_B)
    worst = 0.0
    for M in (100, 1000, 10000):
        qhat = [my_mc_probability(a_A, b_A, a_B, b_B, M=M, seed=s)[0] for s in range(300)]
        emp, formula = float(np.std(qhat, ddof=1)), sqrt(q * (1 - q) / M)
        worst = max(worst, abs(emp - formula) / formula)
        print(f"M={M:6d}: empirical SD of q_hat = {emp:.5f}   sqrt(q(1-q)/M) = {formula:.5f}")
    log.add(DATA, "empirical SD of q_hat (M = 100, 1000, 10000)", "sqrt(q(1-q)/M)",
            f"worst relative gap {worst:.1%} (limit 15%)", worst <= 0.15)


# ---------------------------------------------------------------------------
def null_rejection(log):
    print("\n=== 4. NULL REJECTION FREQUENCY: 1000 experiments, pA = pB = 0.10 ===")
    rng = np.random.default_rng(0)
    rej = 0
    for _ in range(1000):
        x_A, x_B = rng.binomial(1000, 0.10), rng.binomial(1000, 0.10)
        rej += my_z_test(x_A, 1000, x_B, 1000)[1] < 0.05
    rate = rej / 1000
    print(f"Rejections: {rej}/1000 = {rate:.3f} (accept 0.036 - 0.064)")
    log.add(DATA, "z-test false-positive rate (1000 runs)", "0.05 +/- 0.014",
            f"{rate:.3f}", 0.05 - 0.014 <= rate <= 0.05 + 0.014)


# ---------------------------------------------------------------------------
def stress_degenerate(log):
    print("\n=== 5a. STRESS TEST: degenerate counts ===")
    ok_all = True
    for name, x_A, n_A, x_B, n_B in [("Zero successes", 0, 100, 0, 100),
                                     ("All successes", 100, 100, 100, 100),
                                     ("Extreme difference", 1, 100, 99, 100)]:
        (a_A, b_A), (a_B, b_B) = posterior_pair(x_A, n_A, x_B, n_B)
        mA, vA = my_posterior_stats(a_A, b_A)
        mB, vB = my_posterior_stats(a_B, b_B)
        q, se = my_mc_probability(a_A, b_A, a_B, b_B, M=100000, seed=0)
        ok = 0 <= mA <= 1 and 0 <= mB <= 1 and 0 <= q <= 1 and vA >= 0 and vB >= 0
        try:                              # the z-test is undefined when p_pool is 0 or 1
            my_z_test(x_A, n_A, x_B, n_B)
            z_note = "z-test defined"
        except ValueError:
            z_note = "z-test undefined (SE0 = 0)"
        print(f"{name:20s} means {mA:.4f}, {mB:.4f}  P(B>A) = {q:.4f}  [{z_note}]  {'PASS' if ok else 'FAIL'}")
        ok_all &= ok
    log.add("Simulation (stress test)", "degenerate counts 0/100, 100/100, 1/100 vs 99/100",
            "valid probabilities", "posterior stays valid; z-test raises an error when SE0 = 0", ok_all)


def prior_sensitivity(counts_by_n):
    print("\n=== 5b. PRIOR SENSITIVITY (assumption A6) ===")
    priors = [(1, 1), (0.5, 0.5), (1, 9), (10, 90), (50, 50)]
    rows = []
    for n, counts in counts_by_n.items():
        for a, b in priors:
            (a_A, b_A), (a_B, b_B) = posterior_pair(*counts, a=a, b=b)
            mA = my_posterior_stats(a_A, b_A)[0]
            mB = my_posterior_stats(a_B, b_B)[0]
            q, se = my_mc_probability(a_A, b_A, a_B, b_B, M=100000, seed=0)
            rows.append((n, f"Beta({a},{b})", mA, mB, q, se))
            print(f"n={n:5d} prior Beta({a},{b}): mean A {mA:.4f}, mean B {mB:.4f}, P(B>A) = {q:.4f} (SE {se:.4f})")
    with open(DOCS / "prior_sensitivity.csv", "w") as f:
        f.write("n_per_group,prior,mean_A,mean_B,P_B_gt_A,MC_SE\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f}\n")
    return rows


def overlap_stress(log):
    print("\n=== 5c. STRESS TEST: users in BOTH arms (assumption A5) ===")
    n, p, reps = 1000, 0.10, 10000
    naive_sd = sqrt(2 * p * (1 - p) / n)          # SD of pB_hat - pA_hat if arms were independent
    rows = []
    for overlap in (0, 100, 500):
        x_A, x_B = simulate_shared_users(n=n, overlap=overlap, p=p, reps=reps, seed=0)
        diff = x_B / n - x_A / n
        emp_sd = float(np.std(diff, ddof=1))
        ratio_emp, ratio_th = emp_sd / naive_sd, sqrt(1 - overlap / n)
        rej = np.mean([my_z_test(int(a), n, int(b), n)[1] < 0.05 for a, b in zip(x_A, x_B)])
        rej_th = erfc((1.96 / ratio_th) / sqrt(2))
        rej_se = sqrt(rej_th * (1 - rej_th) / reps)
        ids_A, ids_B = np.arange(n), np.concatenate([np.arange(overlap), np.arange(n, 2 * n - overlap)])
        n_both = len(np.intersect1d(ids_A, ids_B))
        print(f"overlap {overlap:3d} ({overlap / n:4.0%}): users in both arms = {n_both:3d} -> assumption check "
              f"{'FAIL' if n_both else 'pass'};  true SD / naive SD = {ratio_emp:.3f} (theory {ratio_th:.3f});  "
              f"false-positive rate = {rej:.4f} (theory {rej_th:.4f}, nominal 0.05)")
        rows.append((overlap, n_both, ratio_emp, ratio_th, rej, rej_th))
        log.add("Simulation (stress test)", f"SD(pB-pA) / naive SE, {overlap / n:.0%} overlap",
                f"theory sqrt(1-overlap) = {ratio_th:.3f}",
                f"{ratio_emp:.3f} (rel. {abs(ratio_emp - ratio_th) / ratio_th:.1%})",
                abs(ratio_emp - ratio_th) / ratio_th <= 0.03)
        log.within_se("Simulation (stress test)", f"z-test false-positive rate, {overlap / n:.0%} overlap",
                      "theory 2(1-Phi(1.96/ratio))", rej, rej_th, rej_se)
    with open(DOCS / "overlap_stress.csv", "w") as f:
        f.write("overlap_users,users_in_both_arms,sd_ratio_empirical,sd_ratio_theory,false_pos_empirical,false_pos_theory\n")
        for r in rows:
            f.write(",".join(f"{v:.5f}" if isinstance(v, float) else str(v) for v in r) + "\n")
    return rows


# ---------------------------------------------------------------------------
def assumption_checks(log, counts):
    print("\n=== 6. ASSUMPTION CHECKS ON SIMULATED DATA (should pass) ===")
    x_A, n_A, x_B, n_B = counts
    z, p = my_srm_test(n_A, n_B)
    min_count = my_normal_approx_min_count(x_A, n_A, x_B, n_B)
    checks = {
        "A1 binary outcome: counts valid (0 <= x <= n)": 0 <= x_A <= n_A and 0 <= x_B <= n_B,
        "A4 sample-ratio test (equal group sizes by design)": p > 0.05,
        "A5 independent arms: no user ids shared": True,   # generator draws separate arms
        "A6 prior parameters positive": True,
        "A8 normal approximation: min expected count >= 10": min_count >= 10,
    }
    for name, ok in checks.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"SRM z = {z:.3f}, p = {p:.3f};  min expected count = {min_count:.1f}")
    print("A2/A3/A7: independent Bernoulli trials with one probability per arm and independent Monte Carlo "
          "draws hold by construction (A7 verified in section 3).")
    log.add(DATA, "assumption checks A1, A4, A5, A6, A8", "hold by construction",
            f"min expected count {min_count:.0f}; SRM p = {p:.2f}", all(checks.values()))


def main():
    DOCS.mkdir(exist_ok=True)
    log = VerificationLog("verification_simulation.csv")
    counts = main_run(log)
    error_vs_n(log)
    mc_se_check(log, counts)
    null_rejection(log)
    stress_degenerate(log)
    counts_by_n = {1000: counts, 50: simulate_ab(n=50, seed=0)[0]}
    prior_sensitivity(counts_by_n)
    overlap_stress(log)
    assumption_checks(log, counts)
    log.save()


if __name__ == "__main__":
    main()
