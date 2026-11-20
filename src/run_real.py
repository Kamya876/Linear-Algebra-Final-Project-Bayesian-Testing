"""Iteration 3 -- real data: Cookie Cats retention, from-scratch methods vs SciPy.

A = gate_30, B = gate_40.  Day-1 retention is the primary outcome, day-7 the secondary one.
Every cleaning step is recorded in docs/real_data_cleaning_log.txt.

Run:  python src/run_real.py
"""
import sys
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from library_checks import (lib_ci_endpoint_tolerance, lib_credible_interval,  # noqa: E402
                            lib_posterior_stats, lib_prob_b_greater_a, lib_z_test)
from methods import (my_beta_posterior, my_credible_interval, my_grid_probability,  # noqa: E402
                     my_mc_probability, my_normal_approx_min_count, my_posterior_stats,
                     my_srm_test, my_z_steps)
from verify import DOCS, ROOT, VerificationLog, report  # noqa: E402

DATA_PATH = ROOT / "data" / "cookie_cats.csv"
REQUIRED = ["userid", "version", "retention_1", "retention_7"]


# ---------------------------------------------------------------------------
def load_and_clean():
    """Load the CSV and record every cleaning step."""
    steps = []
    df = pd.read_csv(DATA_PATH)
    steps.append(f"1. Loaded {DATA_PATH.name}: {len(df)} rows, columns {list(df.columns)}.")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    n_missing = int(df[REQUIRED].isna().sum().sum())
    steps.append(f"2. Missing values in the four analysis columns: {n_missing}; rows removed: {n_missing}.")
    df = df.dropna(subset=REQUIRED)
    dup = int(df["userid"].duplicated().sum())
    steps.append(f"3. Duplicate userids: {dup}; rows removed: {dup}.")
    df = df.drop_duplicates(subset="userid")
    both = int((df.groupby("userid")["version"].nunique() > 1).sum())
    steps.append(f"4. Users appearing in both groups: {both}; rows removed: 0 (reported only).")
    bad = int((~df["version"].isin(["gate_30", "gate_40"])).sum())
    steps.append(f"5. Rows with a version other than gate_30 / gate_40: {bad}; rows removed: {bad}.")
    df = df[df["version"].isin(["gate_30", "gate_40"])].copy()
    df["retention_1"] = df["retention_1"].astype(int)
    df["retention_7"] = df["retention_7"].astype(int)
    steps.append("6. retention_1 and retention_7 converted from True/False to 1/0 (no rows changed).")
    extreme = int((df["sum_gamerounds"] > 10000).sum()) if "sum_gamerounds" in df else 0
    steps.append(f"7. sum_gamerounds is NOT used in the retention analysis; {extreme} player(s) with more than "
                 "10,000 rounds were kept (no outlier removal is needed for a binary outcome).")
    steps.append("8. Train/test split: none. The Bayesian posterior and the z-test use all rows because the "
                 "analysis estimates a parameter of a randomized experiment; it is not a predictive model.")
    steps.append(f"Final analysis set: {len(df)} rows.")
    (DOCS / "real_data_cleaning_log.txt").write_text("\n".join(steps) + "\n")
    print("=== CLEANING LOG ===")
    print("\n".join(steps))
    return df


def group_counts(df, column):
    A, B = df[df["version"] == "gate_30"], df[df["version"] == "gate_40"]
    return int(A[column].sum()), len(A), int(B[column].sum()), len(B)


# ---------------------------------------------------------------------------
def analyze(df, column, log):
    data = f"Real data ({column})"
    x_A, n_A, x_B, n_B = group_counts(df, column)
    print(f"\n=== ANALYSIS: {column} ===")
    print(f"Group A: {x_A}/{n_A} = {x_A / n_A:.6f}    Group B: {x_B}/{n_B} = {x_B / n_B:.6f}")
    a_A, b_A = my_beta_posterior(x_A, n_A)
    a_B, b_B = my_beta_posterior(x_B, n_B)
    m_A, v_A = my_posterior_stats(a_A, b_A)
    m_B, v_B = my_posterior_stats(a_B, b_B)
    print(f"Posterior mean A {m_A:.6f} (var {v_A:.8f});  B {m_B:.6f} (var {v_B:.8f})")

    mc, se = my_mc_probability(a_A, b_A, a_B, b_B, M=100000, seed=0)
    lo, hi = my_credible_interval(a_A, b_A, a_B, b_B, M=200000, seed=0)
    st = my_z_steps(x_A, n_A, x_B, n_B)
    grid = my_grid_probability(a_A, b_A, a_B, b_B, grid_size=10001, auto_window=True)
    print(f"P(B>A): MC {mc:.6f} (SE {se:.6f}), grid {grid:.8f}")
    print(f"95% credible interval for B - A: [{lo:.6f}, {hi:.6f}]")
    print(f"z = {st['z']:.6f}, p = {st['p_value']:.8f}  (p_pool {st['p_pool']:.6f}, SE0 {st['se0']:.6f})")

    # ---- library comparisons (check only) ----
    lib = [*lib_posterior_stats(a_A, b_A), *lib_posterior_stats(a_B, b_B)]
    log.compare(data, "posterior means/variances", "scipy.stats.beta", [m_A, v_A, m_B, v_B], lib, 1e-12)
    quad = lib_prob_b_greater_a(a_A, b_A, a_B, b_B)
    log.compare(data, "P(B>A) grid", "scipy.integrate.quad", grid, quad, 1e-6)
    log.within_se(data, "P(B>A) Monte Carlo", "scipy.integrate.quad", mc, quad, se)
    z_lib, p_lib = lib_z_test(x_A, n_A, x_B, n_B)
    log.compare(data, "z and p-value", "scipy chi2_contingency", [st["z"], st["p_value"]], [z_lib, p_lib], 1e-10)
    elo, ehi = lib_credible_interval(a_A, b_A, a_B, b_B)
    tol = max(lib_ci_endpoint_tolerance(a_A, b_A, a_B, b_B, elo, 200000),
              lib_ci_endpoint_tolerance(a_A, b_A, a_B, b_B, ehi, 200000))
    log.compare(data, "95% interval for pB-pA (MC)", "exact CDF root-finding", [lo, hi], [elo, ehi], tol)
    report(f"{column}: posterior vs scipy", [m_A, v_A, m_B, v_B], lib)
    report(f"{column}: grid vs quad", grid, quad)
    report(f"{column}: z,p vs scipy", [st["z"], st["p_value"]], [z_lib, p_lib])

    # ---- prior sensitivity (assumption A6) ----
    print("Prior sensitivity (P(B>A) by Monte Carlo, M = 100,000):")
    out = []
    for a, b in [(1, 1), (0.5, 0.5), (10, 90), (200, 200)]:
        pa, pb = my_beta_posterior(x_A, n_A, a, b), my_beta_posterior(x_B, n_B, a, b)
        q, qse = my_mc_probability(*pa, *pb, M=100000, seed=0)
        out.append((a, b, q, qse))
        print(f"   Beta({a},{b}): P(B>A) = {q:.6f} (SE {qse:.6f})")
    spread = max(o[2] for o in out) - min(o[2] for o in out)
    log.add(data, "P(B>A) under 4 priors, Beta(1,1) ... Beta(200,200)", "each other",
            f"range {spread:.4f} (prior has little influence if small)", spread < 0.01)
    return {"x_A": x_A, "n_A": n_A, "x_B": x_B, "n_B": n_B, "mc": mc, "se": se, "lo": lo, "hi": hi,
            "z": st["z"], "p": st["p_value"], "prior": out}


# ---------------------------------------------------------------------------
def check_real_assumptions(df, results, log):
    print("\n=== REAL-DATA ASSUMPTION CHECKS ===")
    checks = {
        "A1 binary outcomes (retention_1, retention_7 in {0,1})":
            df["retention_1"].isin([0, 1]).all() and df["retention_7"].isin([0, 1]).all(),
        "No missing required values": not df[REQUIRED].isna().any().any(),
        "A5 independent arms: unique userids": df["userid"].is_unique,
        "A5 independent arms: no user in both groups": (df.groupby("userid")["version"].nunique() <= 1).all(),
        "Both groups have observations": df["version"].nunique() == 2,
    }
    for name, ok in checks.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    log.add("Real data", "data-quality checks (binary, no missing, unique ids, no overlap)", "required",
            "all pass" if all(checks.values()) else "see output", all(checks.values()))

    counts = df["version"].value_counts()
    n_A, n_B = int(counts["gate_30"]), int(counts["gate_40"])
    z, p = my_srm_test(n_A, n_B)
    chi = stats.chisquare([n_A, n_B])
    print(f"\nA4 sample-ratio test for a 50/50 split: nA = {n_A}, nB = {n_B}, ratio {n_A / n_B:.4f}, "
          f"z = {z:.3f}, p = {p:.4f}")
    print(f"   scipy chisquare check: chi2 = {chi.statistic:.4f} (= z^2 = {z * z:.4f}), p = {chi.pvalue:.4f}")
    log.compare("Real data", "sample-ratio z^2 and p-value", "scipy.stats.chisquare",
                [z * z, p], [chi.statistic, chi.pvalue], 1e-9)
    print("   => the 50/50 split is rejected at 5%. This is a FLAG about assignment/logging, not proof of bias; "
          "see the report discussion. It does not change the arithmetic of the two methods.")

    for key, col in (("d1", "retention_1"), ("d7", "retention_7")):
        r = results[col]
        mc = my_normal_approx_min_count(r["x_A"], r["n_A"], r["x_B"], r["n_B"])
        print(f"A8 normal approximation ({col}): min expected count = {mc:.0f} (need >= 10): "
              f"{'PASS' if mc >= 10 else 'FAIL'}")
        log.add(f"Real data ({col})", "A8 min expected count n_g*p_pool", ">= 10",
                f"{mc:.0f}", mc >= 10)

    print("\nNot testable from the CSV: independence between distinct users, the real randomization "
          "mechanism, a stable retention probability within an arm, repeated exposure. sum_gamerounds is "
          "measured AFTER assignment, so it cannot be used as a baseline-balance check.")
    # Day-1 and day-7 outcomes come from the same users:
    both = pd.crosstab(df["retention_1"], df["retention_7"])
    print("\nDay-1 x day-7 retention counts (same users; the two analyses are not independent):")
    print(both.to_string())
    print(f"Players retained on day 7 but not on day 1: {int(both.loc[0, 1])}")


def main():
    DOCS.mkdir(exist_ok=True)
    log = VerificationLog("verification_real.csv")
    df = load_and_clean()
    print("\n=== GROUP SIZES ===")
    print(df["version"].value_counts().to_string())
    results = {c: analyze(df, c, log) for c in ("retention_1", "retention_7")}
    check_real_assumptions(df, results, log)
    log.save()


if __name__ == "__main__":
    main()
