"""Bayesian and frequentist analysis of the Cookie Cats A/B test."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from .common import beta_posterior_parameters, two_proportion_z_test

GROUP_COL = "version"
OUTCOMES = {
    "retention_1": "Day-1 retention",
    "retention_7": "Day-7 retention",
}
PRIOR_A = 1.0
PRIOR_B = 1.0
N_DRAWS = 200_000
SEED = 0  # handout: np.random.default_rng(0) so every member gets the same numbers


def analyze_outcome(df, outcome_col, label, rng):
    """Analyze one binary retention outcome for gate_30 (A) vs gate_40 (B)."""
    summaries = {}
    for group in ("gate_30", "gate_40"):
        values = df.loc[df[GROUP_COL] == group, outcome_col].dropna().astype(int)
        n = int(values.size)
        x = int(values.sum())
        if n == 0:
            raise ValueError(f"No valid observations for {group} in {outcome_col}.")
        a_post, b_post = beta_posterior_parameters(PRIOR_A, PRIOR_B, x, n)
        summaries[group] = {
            "n": n,
            "x": x,
            "rate": x / n,
            "a_post": a_post,
            "b_post": b_post,
        }

    A = summaries["gate_30"]
    B = summaries["gate_40"]
    draws_a = rng.beta(A["a_post"], A["b_post"], N_DRAWS)
    draws_b = rng.beta(B["a_post"], B["b_post"], N_DRAWS)
    probability_b_better = float(np.mean(draws_b > draws_a))
    mc_se = float(np.sqrt(probability_b_better * (1 - probability_b_better) / N_DRAWS))
    # exact equal-tail intervals from the Beta quantile function
    ci_a = stats.beta.ppf([0.025, 0.975], A["a_post"], A["b_post"])
    ci_b = stats.beta.ppf([0.025, 0.975], B["a_post"], B["b_post"])
    diff_ci = np.quantile(draws_b - draws_a, [0.025, 0.975])

    z = two_proportion_z_test(A["x"], A["n"], B["x"], B["n"])
    result = {
        "outcome": outcome_col,
        "label": label,
        "group_A": "gate_30",
        "n_A": A["n"],
        "successes_A": A["x"],
        "rate_A": A["rate"],
        "posterior_A": f"Beta({A['a_post']:.0f}, {A['b_post']:.0f})",
        "ci_A_low": float(ci_a[0]),
        "ci_A_high": float(ci_a[1]),
        "group_B": "gate_40",
        "n_B": B["n"],
        "successes_B": B["x"],
        "rate_B": B["rate"],
        "posterior_B": f"Beta({B['a_post']:.0f}, {B['b_post']:.0f})",
        "ci_B_low": float(ci_b[0]),
        "ci_B_high": float(ci_b[1]),
        "probability_B_better": probability_b_better,
        "monte_carlo_se": mc_se,
        "difference_ci_low": float(diff_ci[0]),
        "difference_ci_high": float(diff_ci[1]),
        "z_statistic": float(z["z"]),
        "frequentist_p_value": float(z["p_value"]),
    }

    print(f"\n=== {label}: {outcome_col} ===")
    print(f"A = gate_30: {A['x']:,}/{A['n']:,} = {A['rate']:.6%}; posterior {result['posterior_A']}")
    print(f"B = gate_40: {B['x']:,}/{B['n']:,} = {B['rate']:.6%}; posterior {result['posterior_B']}")
    print(f"95% credible interval A: [{ci_a[0]:.6f}, {ci_a[1]:.6f}]")
    print(f"95% credible interval B: [{ci_b[0]:.6f}, {ci_b[1]:.6f}]")
    print(f"P(gate_40 > gate_30 | data) = {probability_b_better:.6%}")
    print(f"Monte Carlo standard error = {mc_se:.6f} (draws={N_DRAWS:,})")
    print(f"95% credible interval for (B - A): [{diff_ci[0]:.6f}, {diff_ci[1]:.6f}]")
    print(f"Two-proportion z = {z['z']:.6f}; two-sided p-value = {z['p_value']:.8f}")

    # Posterior density plot
    plt.figure(figsize=(8, 5))
    plt.hist(draws_a, bins=100, density=True, alpha=0.5, label="gate_30 (A)")
    plt.hist(draws_b, bins=100, density=True, alpha=0.5, label="gate_40 (B)")
    plt.xlabel("Retention probability")
    plt.ylabel("Posterior density (histogram approximation)")
    plt.title(f"Bayesian posterior distributions: {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/real_data_{outcome_col}_posterior.png", dpi=160)
    plt.close()

    # Posterior distribution of the difference
    plt.figure(figsize=(8, 5))
    plt.hist(draws_b - draws_a, bins=120, density=True, alpha=0.75)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Difference in retention probability (gate_40 - gate_30)")
    plt.ylabel("Posterior density (histogram approximation)")
    plt.title(f"Posterior difference: {label}")
    plt.tight_layout()
    plt.savefig(f"figures/real_data_{outcome_col}_difference.png", dpi=160)
    plt.close()
    return result


def describe_data(df):
    """Dataset description required by the handout: size, types, missingness,
    group sizes, duplicates (assumption A5), game rounds, sample-ratio check
    (assumption A4) and exploratory plots."""
    print("\n=== Dataset overview ===")
    print(f"Rows: {len(df):,}; columns: {len(df.columns)}")
    print("\nColumn types:\n" + df.dtypes.to_string())
    print("\nMissing values (all columns):\n" + df.isna().sum().to_string())
    n_dup = int(df["userid"].duplicated().sum()) if "userid" in df.columns else None
    print(f"\nDuplicate userid values: {n_dup}")
    print("\nGroup sizes:\n" + df[GROUP_COL].value_counts().to_string())
    if "sum_gamerounds" in df.columns:
        print("\nsum_gamerounds summary:\n" + df["sum_gamerounds"].describe().to_string())

    # Sample-ratio check: under a fair 50/50 randomization, n_A ~ Bin(N, 1/2)
    n_a = int((df[GROUP_COL] == "gate_30").sum())
    n_b = int((df[GROUP_COL] == "gate_40").sum())
    z_srm = (n_b - n_a) / np.sqrt(n_a + n_b)
    p_srm = 2 * stats.norm.sf(abs(z_srm))
    print(f"\nSample-ratio check (H0: 50/50 split): z={z_srm:.3f}, two-sided p={p_srm:.4f}")

    # Plot 1: group sizes and retention rates
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(["gate_30 (A)", "gate_40 (B)"], [n_a, n_b], color=["tab:blue", "tab:orange"])
    ax[0].set_ylabel("Players"); ax[0].set_title("Group sizes")
    rates = df.groupby(GROUP_COL)[list(OUTCOMES)].mean()
    width = 0.35
    xs = np.arange(len(OUTCOMES))
    ax[1].bar(xs - width / 2, [rates.loc["gate_30", c] for c in OUTCOMES], width, label="gate_30 (A)")
    ax[1].bar(xs + width / 2, [rates.loc["gate_40", c] for c in OUTCOMES], width, label="gate_40 (B)")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(["Day 1", "Day 7"])
    ax[1].set_ylabel("Observed retention rate"); ax[1].set_title("Retention by version")
    ax[1].legend(); plt.tight_layout()
    plt.savefig("figures/real_data_group_sizes.png", dpi=160); plt.close()

    # Plot 2: game rounds (heavy right tail -> log scale)
    if "sum_gamerounds" in df.columns:
        plt.figure(figsize=(7, 4.5))
        plt.hist(np.log10(df["sum_gamerounds"] + 1), bins=60)
        plt.xlabel("log10(sum_gamerounds + 1)"); plt.ylabel("Players")
        plt.title("Game rounds are highly right-skewed")
        plt.tight_layout(); plt.savefig("figures/real_data_gamerounds.png", dpi=160); plt.close()

    # Compact summaries for the report
    pd.DataFrame({
        "metric": ["rows", "columns", "missing_total", "duplicate_userid",
                   "n_gate_30", "n_gate_40", "srm_z", "srm_p"],
        "value": [len(df), len(df.columns), int(df.isna().sum().sum()), n_dup,
                  n_a, n_b, z_srm, p_srm]}).to_csv("docs/real_data_overview.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Bayesian and frequentist Cookie Cats analysis")
    parser.add_argument("--csv", default="data/cookie_cats.csv", help="Path to Cookie Cats CSV")
    args = parser.parse_args()
    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Put cookie_cats.csv in the project's data folder.")

    Path("docs").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    df = pd.read_csv(path)
    required = [GROUP_COL, *OUTCOMES.keys()]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    print(f"File: {path}")
    describe_data(df)

    groups = set(df[GROUP_COL].dropna().unique())
    if not {"gate_30", "gate_40"}.issubset(groups):
        raise ValueError(f"Expected gate_30 and gate_40 groups; found {sorted(groups)}")

    rng = np.random.default_rng(SEED)
    results = []
    for outcome_col, label in OUTCOMES.items():
        results.append(analyze_outcome(df, outcome_col, label, rng))

    results_df = pd.DataFrame(results)
    results_df.to_csv("docs/real_data_bayesian_frequentist_results.csv", index=False)
    with open("docs/real_data_summary_for_report.txt", "w", encoding="utf-8") as f:
        f.write("Cookie Cats real-data analysis\n")
        f.write(f"Dataset: {path}; rows={len(df)}; columns={len(df.columns)}\n")
        f.write("Group A = gate_30; Group B = gate_40\n")
        f.write(f"Prior: Beta({PRIOR_A:g}, {PRIOR_B:g}); Monte Carlo draws={N_DRAWS:,}; seed={SEED}\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nInterpretation note: P(B>A|data) is Bayesian posterior evidence; the z-test p-value is a frequentist tail probability. They are different quantities.\n")
    print("\nSaved results: docs/real_data_bayesian_frequentist_results.csv")
    print("Saved report summary: docs/real_data_summary_for_report.txt")
    print("Saved plots in figures/ (posteriors, differences, group sizes, game rounds).")


if __name__ == "__main__":
    main()
