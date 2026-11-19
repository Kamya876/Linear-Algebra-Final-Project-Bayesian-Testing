
import pandas as pd
from pathlib import Path
from methods import (
    my_beta_posterior,
    my_posterior_stats,
    my_mc_probability,
    my_z_test,
    my_grid_probability,
    my_credible_interval
)
from scipy.stats import beta
from scipy.integrate import quad


def load_and_check_data():
    # Locate the dataset relative to this script
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "cookie_cats.csv"

    # Load dataset
    df = pd.read_csv(data_path)

    print("=== COOKIE CATS DATASET ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Check required columns
    required = ["userid", "version", "retention_1", "retention_7"]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check missing values
    print("\n=== MISSING VALUES ===")
    print(df[required].isna().sum())

    # Check duplicate users
    print("\n=== DUPLICATE USERS ===")
    print(f"Duplicate user IDs: {df['userid'].duplicated().sum()}")

    # Check users appearing in both groups
    group_counts = df.groupby("userid")["version"].nunique()
    both_groups = (group_counts > 1).sum()

    print(f"Users in both groups: {both_groups}")

    # Check group sizes
    print("\n=== GROUP SIZES ===")
    print(df["version"].value_counts())

    # Check retention values
    print("\n=== RETENTION VALUES ===")
    print("Day 1:", df["retention_1"].unique())
    print("Day 7:", df["retention_7"].unique())

    return df


def analyze_retention(df, column):
    print(f"\n=== ANALYSIS: {column} ===")

    # A = gate_30, B = gate_40
    group_A = df[df["version"] == "gate_30"]
    group_B = df[df["version"] == "gate_40"]

    n_A = len(group_A)
    n_B = len(group_B)

    x_A = int(group_A[column].sum())
    x_B = int(group_B[column].sum())

    print(f"Group A: {x_A}/{n_A} retained")
    print(f"Group B: {x_B}/{n_B} retained")

    print(f"Observed retention A: {x_A/n_A:.6f}")
    print(f"Observed retention B: {x_B/n_B:.6f}")

    # Bayesian posterior
    a_A, b_A = my_beta_posterior(x_A, n_A)
    a_B, b_B = my_beta_posterior(x_B, n_B)

    mean_A, var_A = my_posterior_stats(a_A, b_A)
    mean_B, var_B = my_posterior_stats(a_B, b_B)

    print("\nPosterior results:")
    print(f"Posterior mean A: {mean_A:.6f}")
    print(f"Posterior mean B: {mean_B:.6f}")
    print(f"Posterior variance A: {var_A:.8f}")
    print(f"Posterior variance B: {var_B:.8f}")

    # Monte Carlo probability
    mc_prob, mc_se = my_mc_probability(
        a_A, b_A, a_B, b_B,
        M=100000, seed=0
    )

    print("\nBayesian comparison:")
    print(f"P(B > A): {mc_prob:.6f}")
    print(f"Monte Carlo SE: {mc_se:.6f}")

    ci_low, ci_high = my_credible_interval(
        a_A, b_A, a_B, b_B,
        M=200000, seed=0
    )

    print("\n95% credible interval for B - A:")
    print(f"[{ci_low:.6f}, {ci_high:.6f}]")



    # Frequentist comparison
    z, p_value = my_z_test(x_A, n_A, x_B, n_B)

    print("\nFrequentist comparison:")
    print(f"Z-statistic: {z:.6f}")
    print(f"Two-sided p-value: {p_value:.6f}")


def verify_real_data(df, column):
    print(f"\n=== SCIPY VERIFICATION: {column} ===")

    group_A = df[df["version"] == "gate_30"]
    group_B = df[df["version"] == "gate_40"]

    x_A = int(group_A[column].sum())
    x_B = int(group_B[column].sum())

    n_A = len(group_A)
    n_B = len(group_B)

    a_A, b_A = my_beta_posterior(x_A, n_A)
    a_B, b_B = my_beta_posterior(x_B, n_B)

    # Our implementation
    mean_A, var_A = my_posterior_stats(a_A, b_A)
    mean_B, var_B = my_posterior_stats(a_B, b_B)

    # SciPy reference
    scipy_mean_A = beta.mean(a_A, b_A)
    scipy_var_A = beta.var(a_A, b_A)
    scipy_mean_B = beta.mean(a_B, b_B)
    scipy_var_B = beta.var(a_B, b_B)

    print(f"{'Metric':<15} {'Ours':<15} {'SciPy':<15} {'Error':<12}")

    results = [
        ("Mean A", mean_A, scipy_mean_A),
        ("Variance A", var_A, scipy_var_A),
        ("Mean B", mean_B, scipy_mean_B),
        ("Variance B", var_B, scipy_var_B)
    ]

    for name, ours, reference in results:
        print(
            f"{name:<15} "
            f"{ours:<15.8f} "
            f"{reference:<15.8f} "
            f"{abs(ours-reference):<12.2e}"
        )

def verify_probability(df, column):
    print(f"\n=== PROBABILITY VERIFICATION: {column} ===")

    group_A = df[df["version"] == "gate_30"]
    group_B = df[df["version"] == "gate_40"]

    x_A = int(group_A[column].sum())
    x_B = int(group_B[column].sum())

    n_A = len(group_A)
    n_B = len(group_B)

    a_A, b_A = my_beta_posterior(x_A, n_A)
    a_B, b_B = my_beta_posterior(x_B, n_B)

    # Our Monte Carlo implementation
    mc_prob, mc_se = my_mc_probability(
        a_A, b_A, a_B, b_B,
        M=100000, seed=0
    )

    # Independent SciPy verification:
    # Integral of f_B(y) * F_A(y)
    def integrand(y):
        return (
            beta.pdf(y, a_B, b_B)
            * beta.cdf(y, a_A, b_A)
        )

    reference, integration_error = quad(
        integrand,
        0.0,
        1.0,
        epsabs=1e-10,
        limit=300,
        points=[
            float(a_A / (a_A + b_A)),
            float(a_B / (a_B + b_B))
        ]
    )

    difference = abs(mc_prob - reference)

    print(f"Our Monte Carlo: {mc_prob:.6f}")
    print(f"SciPy integration: {reference:.6f}")
    print(f"Absolute difference: {difference:.6f}")
    print(f"Monte Carlo SE: {mc_se:.6f}")
    print(f"Difference / SE: {difference / mc_se:.3f}")

    # Allow for ordinary Monte Carlo variation
    passed = difference <= 3 * mc_se

    print(f"Verification: {'PASS' if passed else 'CHECK'}")


def check_real_assumptions(df):
    print("\n=== REAL DATA ASSUMPTION CHECKS ===")

    required = ["userid", "version", "retention_1", "retention_7"]

    checks = {
        "No missing required values":
            not df[required].isna().any().any(),

        "Unique user IDs":
            df["userid"].is_unique,

        "Valid experiment groups":
            df["version"].isin(["gate_30", "gate_40"]).all(),

        "Binary day-1 retention":
            df["retention_1"].isin([0, 1, False, True]).all(),

        "Binary day-7 retention":
            df["retention_7"].isin([0, 1, False, True]).all(),

        "Both groups have observations":
            (df["version"].value_counts()
             .reindex(["gate_30", "gate_40"], fill_value=0) > 0).all()
    }

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print("\n=== GROUP BALANCE ===")

    counts = df["version"].value_counts()
    n_A = counts["gate_30"]
    n_B = counts["gate_40"]

    print(f"Gate 30: {n_A} users")
    print(f"Gate 40: {n_B} users")
    print(f"Group size ratio: {n_A / n_B:.4f}")

    print("\n=== MODEL ASSUMPTIONS ===")
    print("Independent users: not directly testable")
    print("Random assignment: not directly verified")
    print("Fixed retention probability per group: model assumption")
    print("Beta(1,1) prior: specified in methods.py")

    assert all(checks.values())

    print("\nAll numerical data-quality checks passed.")



if __name__ == "__main__":
    df = load_and_check_data()

    analyze_retention(df, "retention_1")
    analyze_retention(df, "retention_7")

    verify_real_data(df, "retention_1")
    verify_real_data(df, "retention_7")

    verify_probability(df, "retention_1")
    verify_probability(df, "retention_7")

    check_real_assumptions(df)
    
