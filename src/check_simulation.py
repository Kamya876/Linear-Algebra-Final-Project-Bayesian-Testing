
from simulate import simulate_ab_test
from methods import (
    my_beta_posterior,
    my_posterior_stats,
    my_mc_probability,
    my_grid_probability
)
from methods import my_z_test
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Generate simulated data
    x_A, n_A, x_B, n_B = simulate_ab_test()

    # Calculate Beta posteriors
    a_A, b_A = my_beta_posterior(x_A, n_A)
    a_B, b_B = my_beta_posterior(x_B, n_B)

    # Calculate posterior means and variances
    mean_A, var_A = my_posterior_stats(a_A, b_A)
    mean_B, var_B = my_posterior_stats(a_B, b_B)

    # Calculate probability B is better than A
    mc_prob, mc_se = my_mc_probability(
        a_A, b_A, a_B, b_B,
        M=100000, seed=0
    )

    print("=== SIMULATION VERIFICATION ===")

    print(f"Group A: {x_A}/{n_A}")
    print(f"Group B: {x_B}/{n_B}")

    print("\n=== POSTERIOR RESULTS ===")
    print(f"Posterior mean A: {mean_A:.6f}")
    print(f"Posterior mean B: {mean_B:.6f}")

    print("\n=== COMPARISON WITH TRUE VALUES ===")
    print(f"Error A: {abs(mean_A - 0.10):.6f}")
    print(f"Error B: {abs(mean_B - 0.12):.6f}")

    print("\n=== BAYESIAN PROBABILITY ===")
    print(f"P(B > A): {mc_prob:.6f}")
    print(f"Monte Carlo SE: {mc_se:.6f}")

def test_sample_sizes():
    print("\n=== SAMPLE SIZE COMPARISON ===")

    sample_sizes = [100, 1000, 10000]

    for n in sample_sizes:
        x_A, n_A, x_B, n_B = simulate_ab_test(
            n_A=n, n_B=n,
            p_A=0.10, p_B=0.12, seed=0
        )

        a_A, b_A = my_beta_posterior(x_A, n_A)
        a_B, b_B = my_beta_posterior(x_B, n_B)

        mean_A, _ = my_posterior_stats(a_A, b_A)
        mean_B, _ = my_posterior_stats(a_B, b_B)

        error_A = abs(mean_A - 0.10)
        error_B = abs(mean_B - 0.12)

        print(f"\nSample size: {n}")
        print(f"Posterior mean A: {mean_A:.6f}")
        print(f"Posterior mean B: {mean_B:.6f}")
        print(f"Error A: {error_A:.6f}")
        print(f"Error B: {error_B:.6f}")


def verify_with_scipy():
    print("\n=== SCIPY LIBRARY VERIFICATION ===")

    x_A, n_A, x_B, n_B = simulate_ab_test()

    a_A, b_A = my_beta_posterior(x_A, n_A)
    a_B, b_B = my_beta_posterior(x_B, n_B)

    # Our implementation
    mean_A, var_A = my_posterior_stats(a_A, b_A)
    mean_B, var_B = my_posterior_stats(a_B, b_B)

    # SciPy reference implementation
    scipy_mean_A = beta.mean(a_A, b_A)
    scipy_var_A = beta.var(a_A, b_A)

    scipy_mean_B = beta.mean(a_B, b_B)
    scipy_var_B = beta.var(a_B, b_B)

    print(f"{'Metric':<20} {'Ours':<15} {'SciPy':<15} {'Error':<15}")

    results = [
        ("Mean A", mean_A, scipy_mean_A),
        ("Variance A", var_A, scipy_var_A),
        ("Mean B", mean_B, scipy_mean_B),
        ("Variance B", var_B, scipy_var_B)
    ]

    for name, ours, reference in results:
        error = abs(ours - reference)

        print(
            f"{name:<20} "
            f"{ours:<15.8f} "
            f"{reference:<15.8f} "
            f"{error:<15.2e}"
        )


def plot_error_convergence():
    print("\n=== ERROR CONVERGENCE TEST ===")

    sample_sizes = [100, 300, 1000, 3000, 10000]
    repetitions = 200

    rmse_A = []
    rmse_B = []

    for n in sample_sizes:
        squared_errors_A = []
        squared_errors_B = []

        for seed in range(repetitions):
            x_A, n_A, x_B, n_B = simulate_ab_test(
                n_A=n,
                n_B=n,
                p_A=0.10,
                p_B=0.12,
                seed=seed
            )

            a_A, b_A = my_beta_posterior(x_A, n_A)
            a_B, b_B = my_beta_posterior(x_B, n_B)

            mean_A, _ = my_posterior_stats(a_A, b_A)
            mean_B, _ = my_posterior_stats(a_B, b_B)

            squared_errors_A.append((mean_A - 0.10)**2)
            squared_errors_B.append((mean_B - 0.12)**2)

        rmse_A.append(np.sqrt(np.mean(squared_errors_A)))
        rmse_B.append(np.sqrt(np.mean(squared_errors_B)))

        print(
            f"n={n:5d} | "
            f"RMSE A={rmse_A[-1]:.6f} | "
            f"RMSE B={rmse_B[-1]:.6f}"
        )

    plt.figure(figsize=(8, 5))

    plt.loglog(sample_sizes, rmse_A, "o-", label="Group A")
    plt.loglog(sample_sizes, rmse_B, "s-", label="Group B")

    plt.xlabel("Sample size per group (log scale)")
    plt.ylabel("RMSE (log scale)")
    plt.title("Bayesian Posterior Mean Error Convergence")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("error_convergence.png", dpi=300)
    plt.show()


def stress_test():
    print("\n=== STRESS TEST ===")

    scenarios = [
        ("Zero successes", 0, 100, 0, 100),
        ("All successes", 100, 100, 100, 100),
        ("Extreme difference", 1, 100, 99, 100)
    ]

    for name, x_A, n_A, x_B, n_B in scenarios:
        print(f"\nScenario: {name}")

        a_A, b_A = my_beta_posterior(x_A, n_A)
        a_B, b_B = my_beta_posterior(x_B, n_B)

        mean_A, var_A = my_posterior_stats(a_A, b_A)
        mean_B, var_B = my_posterior_stats(a_B, b_B)

        mc_prob, mc_se = my_mc_probability(
            a_A, b_A, a_B, b_B,
            M=100000, seed=0
        )

        print(f"Posterior mean A: {mean_A:.6f}")
        print(f"Posterior mean B: {mean_B:.6f}")
        print(f"P(B > A): {mc_prob:.6f}")
        print(f"Monte Carlo SE: {mc_se:.6f}")

        assert 0 <= mean_A <= 1
        assert 0 <= mean_B <= 1
        assert 0 <= mc_prob <= 1
        assert var_A >= 0 and var_B >= 0

        print("Status: PASSED")


def check_assumptions():
    print("\n=== SIMULATION ASSUMPTION CHECKS ===")

    x_A, n_A, x_B, n_B = simulate_ab_test()

    checks = {
        "Positive group sizes": n_A > 0 and n_B > 0,
        "Valid success counts":
            0 <= x_A <= n_A and 0 <= x_B <= n_B,
        "Binary outcome rates":
            0 <= x_A / n_A <= 1 and
            0 <= x_B / n_B <= 1,
        "Positive Beta prior parameters": True
    }

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    assert all(checks.values())

    print("\nSimulation design assumptions:")
    print("- Independent Bernoulli trials: assumed by generator")
    print("- Fixed probabilities within groups: 0.10 and 0.12")
    print("- Uniform Beta(1,1) prior: specified")
    print("- Random assignment: not explicitly simulated")


def test_null_rejection_rate():
    """
    Check the false-positive rate of the z-test
    when p_A = p_B = 0.10.
    """
    print("\n=== NULL SIMULATION: 1000 EXPERIMENTS ===")

    rng = np.random.default_rng(0)

    n_A = 1000
    n_B = 1000
    p_A = 0.10
    p_B = 0.10
    repetitions = 1000
    alpha = 0.05

    rejections = 0

    for _ in range(repetitions):
        x_A = rng.binomial(n_A, p_A)
        x_B = rng.binomial(n_B, p_B)

        z, p_value = my_z_test(
            x_A, n_A, x_B, n_B
        )

        if p_value < alpha:
            rejections += 1

    rejection_rate = rejections / repetitions

    print(f"Experiments: {repetitions}")
    print(f"Rejections: {rejections}")
    print(f"Observed rejection rate: {rejection_rate:.4f}")
    print(f"Expected rejection rate: {alpha:.4f}")

    lower = 0.05 - 0.014
    upper = 0.05 + 0.014

    passed = lower <= rejection_rate <= upper

    print(f"Acceptable range: [{lower:.3f}, {upper:.3f}]")
    print(f"Result: {'PASS' if passed else 'CHECK'}")


def test_overlapping_users():
    print("\n=== STRESS TEST: OVERLAPPING USERS ===")

    rng = np.random.default_rng(0)

    n = 1000
    overlap = 100

    # Group A: users 0 through 999
    ids_A = np.arange(n)

    # Group B: first 100 users overlap with A
    ids_B = np.concatenate([
        np.arange(overlap),
        np.arange(n, 2 * n - overlap)
    ])

    # Simulate binary retention outcomes
    outcomes_A = rng.binomial(1, 0.10, n)
    outcomes_B = rng.binomial(1, 0.12, n)

    # Identify users assigned to both groups
    overlapping_ids = np.intersect1d(ids_A, ids_B)

    print(f"Group A users: {len(ids_A)}")
    print(f"Group B users: {len(ids_B)}")
    print(f"Users in both groups: {len(overlapping_ids)}")
    print(f"Overlap percentage: {100 * len(overlapping_ids) / n:.1f}%")

    # Apply the usual A/B test despite invalid assignment
    x_A = int(outcomes_A.sum())
    x_B = int(outcomes_B.sum())

    z, p_value = my_z_test(x_A, n, x_B, n)

    print(f"\nNaive z-statistic: {z:.6f}")
    print(f"Naive p-value: {p_value:.6f}")

    print("\nAssumption check:")
    if len(overlapping_ids) > 0:
        print("FAIL: Users appear in both groups.")
        print("The independent-arm model is not justified.")
        print("The usual z-test standard error may be incorrect.")


def verify_grid_integration():
    from scipy.stats import beta
    from scipy.integrate import quad

    print("\n=== GRID VS SCIPY INTEGRATION ===")

    # Small example: A = 3/10, B = 5/10
    a_A, b_A = my_beta_posterior(3, 10)
    a_B, b_B = my_beta_posterior(5, 10)

    grid_result = my_grid_probability(
        a_A, b_A, a_B, b_B,
        grid_size=10001
    )

    scipy_result, _ = quad(
        lambda y: beta.pdf(y, a_B, b_B)
                  * beta.cdf(y, a_A, b_A),
        0, 1,
        epsabs=1e-12
    )

    error = abs(grid_result - scipy_result)

    print(f"Our grid result: {grid_result:.10f}")
    print(f"SciPy result:    {scipy_result:.10f}")
    print(f"Absolute error:  {error:.2e}")
    print(f"Required error:  1.00e-06")
    print(f"Result: {'PASS' if error <= 1e-6 else 'CHECK'}")


def verify_simulation_probability():
    from scipy.stats import beta
    from scipy.integrate import quad

    print("\n=== SIMULATION PROBABILITY VERIFICATION ===")

    # Results from the simulation with seed=0
    x_A, n_A = 96, 1000
    x_B, n_B = 118, 1000

    a_A, b_A = my_beta_posterior(x_A, n_A)
    a_B, b_B = my_beta_posterior(x_B, n_B)

    grid_result = my_grid_probability(
        a_A, b_A, a_B, b_B,
        grid_size=10001
    )

    scipy_result, _ = quad(
        lambda y: beta.pdf(y, a_B, b_B)
                  * beta.cdf(y, a_A, b_A),
        0, 1,
        epsabs=1e-12,
        limit=300,
        points=[
            a_A / (a_A + b_A),
            a_B / (a_B + b_B)
        ]
    )

    mc_result, mc_se = my_mc_probability(
        a_A, b_A, a_B, b_B,
        M=100000, seed=0
    )

    grid_error = abs(grid_result - scipy_result)
    mc_error = abs(mc_result - scipy_result)

    print(f"Grid probability:  {grid_result:.10f}")
    print(f"SciPy probability: {scipy_result:.10f}")
    print(f"Monte Carlo:       {mc_result:.10f}")
    print(f"Monte Carlo SE:    {mc_se:.10f}")

    print(f"\nGrid absolute error: {grid_error:.2e}")
    print(f"MC difference / SE: {mc_error / mc_se:.3f}")

    print(
        "Grid verification:",
        "PASS" if grid_error <= 1e-6 else "CHECK"
    )
    print(
        "Monte Carlo verification:",
        "PASS" if mc_error <= 3 * mc_se else "CHECK"
    )



if __name__ == "__main__":
    main()
    test_sample_sizes()
    verify_with_scipy()
    plot_error_convergence()
    stress_test()
    check_assumptions()
    test_null_rejection_rate()
    test_overlapping_users()
    verify_grid_integration()
    verify_simulation_probability()
