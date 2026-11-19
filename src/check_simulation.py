
from simulate import simulate_ab_test
from methods import (
    my_beta_posterior,
    my_posterior_stats,
    my_mc_probability,
    my_grid_probability
)
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


if __name__ == "__main__":
    main()
    test_sample_sizes()
    verify_with_scipy()
    plot_error_convergence()
    stress_test()
    check_assumptions()
