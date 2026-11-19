
import numpy as np


def simulate_ab_test(n_A=1000, n_B=1000,
                     p_A=0.10, p_B=0.12, seed=0):
    """
    Simulate an A/B test with known conversion rates.

    Returns the number of successes and total
    observations for both groups.
    """

    rng = np.random.default_rng(seed)

    # Generate binary outcomes:
    # 1 = success, 0 = failure
    outcomes_A = rng.binomial(1, p_A, size=n_A)
    outcomes_B = rng.binomial(1, p_B, size=n_B)

    # Count successes
    x_A = int(np.sum(outcomes_A))
    x_B = int(np.sum(outcomes_B))

    return x_A, n_A, x_B, n_B


if __name__ == "__main__":
    x_A, n_A, x_B, n_B = simulate_ab_test()

    print("=== SIMULATED A/B TEST ===")

    print(f"Group A: {x_A}/{n_A} successes")
    print(f"Group B: {x_B}/{n_B} successes")

    print(f"Observed rate A: {x_A/n_A:.4f}")
    print(f"Observed rate B: {x_B/n_B:.4f}")

    print("True rate A: 0.10")
    print("True rate B: 0.12")
