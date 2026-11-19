
from methods import (
    my_beta_posterior,
    my_posterior_stats,
    my_mc_probability,
    my_z_test,
    my_grid_probability
)

def main():
    # Iteration 3, Project 7: Small Worked Example
    x_A, n_A = 3, 10
    x_B, n_B = 5, 10

    print("=== STEP 1: OBSERVED DATA ===")
    print(f"A: {x_A} successes out of {n_A}")
    print(f"B: {x_B} successes out of {n_B}")

    # Step 2: Beta posterior
    a_A, b_A = my_beta_posterior(x_A, n_A)
    a_B, b_B = my_beta_posterior(x_B, n_B)

    print("\n=== STEP 2: BETA POSTERIORS ===")
    print(f"A: Beta({a_A}, {b_A})")
    print(f"B: Beta({a_B}, {b_B})")

    # Step 3: Posterior statistics
    mean_A, var_A = my_posterior_stats(a_A, b_A)
    mean_B, var_B = my_posterior_stats(a_B, b_B)

    print("\n=== STEP 3: POSTERIOR STATISTICS ===")
    print(f"A: mean={mean_A:.6f}, variance={var_A:.6f}")
    print(f"B: mean={mean_B:.6f}, variance={var_B:.6f}")

    # Step 4: Monte Carlo
    mc_prob, mc_se = my_mc_probability(
        a_A, b_A, a_B, b_B, M=100000, seed=0
    )

    print("\n=== STEP 4: MONTE CARLO ===")
    print(f"P(B > A) = {mc_prob:.6f}")
    print(f"Standard error = {mc_se:.6f}")

    # Step 5: Frequentist z-test
    z, p_value = my_z_test(x_A, n_A, x_B, n_B)

    print("\n=== STEP 5: TWO-PROPORTION Z-TEST ===")
    print(f"Z-statistic = {z:.6f}")
    print(f"P-value = {p_value:.6f}")

    # Step 6: Numerical integration
    grid_prob = my_grid_probability(a_A, b_A, a_B, b_B)

    print("\n=== STEP 6: NUMERICAL VERIFICATION ===")
    print(f"Grid probability = {grid_prob:.6f}")
    print(f"MC difference = {abs(mc_prob-grid_prob):.6f}")
    print(f"Difference / MC SE = {abs(mc_prob-grid_prob)/mc_se:.3f}")

if __name__ == "__main__":
    main()
