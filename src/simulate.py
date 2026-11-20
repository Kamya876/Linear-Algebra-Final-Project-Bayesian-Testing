"""Known-truth simulation for Project 7 (same model, parameters and seed as Iteration 2).

simulate_ab(n, seed) draws x_A ~ Binomial(n, p_A) then x_B ~ Binomial(n, p_B) from
numpy.random.default_rng(seed), exactly as in the Iteration 2 starter code, so
seed 0 with n = 1000 gives x_A = 92, x_B = 121 in BOTH iterations.
"""
import numpy as np

P_A_TRUE, P_B_TRUE = 0.10, 0.12


def simulate_ab(n=1000, seed=0, p_A=P_A_TRUE, p_B=P_B_TRUE):
    """Return (x_A, n_A, x_B, n_B) together with the true parameters.

    Output: (x_A, n_A, x_B, n_B), truth   where truth = {"p_A": ..., "p_B": ...}
    """
    rng = np.random.default_rng(seed)
    x_A = int(rng.binomial(n, p_A))
    x_B = int(rng.binomial(n, p_B))
    return (x_A, n, x_B, n), {"p_A": p_A, "p_B": p_B}


def simulate_ab_test(n_A=1000, n_B=1000, p_A=P_A_TRUE, p_B=P_B_TRUE, seed=0):
    """Backward-compatible wrapper allowing unequal group sizes."""
    rng = np.random.default_rng(seed)
    return int(rng.binomial(n_A, p_A)), n_A, int(rng.binomial(n_B, p_B)), n_B


def simulate_shared_users(n=1000, overlap=100, p=0.10, reps=10000, seed=0):
    """Stress test: `overlap` users are in BOTH arms and keep the same outcome.

    Under H0 (p_A = p_B = p) each arm has n users; the `overlap` shared users
    contribute the same 0/1 outcome to both arms, so p_hat_A and p_hat_B are
    positively correlated (Cov = overlap * p(1-p) / n^2) and the independent-arm
    standard error is too large.  Returns arrays x_A, x_B of length `reps`.
    """
    rng = np.random.default_rng(seed)
    shared = rng.binomial(overlap, p, size=reps)
    only_A = rng.binomial(n - overlap, p, size=reps)
    only_B = rng.binomial(n - overlap, p, size=reps)
    return shared + only_A, shared + only_B


if __name__ == "__main__":
    data, truth = simulate_ab()
    print("x_A, n_A, x_B, n_B =", data, "| truth:", truth)
