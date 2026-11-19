
import numpy as np

def my_beta_posterior(x, n, a=1.0, b=1.0):
    """
    Calculate Beta posterior parameters.

    x: number of successes (clicks/returns)
    n: total number of trials (visitors)
    a, b: prior parameters
    """
    if not (0 <= x <= n):
        raise ValueError("Successes must be between 0 and n.")

    if a <= 0 or b <= 0:
        raise ValueError("Prior parameters must be positive.")

    a_post = a + x
    b_post = b + n - x

    return a_post, b_post


def my_posterior_stats(a_post, b_post):
    """
    Compute the mean and variance of a Beta posterior.
    """
    total = a_post + b_post

    mean = a_post / total
    variance = (a_post * b_post) / (total**2 * (total + 1))

    return mean, variance

# Equation:
# P(pB > pA | data) ≈ (1/M) * sum(I(pB_i > pA_i))
# SE ≈ sqrt(P_hat * (1 - P_hat) / M)

def my_mc_probability(a_A, b_A, a_B, b_B,
                      M=100000, seed=0):
    """
    Estimate P(pB > pA | data) using Monte Carlo sampling.

    Returns:
        probability: estimated probability B is better than A
        standard_error: Monte Carlo standard error
    """
    rng = np.random.default_rng(seed)

    samples_A = rng.beta(a_A, b_A, size=M)
    samples_B = rng.beta(a_B, b_B, size=M)

    indicators = (samples_B > samples_A).astype(float)

    probability = np.mean(indicators)

    standard_error = np.sqrt(
        probability * (1 - probability) / M
    )

    return probability, standard_error


def my_credible_interval(a_A, b_A, a_B, b_B,
                         M=200000, seed=0, level=0.95):
    """
    Compute an equal-tailed credible interval
    for the difference p_B - p_A.
    """
    if not 0 < level < 1:
        raise ValueError("Credible level must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    samples_A = rng.beta(a_A, b_A, size=M)
    samples_B = rng.beta(a_B, b_B, size=M)

    differences = samples_B - samples_A

    alpha = 1 - level

    lower = np.quantile(differences, alpha / 2)
    upper = np.quantile(differences, 1 - alpha / 2)

    return lower, upper




# Equation:
# z = (pB_hat - pA_hat) /
#     sqrt(p_pool * (1-p_pool) * (1/nA + 1/nB))

def my_z_test(x_A, n_A, x_B, n_B):
    """
    Calculate the two-proportion z statistic.

    Returns:
        z: test statistic
        p_value: two-sided p-value
    """
    from math import erfc, sqrt

    if n_A <= 0 or n_B <= 0:
        raise ValueError("Group sizes must be positive.")

    if not (0 <= x_A <= n_A and 0 <= x_B <= n_B):
        raise ValueError("Invalid success counts.")

    p_A = x_A / n_A
    p_B = x_B / n_B

    p_pool = (x_A + x_B) / (n_A + n_B)

    se = np.sqrt(
        p_pool * (1 - p_pool) *
        (1 / n_A + 1 / n_B)
    )

    if se == 0:
        raise ValueError("Pooled standard error is zero.")

    z = (p_B - p_A) / se

    # Two-sided normal-tail probability
    p_value = erfc(abs(z) / sqrt(2))

    return z, p_value


# Equation:
# P(pB > pA) = integral_0^1 fB(y) * FA(y) dy
# Numerical integration using the trapezoidal rule.

def my_grid_probability(a_A, b_A, a_B, b_B, grid_size=10001):
    """
    Estimate P(pB > pA) using numerical integration.

    Uses a grid and the trapezoidal rule.
    """
    from math import lgamma

    if min(a_A, b_A, a_B, b_B) <= 0:
        raise ValueError("Beta parameters must be positive.")
    if grid_size < 3:
        raise ValueError("Grid size must be at least 3.")

    grid = np.linspace(0, 1, grid_size)
    dx = grid[1] - grid[0]

    def beta_pdf(x, a, b):
        log_constant = lgamma(a + b) - lgamma(a) - lgamma(b)
        result = np.zeros_like(x)

        interior = (x > 0) & (x < 1)
        t = x[interior]

        result[interior] = np.exp(
            log_constant
            + (a - 1) * np.log(t)
            + (b - 1) * np.log1p(-t)
        )

        # Boundary values for parameters >= 1
        if a == 1:
            result[0] = np.exp(log_constant) if b >= 1 else 0
        if b == 1:
            result[-1] = np.exp(log_constant) if a >= 1 else 0

        return result

    pdf_A = beta_pdf(grid, a_A, b_A)
    pdf_B = beta_pdf(grid, a_B, b_B)

    # Cumulative trapezoidal integration for F_A(y)
    cdf_A = np.zeros(grid_size)
    cdf_A[1:] = np.cumsum(
        (pdf_A[:-1] + pdf_A[1:]) * dx / 2
    )

    # Integrate f_B(y) * F_A(y)
    integrand = pdf_B * cdf_A

    probability = np.sum(
        (integrand[:-1] + integrand[1:]) * dx / 2
    )

    return probability




if __name__ == "__main__":
    a_A, b_A = my_beta_posterior(3, 10)
    a_B, b_B = my_beta_posterior(5, 10)

    mc_prob, mc_se = my_mc_probability(
        a_A, b_A, a_B, b_B,
        M=100000, seed=0
    )

    grid_prob = my_grid_probability(
        a_A, b_A, a_B, b_B
    )

    print("Monte Carlo probability:", round(mc_prob, 6))
    print("Numerical integration:", round(grid_prob, 6))
    print("Absolute difference:", round(abs(mc_prob - grid_prob), 6))
    print("Monte Carlo SE:", round(mc_se, 6))




