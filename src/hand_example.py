"""Reproduce the small hand-worked example in docs/main.tex."""
from .common import beta_posterior_parameters, beta_posterior_mean, two_proportion_z_test

a0 = b0 = 1
x_a, n_a = 2, 10
x_b, n_b = 4, 10

aa, ba = beta_posterior_parameters(a0, b0, x_a, n_a)
ab, bb = beta_posterior_parameters(a0, b0, x_b, n_b)
print("A posterior:", f"Beta({aa},{ba})", "mean =", beta_posterior_mean(aa, ba))
print("B posterior:", f"Beta({ab},{bb})", "mean =", beta_posterior_mean(ab, bb))
print("Frequentist:", two_proportion_z_test(x_a, n_a, x_b, n_b))
