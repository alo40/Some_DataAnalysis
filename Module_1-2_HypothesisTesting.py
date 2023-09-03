import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, hypergeom


# contingency table ("success" and "failure" are generic words for the outcome of the Bernoulli trial)
controlGroup_success = 63
controlGroup_failure = 30937
testingGroup_success = 39
testingGroup_failure = 30961
contingency_table = np.array([[testingGroup_success, controlGroup_success],
                              [testingGroup_failure, controlGroup_failure]])

# Fisher exact test using fisher_exact method
p_value_Fisher = fisher_exact(contingency_table, alternative='less')
print(f"The p-value of the Hypothesis Test using fisher_exact method is {p_value_Fisher[1]:.8}")

# parameters for hypergeometric distribution
n = 31_000
M = 31_000 + n
k = 39
N = 63 + k

# Fisher exact test using hypergeometric method
rv = hypergeom(M, n, N)
x = np.arange(0, k)
p_value_Hypergeom = sum(rv.pmf(x))
print(f"The p-value of the Hypothesis Test using hypergeom method is {p_value_Hypergeom:.8}")

