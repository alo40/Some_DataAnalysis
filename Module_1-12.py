from scipy.special import comb

n = 31000
k = 63
p = k / n
f = comb(n, k) * (p ** k) * (1 - p) ** (n - k)
print(f"Probability of {k} ocurrences in {n} cases is {f}")