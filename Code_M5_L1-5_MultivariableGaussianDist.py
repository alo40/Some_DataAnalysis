# city 1
u1 = 60  # mean
s1 = 10  # std-var

# city 2
x2 = 75  # temperature
u2 = 90  # mean
s2 = 20  # std-var

cov12 = 100  # covariance city1 - city2

# correction after measuring temperature in city 2
u1_corr = u1 + cov12 / s2**2 * (x2 - u2)
s1_corr = s1**2 - cov12 / s2**2

print(f"u1 corrected = {u1_corr}")
print(f"s1 corrected = {s1_corr}")