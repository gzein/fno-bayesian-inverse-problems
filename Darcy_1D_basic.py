# our theta is from a subset of L^2(R, R) - limitation: you can only have R^D.
# estimating the sample standard deviation is a whole other issue. There are multiple fixes
# 1. Treat it as an unkown, so run bayesian inference on it too
# 2. Empirical estimate from data, if you have a good guess for theta
# 3. Cross-validation
# 4. Domain knowledge
#
# Here, we assume we know it

# Diagnostics:
# 1. My theta is biased!
# Solutions: data too noisy (sigma too big), not enough datapoints (n_samples too low), burn in period too low (however this typically is not the issue)
# Can also experiment with a more informed prior

from _helper_FDM_Darcy_1D_basic import solve_darcy_1d_fd, interpolator
import numpy as np
import matplotlib.pyplot as plt


# remember to modify this function when you modify the true_theta, g, bc 
def loglikelihood(theta, X, Y, noise_std=0.01):
    """Compute log-likelihood of observations given theta."""
    theta_func = lambda x: np.exp(theta * x**2)
    fmap = interpolator(*solve_darcy_1d_fd(theta_func, 1.0, 
                        {"left": 0.0, "right": 1.0}, 
                        domain_length=1.0, num_points=101))
    Y_pred = fmap(X)
    residuals = Y - Y_pred
    ll = -0.5 * np.sum(residuals**2) / (noise_std**2)
    return ll

# set seeds 

k = 2
true_theta = lambda x: np.exp(k*x**2)
g = 1.0
bc = {"left": 0.0, "right": 1.0}
sigma = 0.05
n_samples = 40

# generate synthetic observations

true_fmap = interpolator(*solve_darcy_1d_fd(true_theta, g, bc, domain_length=1.0, num_points=101))

X = np.random.rand(n_samples)
Y = true_fmap(X) + np.random.normal(0, sigma, size=X.shape)


# Test how sensitive your predictions are to theta changes
test_thetas = np.linspace(k-0.5, k+0.5, 100)
test_lls = []
for t in test_thetas:
    ll = loglikelihood(t, X, Y, noise_std=sigma)
    test_lls.append(ll)

plt.plot(test_thetas, test_lls)
plt.axvline(x=k, color='r', linestyle='--', label='True value')
plt.xlabel('theta')
plt.ylabel('Log-likelihood')
plt.legend()
plt.show()

# plot_XY(X, Y)

# Our prior is that k ~ N(0, 1)
# delta, n_iterations are taken from Nickl [90], "Consistent Inversion of Noisy Non-Abelian X-Ray Transforms"
delta = 0.000025
n_iterations = 100_000  # this is essentially a burn in period
theta_m = np.random.normal(0, 1, size=1)  # initial proposal
accepted = 0  # counter for accepted proposals
# for i in range(n_iterations):
#     xi = np.random.normal(0, 1, size=1)
#     s_m = np.sqrt(1 - 2*delta) * theta_m + np.sqrt(2*delta) * xi
#     # now compute the acceptance probability
#     ll_sm = loglikelihood(s_m, X, Y)
#     ll_tm = loglikelihood(theta_m, X, Y)
#     acceptance_p = min(1, np.exp(ll_sm - ll_tm))

#     u = np.random.rand()
#     if u < acceptance_p:
#         accepted += 1
#         theta_m = s_m  # accept
#     # else reject
#     if i % 1000 == 0:
#         print(f"Iteration {i}/{n_iterations} complete.")

# print(f"Acceptance rate: {accepted/n_iterations:.2f}")

# Track theta values and log-likelihoods
theta_history = []
ll_history = []
acceptance_history = []

for i in range(n_iterations):
    xi = np.random.normal(0, 1, size=1)
    s_m = np.sqrt(1 - 2*delta) * theta_m + np.sqrt(2*delta) * xi
    
    ll_sm = loglikelihood(s_m, X, Y, noise_std=sigma)
    ll_tm = loglikelihood(theta_m, X, Y, noise_std=sigma)
    acceptance_p = min(1, np.exp(ll_sm - ll_tm))
    
    u = np.random.rand()
    if u < acceptance_p:
        accepted += 1
        theta_m = s_m
    
    theta_history.append(theta_m[0])
    ll_history.append(ll_tm)
    acceptance_history.append(acceptance_p)
    
    if i % 10000 == 0:
        print(f"Iter {i}: theta={theta_m[0]:.4f}, ll={ll_tm:.4f}, "
              f"recent accept rate={np.mean(acceptance_history[-1000:]):.3f}")

# Plot diagnostics
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(theta_history)
plt.axhline(y=k, color='r', linestyle='--', label='True value')
plt.xlabel('Iteration')
plt.ylabel('theta')
plt.legend()

plt.subplot(122)
plt.plot(theta_history[90000:])  # Last 10k iterations
plt.axhline(y=k, color='r', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('theta')
plt.title('Burn-in period')
plt.tight_layout()
plt.show()

plt.subplot(132)
plt.plot(ll_history)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')

plt.subplot(133)
plt.hist(theta_history[10000:], bins=50)  # After burn-in
plt.axvline(x=k, color='r', linestyle='--', label='True value')
plt.xlabel('theta')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()


# Now compute the ergodic average
J = 1000
total = 0
for i in range(J):
    xi = np.random.normal(0, 1, size=1)
    s_m = np.sqrt(1 - 2*delta) * theta_m + np.sqrt(2*delta) * xi
    # now compute the acceptance probability
    ll_sm = loglikelihood(s_m, X, Y)
    ll_tm = loglikelihood(theta_m, X, Y)
    acceptance_p = min(1, np.exp(ll_sm - ll_tm))

    u = np.random.rand()
    if u < acceptance_p:
        theta_m = s_m  # accept
    # else reject
    total += theta_m

print(total/J)
