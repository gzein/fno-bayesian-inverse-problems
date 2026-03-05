import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import matplotlib.pyplot as plt


class InverseBayesDarcySolver1D:
    """Inverse Bayes solver for 1D Darcy flow problem."""
    
    def __init__(self, domain_length=1.0, n_cells=50, prior_mean=0.0, prior_std=1.0):
        """
        Initialize the solver.
        
        Parameters:
        - domain_length: Length of domain
        - n_cells: Number of cells for discretization
        - prior_mean: Prior mean for log-permeability
        - prior_std: Prior standard deviation for log-permeability
        """
        self.L = domain_length
        self.n = n_cells
        self.dx = self.L / self.n
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        
    def forward_model(self, log_k, boundary_left=1.0, boundary_right=0.0):
        """
        Solve 1D Darcy equation: -d/dx(k*dp/dx) = 0
        Using finite differences.
        """
        k = np.exp(log_k)
        
        # System matrix
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            if i == 0:
                A[i, i] = 1.0
            elif i == self.n - 1:
                A[i, i] = 1.0
            else:
                k_left = (k[i-1] + k[i]) / 2
                k_right = (k[i] + k[i+1]) / 2
                A[i, i-1] = -k_left / (self.dx**2)
                A[i, i] = (k_left + k_right) / (self.dx**2)
                A[i, i+1] = -k_right / (self.dx**2)
        
        # Boundary conditions
        b = np.zeros(self.n)
        b[0] = boundary_left
        b[-1] = boundary_right
        
        # Solve for pressure
        p = np.linalg.solve(A, b)
        return p
    
    def log_likelihood(self, log_k, pressure_obs, obs_indices, noise_std=0.01):
        """Calculate log-likelihood of observations."""
        p_pred = self.forward_model(log_k)
        p_obs_pred = p_pred[obs_indices]
        
        residuals = pressure_obs - p_obs_pred
        log_like = -0.5 * np.sum((residuals / noise_std)**2)
        return log_like
 
    def log_prior(self, log_k):
        """Calculate log-prior (Gaussian prior)."""
        return -0.5 * np.sum(((log_k - self.prior_mean) / self.prior_std)**2)

    def log_posterior(self, log_k, pressure_obs, obs_indices, noise_std=0.01):
        """Calculate log-posterior."""
        return self.log_likelihood(log_k, pressure_obs, obs_indices, noise_std) + self.log_prior(log_k)

    def invert(self, pressure_obs, obs_indices, noise_std=0.01, initial_guess=None):
        """
        Invert for log-permeability using MAP estimation.
        """
        if initial_guess is None:
            initial_guess = np.ones(self.n) * self.prior_mean
        
        def objective(log_k):
            return -self.log_posterior(log_k, pressure_obs, obs_indices, noise_std)
        
        result = minimize(objective, initial_guess, method='L-BFGS-B')
        return result.x, result.fun


# Example usage
if __name__ == "__main__":
    # Setup solver
    solver = InverseBayesDarcySolver1D(domain_length=1.0, n_cells=50, prior_std=0.5)
    
    # True permeability (log scale)
    true_log_k = np.sin(np.pi * np.linspace(0, 1, solver.n))
    
    # Forward solve to generate synthetic observations
    true_pressure = solver.forward_model(true_log_k)
    obs_indices = np.array([10, 20, 30, 40])
    pressure_obs = true_pressure[obs_indices] + np.random.normal(0, 0.01, len(obs_indices))
    
    # Invert
    inverted_log_k, misfit = solver.invert(pressure_obs, obs_indices, noise_std=0.01)
    inverted_pressure = solver.forward_model(inverted_log_k)
    
    # Plot results
    x = np.linspace(0, 1, solver.n)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, true_log_k, 'b-', label='True log(k)', linewidth=2)
    plt.plot(x, inverted_log_k, 'r--', label='Inverted log(k)', linewidth=2)
    plt.xlabel('Position')
    plt.ylabel('log(Permeability)')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(x, true_pressure, 'b-', label='True pressure', linewidth=2)
    plt.plot(x, inverted_pressure, 'r--', label='Predicted pressure', linewidth=2)
    plt.plot(x[obs_indices], pressure_obs, 'ko', markersize=8, label='Observations')
    plt.xlabel('Position')
    plt.ylabel('Pressure')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()