# Finite Difference Solver for 1D Darcy Equation. This code was taken from Claude, so make sure to modify and make it your own.

# finite difference tldr: discretise into grid, in this case points on [0, 1] but this doesn't really matter thanks to translation invariance,
# and then treat the differences as a system of algebraic equations to solve for using, for example, matrix methods.

import numpy as np
import matplotlib.pyplot as plt


def solve_darcy_1d_fd(K, f, boundary_conditions, domain_length=1.0, num_points=101):
    """
    Solve the 1D Darcy equation using finite differences: -d/dx(K dp/dx) = f
    
    Parameters:
    -----------
    K : float, array, or callable
        Permeability coefficient
        - float: constant permeability
        - array: permeability at each grid point
        - callable: function K(x) that returns permeability
    f : float, array, or callable
        Source term
        - float: constant source
        - array: source at each grid point
        - callable: function f(x) that returns source
    boundary_conditions : dict
        Dictionary with 'left' and 'right' keys containing boundary values
        Example: {'left': 0.0, 'right': 1.0}
    domain_length : float
        Length of the 1D domain
    num_points : int
        Number of grid points (including boundaries)
    
    Returns:
    --------
    x : ndarray
        Grid points
    p : ndarray
        Solution (pressure) at each grid point
    """
    
    # Create grid
    x = np.linspace(0, domain_length, num_points)
    dx = x[1] - x[0]
    
    # Convert K to array if needed
    if callable(K):
        K_vals = K(x)
    elif np.isscalar(K):
        K_vals = K * np.ones(num_points)
    else:
        K_vals = np.array(K)
    
    # Convert f to array if needed
    if callable(f):
        f_vals = f(x)
    elif np.isscalar(f):
        f_vals = f * np.ones(num_points)
    else:
        f_vals = np.array(f)
    
    # Build coefficient matrix A and RHS vector b
    # Using centered differences: -d/dx(K dp/dx) ≈ -(K_{i+1/2}(p_{i+1}-p_i) - K_{i-1/2}(p_i-p_{i-1}))/dx²
    
    n = num_points
    A = np.zeros((n, n))
    b = f_vals.copy()
    
    # Interior points (i = 1, 2, ..., n-2)
    for i in range(1, n-1):
        # K at midpoints
        K_right = 0.5 * (K_vals[i] + K_vals[i+1])
        K_left = 0.5 * (K_vals[i] + K_vals[i-1])
        
        A[i, i-1] = -K_left / dx**2
        A[i, i] = (K_left + K_right) / dx**2
        A[i, i+1] = -K_right / dx**2
    
    # Boundary conditions
    # Left boundary (i = 0)
    A[0, 0] = 1.0
    b[0] = boundary_conditions['left']
    
    # Right boundary (i = n-1)
    A[n-1, n-1] = 1.0
    b[n-1] = boundary_conditions['right']
    
    # Solve the system
    p = np.linalg.solve(A, b)
    
    return x, p


def interpolator(x, p):
    def interp_func(x_new):
        return np.interp(x_new, x, p)
    return interp_func



if __name__ == "__main__":
    # Example 1: Constant permeability, constant source
    print("Example 1: Constant K and f")
    K = 1.0
    f = 1.0
    bc = {'left': 0.0, 'right': 0.0}

    x1, p1 = solve_darcy_1d_fd(K, f, bc, domain_length=1.0, num_points=51)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(x1, p1, 'b-', linewidth=2, label='Numerical')
    # Analytical solution: p(x) = 0.5*x*(1-x) for K=1, f=1, p(0)=p(1)=0
    p_exact = 0.5 * x1 * (1 - x1)
    plt.plot(x1, p_exact, 'r--', linewidth=1, label='Analytical', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Example 1: Constant K=1, f=1')
    plt.legend()
    plt.grid(True)


    # Example 2: Variable permeability
    print("\nExample 2: Variable permeability")
    K_func = lambda x: 1.0 + x**2
    f = 0.0
    bc = {'left': 0.0, 'right': 1.0}

    x2, p2 = solve_darcy_1d_fd(K_func, f, bc, domain_length=1.0, num_points=51)

    plt.subplot(1, 3, 2)
    plt.plot(x2, p2, 'r-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Example 2: K=1+x², f=0')
    plt.grid(True)


    # Example 3: Localized source
    print("\nExample 3: Localized source term")
    K = 1.0
    f_func = lambda x: 10 * np.exp(-100 * (x - 0.5)**2)
    bc = {'left': 0.0, 'right': 0.0}

    x3, p3 = solve_darcy_1d_fd(K, f_func, bc, domain_length=1.0, num_points=101)

    plt.subplot(1, 3, 3)
    plt.plot(x3, p3, 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Example 3: Gaussian source')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('darcy_1d_solutions.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nSolutions computed successfully!")
    print(f"Example 1 max pressure: {np.max(p1):.4f}")
    print(f"Example 2 max pressure: {np.max(p2):.4f}")
    print(f"Example 3 max pressure: {np.max(p3):.4f}")

    # Verify Example 1 against analytical solution
    error = np.max(np.abs(p1 - p_exact))
    print(f"\nExample 1 max error vs analytical: {error:.6f}")