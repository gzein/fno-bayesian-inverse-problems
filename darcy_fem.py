"""
FEniCSx forward solver for Darcy's Law.
Solves: -div(K * grad(u)) = 0 on [0,1]^2
Left BC: u = 1, Right BC: u = 0, Top/Bottom: no-flux (natural)
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem.petsc import LinearProblem
import ufl


def solve_darcy(K_array, u_left=1.0, u_right=0.0):
    """
    Solve Darcy's law given a permeability field K_array.

    Parameters
    ----------
    K_array : 2D numpy array, shape (n, n)
        Permeability field. Works for both continuous and discrete 0-1 fields.
    u_left : float
        Pressure on left boundary (x=0).
    u_right : float
        Pressure on right boundary (x=1).

    Returns
    -------
    u_cells : 2D numpy array, shape (n, n)
        Pressure field evaluated at cell centres of the K grid.
        Same spatial resolution as K_array, ready for direct
        comparison with your FNO outputs.
    """
    n = K_array.shape[0]

    # --- Mesh ---
    # Quadrilateral mesh matches your grid-based permeability data naturally.
    # n+1 nodes per side gives n elements per side, one per K cell.
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD,
        n, n,
        mesh.CellType.quadrilateral
    )

    # --- Function spaces ---
    # Q1 (bilinear) for pressure — standard for elliptic problems
    V = fem.functionspace(domain, ("Lagrange", 1))

    # DG0 (piecewise constant) for permeability — correct for discontinuous K
    # This is the key choice: DG0 does NOT interpolate across sharp edges
    V_K = fem.functionspace(domain, ("DG", 0))
    K_func = fem.Function(V_K)

    # --- Interpolate K_array onto mesh ---
    # FEniCSx passes physical coordinates x with shape (3, n_points)
    def K_values(x):
        # Map physical coords in [0,1] to K_array indices
        col = np.clip((x[0] * n).astype(int), 0, n - 1)  # x -> column
        row = np.clip((x[1] * n).astype(int), 0, n - 1)  # y -> row
        vals = K_array[row, col].astype(float)
        # Clamp to avoid singular system where K=0 exactly
        return np.clip(vals, 1e-10, None)

    K_func.interpolate(K_values)

    # --- Variational formulation ---
    # Weak form: find u in V such that for all v in V:
    #   integral(K * grad(u) . grad(v) dx) = 0
    u_trial = ufl.TrialFunction(V)
    v_test  = ufl.TestFunction(V)

    a = K_func * ufl.dot(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L = fem.Constant(domain, 0.0) * v_test * ufl.dx

    # --- Dirichlet boundary conditions ---
    def left_boundary(x):
        return np.isclose(x[0], 0.0)

    def right_boundary(x):
        return np.isclose(x[0], 1.0)

    left_dofs  = fem.locate_dofs_geometrical(V, left_boundary)
    right_dofs = fem.locate_dofs_geometrical(V, right_boundary)

    bc_left  = fem.dirichletbc(
        fem.Constant(domain, float(u_left)),  left_dofs,  V
    )
    bc_right = fem.dirichletbc(
        fem.Constant(domain, float(u_right)), right_dofs, V
    )

    # --- Solve ---
    # LU direct solver: exact, fast for small-medium grids (up to ~64x64)
    # For larger grids switch to: "ksp_type": "cg", "pc_type": "hypre"
    problem = LinearProblem(
        a, L,
        bcs=[bc_left, bc_right],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    u_sol = problem.solve()

    # --- Evaluate at cell centres ---
    # Returns array at same resolution as K_array for direct MCMC comparison
    u_cells = _evaluate_at_cell_centres(u_sol, domain, n)
    return u_cells


def _evaluate_at_cell_centres(u_sol, domain, n):
    """
    Evaluate FEniCSx solution u_sol at the centre of each K grid cell.
    Returns numpy array of shape (n, n).
    """
    # Cell centre coordinates
    coords_1d = (np.arange(n) + 0.5) / n
    cx, cy = np.meshgrid(coords_1d, coords_1d)
    points = np.column_stack([
        cx.ravel(),
        cy.ravel(),
        np.zeros(n * n)   # z=0 for 2D
    ])

    # FEniCSx point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)

    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(
        domain, cell_candidates, points
    )

    # Collect valid points and their cells
    valid_points = []
    valid_cells  = []
    for i in range(len(points)):
        cells_for_point = colliding_cells.links(i)
        if len(cells_for_point) > 0:
            valid_points.append(points[i])
            valid_cells.append(cells_for_point[0])

    u_vals = u_sol.eval(np.array(valid_points), np.array(valid_cells))
    return u_vals.reshape(n, n)


def log_likelihood_fem(x_tensor, y_tensor, sigma=1.0):
    """
    Drop-in replacement for your FNO log_likelihood.

    Parameters
    ----------
    x_tensor : torch.Tensor, shape (1, 1, n, n)
        Proposed permeability field (your theta_proposal_tensor)
    y_tensor : torch.Tensor, shape (1, 1, n, n)
        Observed flow field (your true_observations)
    sigma : float
        Observation noise standard deviation

    Returns
    -------
    float : log p(y | K)
    """
    K   = x_tensor[0, 0].cpu().numpy()
    y   = y_tensor[0, 0].cpu().numpy()

    try:
        u_cells = solve_darcy(K)
        residuals = y - u_cells
        return -0.5 * np.sum(residuals**2) / sigma**2
    except Exception as e:
        print(f"FEM solve failed: {e}")
        return -np.inf