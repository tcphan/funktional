import numpy as np
from scipy.linalg import solve
from fda.basis import Basis
from fda.representation import FunctionalData


def smooth_data(t, y, basis: Basis, lmbda: float=0.0, penalty_order: int=2):
    """Convert discrete observations into a FunctionalData object using penalized least squares.
    
    Fits discrete observations to a basis expansion by minimizing:
    SSE(c) = \sum_j (y_{ij} - x_i(t_j))^2 + \lambda \int_a^b [x_i^{(m)}(t)]^2 dt
    
    where x_i(t) = \sum_k c_{ik} \phi_k(t) and m is the penalty_order.
    
    Parameters
    ----------
    t : array-like of shape (n_points,) or list of array-like
        Observation times. If a single 1D array, all curves are assumed
        to be observed at the same time points. If a list of 1D arrays,
        each curve can be observed at different time points.
    y : array-like or list of array-like
        Observed values.
        - If `t` is a single 1D array, `y` must be a 1D array of shape (n_points,)
          (for a single curve) or a 2D array of shape (n_curves, n_points).
        - If `t` is a list, `y` must be a list of 1D arrays of corresponding sizes.
    basis : Basis
        The basis system to project the data onto (e.g. BSplineBasis, FourierBasis).
    lmbda : float, default=0.0
        Smoothing parameter (lambda). If 0.0, performs standard least squares fit.
    penalty_order : int, default=2
        The order of the derivative to penalize in the roughness penalty.
        
    Returns
    -------
    FunctionalData
        A FunctionalData object containing the smoothed curves.
    """
    
    if lmbda < 0:
        raise ValueError("Smoothing parameter lmbda must be non-negative.")
        
    # Check if we have individual grids for each curve
    if isinstance(t, list) or (isinstance(t, np.ndarray) and t.dtype == object):
        if not isinstance(y, (list, np.ndarray)):
            raise ValueError("y must be a list or array of arrays if t is a list.")
        if len(t) != len(y):
            raise ValueError("Lengths of t and y lists must match.")
            
        n_curves = len(t)
        coefs = np.zeros((n_curves, basis.n_basis))
        
        # Precompute penalty matrix R if lambda > 0
        if lmbda > 0:
            R = basis.penalty_matrix(order=penalty_order)
        else:
            R = np.zeros((basis.n_basis, basis.n_basis))
            
        # Smooth each curve individually
        for i in range(n_curves):
            t_i = np.asarray(t[i])
            y_i = np.asarray(y[i])
            
            Phi = basis.evaluate(t_i) # Shape (n_points_i, n_basis)
            
            # System matrix: Phi.T @ Phi + lambda * R
            A = Phi.T @ Phi + lmbda * R
            b = Phi.T @ y_i
            
            # Solve for coefficients
            coefs[i, :] = solve(A, b, assume_a='pos')
            
        return FunctionalData(basis, coefs)
        
    else:
        # Shared grid case
        t = np.asarray(t)
        y = np.asarray(y)
        
        if y.ndim == 1:
            y = y.reshape(1, -1)
            
        n_curves, n_points = y.shape
        if len(t) != n_points:
            raise ValueError(f"Length of t ({len(t)}) must match the number of columns in y ({n_points}).")
            
        Phi = basis.evaluate(t) # Shape (n_points, n_basis)
        
        # Precompute penalty matrix R if lambda > 0
        if lmbda > 0:
            R = basis.penalty_matrix(order=penalty_order)
        else:
            R = np.zeros((basis.n_basis, basis.n_basis))
            
        # System matrix for the coefficients
        A = Phi.T @ Phi + lmbda * R
        
        # solve solves AX = B where X and B are matrices.
        # B = Phi.T @ y.T has shape (n_basis, n_curves)
        # So X will have shape (n_basis, n_curves)
        b = Phi.T @ y.T
        coefs_T = solve(A, b, assume_a='pos')
        
        return FunctionalData(basis, coefs_T.T)
