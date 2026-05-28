import numpy as np
from fda.representation import FunctionalData

def functional_mean(fd: FunctionalData):
    """Compute the functional mean curve of a FunctionalData object.
    
    Parameters
    ----------
    fd : FunctionalData
        The functional data object.
        
    Returns
    -------
    FunctionalData
        A new FunctionalData object representing the mean curve.
    """
    return fd.mean()

def functional_variance(fd: FunctionalData, eval_points: np.ndarray):
    """Compute the functional variance curve evaluated at given points.
    
    The variance curve is defined as:
    Var(x(t)) = 1/(N-1) \sum_{i=1}^N (x_i(t) - \bar{x}(t))^2
    
    Parameters
    ----------
    fd : FunctionalData
        The functional data object.
    eval_points : array-like
        Points at which to evaluate the variance curve.
        
    Returns
    -------
    numpy.ndarray
        1D array of shape (len(eval_points),) representing the variance values.
    """
    
    eval_points = np.asarray(eval_points)
    if fd.n_curves <= 1:
        return np.zeros(len(eval_points))
        
    # Center the coefficients
    coef_centered = fd.coef - np.mean(fd.coef, axis=0)
    
    # Compute the covariance matrix of coefficients
    # Sigma_c shape: (n_basis, n_basis)
    Sigma_c = (coef_centered.T @ coef_centered) / (fd.n_curves - 1)
    
    # Evaluate basis at evaluation points
    # Phi shape: (n_points, n_basis)
    Phi = fd.basis.evaluate(eval_points)
    
    # Compute the diagonal of Phi @ Sigma_c @ Phi.T efficiently:
    # Var(x(t)) = \sum_{j,k} \Phi_{ij} \Sigma_c_{jk} \Phi_{ik}
    # This is equivalent to row-wise sum of (Phi @ Sigma_c) * Phi
    var_vals = np.sum((Phi @ Sigma_c) * Phi, axis=1)
    return var_vals

def functional_covariance(fd1: FunctionalData, fd2: FunctionalData=None, eval_points1: np.ndarray=None, eval_points2: np.ndarray=None):
    """Compute the covariance surface between two sets of functional curves.
    
    The covariance surface is defined as:
    Cov(x(s), y(t)) = 1/(N-1) \sum_{i=1}^N (x_i(s) - \bar{x}(s))(y_i(t) - \bar{y}(t))
    
    Parameters
    ----------
    fd1 : FunctionalData
        First functional data object.
    fd2 : FunctionalData, optional
        Second functional data object. If None, computes the auto-covariance of fd1.
    eval_points1 : array-like, optional
        Points at which to evaluate the first domain coordinate (s).
        If None, uses 100 equidistant points over the domain of fd1.
    eval_points2 : array-like, optional
        Points at which to evaluate the second domain coordinate (t).
        If None, uses 100 equidistant points over the domain of fd2.
        
    Returns
    -------
    numpy.ndarray
        2D array of shape (len(eval_points1), len(eval_points2)) representing the covariance surface.
    """
    
    if fd2 is None:
        fd2 = fd1
        
    if fd1.n_curves != fd2.n_curves:
        raise ValueError("Both functional datasets must have the same number of curves.")
        
    if fd1.n_curves <= 1:
        raise ValueError("Covariance requires at least 2 curves.")
        
    if eval_points1 is None:
        a, b = fd1.domain_range
        eval_points1 = np.linspace(a, b, 100)
    else:
        eval_points1 = np.asarray(eval_points1)
        
    if eval_points2 is None:
        a, b = fd2.domain_range
        eval_points2 = np.linspace(a, b, 100)
    else:
        eval_points2 = np.asarray(eval_points2)
        
    # Center the coefficients
    coef1_centered = fd1.coef - np.mean(fd1.coef, axis=0)
    coef2_centered = fd2.coef - np.mean(fd2.coef, axis=0)
    
    # Compute cross-covariance of coefficients
    # Sigma_c shape: (n_basis1, n_basis2)
    Sigma_c = (coef1_centered.T @ coef2_centered) / (fd1.n_curves - 1)
    
    # Evaluate bases
    Phi1 = fd1.basis.evaluate(eval_points1) # (len(eval_points1), n_basis1)
    Phi2 = fd2.basis.evaluate(eval_points2) # (len(eval_points2), n_basis2)
    
    # Covariance surface: V = Phi1 @ Sigma_c @ Phi2.T
    # Shape: (len(eval_points1), len(eval_points2))
    cov_surface = Phi1 @ Sigma_c @ Phi2.T
    return cov_surface
