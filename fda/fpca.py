import numpy as np
from scipy.linalg import cholesky, solve_triangular, eigh
from fda.representation import FunctionalData
from fda.stats import functional_mean

class FPCA:
    """Functional Principal Component Analysis (FPCA).
    
    Fits functional principal components (eigenfunctions) to functional data,
    allowing dimensionality reduction and extraction of key modes of variation.
    
    The FPCA API mirrors scikit-learn's standard estimators.
    """

    def __init__(self, n_components:int=3):
        """Initialize the FPCA object.
        
        Parameters
        ----------
        n_components : int, default=3
            The number of principal components to retain.
        """

        if n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = int(n_components)
        
        # Fitted attributes
        self.mean_ = None
        self.components_ = None
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None

    def fit(self, fd: FunctionalData):
        """Fit the FPCA model on the functional data.
        
        Parameters
        ----------
        fd : FunctionalData
            The functional data to fit.
            
        Returns
        -------
        self : FPCA
            The fitted FPCA estimator.
        """

        n_curves = fd.n_curves
        if n_curves < 2:
            raise ValueError("FPCA requires at least 2 curves to compute principal components.")
            
        # Limit components to minimum of n_components, n_curves, and n_basis
        n_comp = min(self.n_components, n_curves - 1, fd.n_basis)
        
        # 1. Compute basis inner product (overlap) matrix W
        W = fd.basis.penalty_matrix(order=0)
        
        # Ensure W is symmetric and positive definite (numerical stability)
        W = (W + W.T) / 2.0
        
        # Cholesky decomposition of W: W = L @ L.T
        L = cholesky(W, lower=True)
        
        # 2. Compute the sample covariance matrix of centered coefficients
        self.mean_ = functional_mean(fd)
        coef_centered = fd.coef - self.mean_.coef
        Sigma_c = (coef_centered.T @ coef_centered) / (n_curves - 1)
        
        # 3. Form the symmetric matrix H = L.T @ Sigma_c @ L
        H = L.T @ Sigma_c @ L
        H = (H + H.T) / 2.0  # Force symmetry
        
        # 4. Solve the standard eigenvalue problem H v = lambda v
        # eigh returns eigenvalues in ascending order
        evals, V = eigh(H)
        
        # Sort in descending order
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        V = V[:, idx]
        
        # Select the top n_comp components
        self.eigenvalues_ = evals[:n_comp]
        V_subset = V[:, :n_comp]
        
        # 5. Transform eigenvectors back to basis coefficients: u = L^{-T} v
        # This solves L.T @ U = V_subset
        U = solve_triangular(L.T, V_subset, lower=False)
        
        # The eigenfunctions are represented as FunctionalData
        # Coefficients shape: (n_components, n_basis)
        self.components_ = FunctionalData(fd.basis, U.T)
        
        # 6. Compute explained variance ratios
        # Total variance in the basis representation is the trace of Sigma_c @ W
        # mathematically: trace(H) = trace(L.T @ Sigma_c @ L) = trace(Sigma_c @ W)
        total_var = np.trace(H)
        if total_var > 0:
            self.explained_variance_ratio_ = self.eigenvalues_ / total_var
        else:
            self.explained_variance_ratio_ = np.zeros(n_comp)
            
        return self

    def transform(self, fd: FunctionalData):
        """Project functional data onto the eigenfunctions to obtain principal component scores.
        
        Parameters
        ----------
        fd : FunctionalData
            Functional data to project.
            
        Returns
        -------
        numpy.ndarray
            Scores matrix of shape (n_curves, n_components).
        """

        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("FPCA model must be fitted before calling transform.")
            
        if fd.basis.n_basis != self.components_.basis.n_basis:
            raise ValueError("Input functional data must have the same basis system as the fitted model.")
            
        # Center the coefficients using the fitted mean
        coef_centered = fd.coef - self.mean_.coef
        
        # W = basis overlap matrix
        W = self.components_.basis.penalty_matrix(order=0)
        
        # Eigenfunction coefficients (shape: n_basis, n_components)
        U = self.components_.coef.T
        
        # Scores: F = C_centered @ W @ U
        scores = coef_centered @ W @ U
        return scores

    def fit_transform(self, fd: FunctionalData):
        """Fit the FPCA model and return the principal component scores.
        
        Parameters
        ----------
        fd : FunctionalData
            Functional data to fit and project.
            
        Returns
        -------
        numpy.ndarray
            Scores matrix of shape (n_curves, n_components).
        """

        return self.fit(fd).transform(fd)

    def inverse_transform(self, scores: np.ndarray):
        """Reconstruct functional curves from principal component scores.
        
        Reconstructed curve:
        x_reconstructed(t) = mean(t) + \sum_{k=1}^K score_k * component_k(t)
        
        Parameters
        ----------
        scores : array-like of shape (n_curves, n_components)
            Scores obtained from transform.
            
        Returns
        -------
        FunctionalData
            Reconstructed FunctionalData object.
        """

        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("FPCA model must be fitted before calling inverse_transform.")
            
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
            
        n_curves, n_comp = scores.shape
        if n_comp != self.components_.n_curves:
            raise ValueError(f"scores must have {self.components_.n_curves} columns.")
            
        # Reconstruction: coef = mean_coef + scores @ component_coef
        recon_coef = self.mean_.coef + scores @ self.components_.coef
        return FunctionalData(self.components_.basis, recon_coef)
