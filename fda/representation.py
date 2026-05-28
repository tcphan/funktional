import numpy as np
from fda.basis import Basis

class FunctionalData:
    """Class representing a set of functional data curves.
    
    Each curve is represented as a linear combination of basis functions:
    x_i(t) = \sum_{k=1}^K c_{ik} \phi_k(t)
    """

    def __init__(self, basis: Basis, coef: np.ndarray):
        """Initialize the functional data object.
        
        Parameters
        ----------
        basis : Basis
            An instance of a class inheriting from Basis.
        coef : array-like
            Coefficients of shape (n_curves, n_basis).
        """

        self.basis = basis
        self.coef = np.asarray(coef, dtype=float)
        
        if self.coef.ndim == 1:
            # Reshape 1D array to a single curve
            self.coef = self.coef.reshape(1, -1)
            
        if self.coef.shape[1] != self.basis.n_basis:
            raise ValueError(
                f"Number of coefficients per curve ({self.coef.shape[1]}) "
                f"does not match the number of basis functions ({self.basis.n_basis})."
            )

    @property
    def n_curves(self):
        """Number of curves in the dataset."""
        return self.coef.shape[0]

    @property
    def n_basis(self):
        """Number of basis functions."""
        return self.basis.n_basis

    @property
    def domain_range(self):
        """Domain of the functional data."""
        return self.basis.domain_range

    def __call__(self, eval_points: np.ndarray):
        """Evaluate the functional data curves at given points.
        
        Parameters
        ----------
        eval_points : array-like
            Points at which to evaluate the curves.
            
        Returns
        -------
        numpy.ndarray
            Matrix of shape (n_curves, len(eval_points)) representing
            the evaluated curves.
        """

        # Phi shape: (len(eval_points), n_basis)
        Phi = self.basis.evaluate(eval_points)
        # Result shape: (n_curves, len(eval_points))
        return self.coef @ Phi.T

    def evaluate_derivative(self, eval_points: np.ndarray, order: int=1):
        """Evaluate the derivative of the functional data curves at given points.
        
        Parameters
        ----------
        eval_points : array-like
            Points at which to evaluate the derivative.
        order : int, default=1
            The order of the derivative.
            
        Returns
        -------
        numpy.ndarray
            Matrix of shape (n_curves, len(eval_points)) representing
            the evaluated derivatives.
        """

        Phi_deriv = self.basis.evaluate_derivative(eval_points, order=order)
        return self.coef @ Phi_deriv.T

    def mean(self):
        """Compute the functional mean curve.
        
        Returns
        -------
        FunctionalData
            A new FunctionalData object representing the mean curve.
        """

        mean_coef = np.mean(self.coef, axis=0, keepdims=True)
        return FunctionalData(self.basis, mean_coef)

    def __add__(self, other):
        """Add two functional data objects or a functional data object and a constant/numpy array."""

        if isinstance(other, FunctionalData):
            if self.basis != other.basis:
                # We check if basis type and params are same
                if type(self.basis) != type(other.basis) or self.basis.n_basis != other.basis.n_basis or self.basis.domain_range != other.basis.domain_range:
                    raise ValueError("Cannot add FunctionalData objects with different bases.")
            
            if self.n_curves != other.n_curves and self.n_curves != 1 and other.n_curves != 1:
                raise ValueError("Incompatible number of curves for addition.")
                
            return FunctionalData(self.basis, self.coef + other.coef)
        else:
            # Assume other is a numeric scalar or array compatible with broadcasting on coef
            return FunctionalData(self.basis, self.coef + other)

    def __sub__(self, other):
        """Subtract two functional data objects or a functional data object and a constant/numpy array."""

        if isinstance(other, FunctionalData):
            if self.basis != other.basis:
                if type(self.basis) != type(other.basis) or self.basis.n_basis != other.basis.n_basis or self.basis.domain_range != other.basis.domain_range:
                    raise ValueError("Cannot subtract FunctionalData objects with different bases.")
            
            if self.n_curves != other.n_curves and self.n_curves != 1 and other.n_curves != 1:
                raise ValueError("Incompatible number of curves for subtraction.")
                
            return FunctionalData(self.basis, self.coef - other.coef)
        else:
            return FunctionalData(self.basis, self.coef - other)

    def __mul__(self, other):
        """Multiply functional data by a scalar or broadcastable numpy array."""
        
        if isinstance(other, (int, float, np.ndarray)):
            # If multiplying by array, shape must match or broadcast
            return FunctionalData(self.basis, self.coef * other)
        raise TypeError("Multiplication is only supported by scalar or numpy array.")

    def __rmul__(self, other):
        return self.__mul__(other)
