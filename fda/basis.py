import numpy as np
from scipy.interpolate import BSpline
from abc import ABC, abstractmethod

class Basis(ABC):
    """Abstract base class for functional data basis functions.
    
    All basis implementations must define the domain range, the number of basis functions,
    and how to evaluate the basis functions and their derivatives.
    """

    def __init__(self, domain_range: tuple[float, float], n_basis: int):
        """Initialize the basis.
        
        Parameters
        ----------
        domain_range : tuple of float
            A tuple (a, b) representing the start and end of the domain.
        n_basis : int
            The number of basis functions.
        """

        if len(domain_range) != 2 or domain_range[0] >= domain_range[1]:
            raise ValueError("domain_range must be a tuple of two floats (a, b) with a < b.")
        if n_basis <= 0:
            raise ValueError("n_basis must be a positive integer.")
            
        self.domain_range = (float(domain_range[0]), float(domain_range[1]))
        self.n_basis = int(n_basis)

    @abstractmethod
    def evaluate(self, eval_points: np.ndarray) -> np.ndarray:
        """Evaluate the basis functions at the given points.
        
        Parameters
        ----------
        eval_points : np.ndarray
            Points at which to evaluate the basis functions.
            
        Returns
        -------
        np.ndarray
            Matrix of shape (len(eval_points), n_basis) where the (i, j)-th entry
            is the j-th basis function evaluated at the i-th point.
        """

        pass

    @abstractmethod
    def evaluate_derivative(self, eval_points: np.ndarray, order: int=1) -> np.ndarray:
        """Evaluate the derivative of the basis functions at the given points.
        
        Parameters
        ----------
        eval_points : np.ndarray
            Points at which to evaluate the derivative.
        order : int, default=1
            The order of the derivative to evaluate.
            
        Returns
        -------
        np.ndarray
            Matrix of shape (len(eval_points), n_basis) representing the derivative values.
        """

        pass

    def penalty_matrix(self, order: int=2, n_points: int=1000) -> np.ndarray:
        """Compute the penalty matrix of a given derivative order.
        
        The penalty matrix is defined as:
        P_{jk} = \int_a^b \phi_j^{(order)}(t) \phi_k^{(order)}(t) dt
        
        Parameters
        ----------
        order : int, default=2
            The order of the derivative to penalize.
            - order=0: Basis inner product matrix (overlap matrix).
            - order=2: Roughness penalty matrix (curvature penalty).
        n_points : int, default=1000
            Number of points used for trapezoidal integration.
            
        Returns
        -------
        np.ndarray
            Symmetric matrix of shape (n_basis, n_basis).
        """

        t = np.linspace(self.domain_range[0], self.domain_range[1], n_points)
        dt = (self.domain_range[1] - self.domain_range[0]) / (n_points - 1)
        
        # Evaluate derivative on the grid
        deriv = self.evaluate_derivative(t, order=order) # Shape (n_points, n_basis)
        
        # Trapezoidal weights
        weights = np.ones(n_points) * dt
        weights[0] /= 2.0
        weights[-1] /= 2.0
        
        # Compute the weighted outer product sum: deriv.T @ diag(weights) @ deriv
        # This is equivalent to (deriv.T * weights) @ deriv
        penalty = (deriv.T * weights) @ deriv
        return penalty



class BSplineBasis(Basis):
    """B-spline basis functions over a domain."""

    def __init__(self, domain_range: tuple[float, float], n_basis: int, degree: int=3):
        """Initialize B-spline basis.
        
        Parameters
        ----------
        domain_range : tuple[float, float]
            Domain (a, b).
        n_basis : int
            Number of basis functions. Must be greater than `degree`.
        degree : int, default=3
            Degree of the B-splines (e.g., 3 for cubic splines).
        """
        super().__init__(domain_range, n_basis)
        if degree < 0:
            raise ValueError("degree must be a non-negative integer.")
        if n_basis <= degree:
            raise ValueError(f"n_basis ({n_basis}) must be greater than degree ({degree}).")
            
        self.degree = int(degree)
        self._setup_knots()

    def _setup_knots(self):
        """Setup the knot vector for B-splines."""

        a, b = self.domain_range
        # Number of internal knots
        n_internal = self.n_basis - self.degree - 1
        
        # Equidistant internal knots
        if n_internal > 0:
            internal_knots = np.linspace(a, b, n_internal + 2)[1:-1]
        else:
            internal_knots = np.array([])
            
        # Coincident boundary knots
        self.knots = np.concatenate([
            np.repeat(a, self.degree + 1),
            internal_knots,
            np.repeat(b, self.degree + 1)
        ])

    def evaluate(self, eval_points: np.ndarray) -> np.ndarray:
        eval_points = np.asarray(eval_points)
        
        # We can evaluate all basis functions at once by passing an identity matrix
        # as coefficients to the BSpline class.
        c = np.eye(self.n_basis)
        spl = BSpline(self.knots, c, self.degree, extrapolate=False)
        
        # SciPy's BSpline evaluates to NaN outside the knot interval.
        # We will handle boundary issues by clamping slightly or raising a warning,
        # but standard FDA assumes eval_points are within the domain.
        res = spl(eval_points)
        
        # Replace NaNs with 0 if points are close to boundaries but slightly outside due to numerical precision
        res = np.nan_to_num(res, nan=0.0)
        return res

    def evaluate_derivative(self, eval_points: np.ndarray, order: int=1) -> np.ndarray:
        eval_points = np.asarray(eval_points)
        if order < 0:
            raise ValueError("Derivative order must be non-negative.")
        if order == 0:
            return self.evaluate(eval_points)
            
        c = np.eye(self.n_basis)
        spl = BSpline(self.knots, c, self.degree, extrapolate=False)
        
        try:
            spl_deriv = spl.derivative(order)
            res = spl_deriv(eval_points)
            res = np.nan_to_num(res, nan=0.0)
            return res
        except ValueError as e:
            # If order is greater than the degree, SciPy's derivative might fail
            # or return zero.
            if order > self.degree:
                return np.zeros((len(eval_points), self.n_basis))
            raise e


class FourierBasis(Basis):
    """Fourier basis functions (sine and cosine) for periodic functional data."""
    
    def __init__(self, domain_range: tuple[float, float], n_basis: int):
        """Initialize Fourier basis.
        
        Parameters
        ----------
        domain_range : tuple of float
            Domain (a, b).
        n_basis : int
            Number of basis functions. Must be an odd positive integer
            to have a symmetric set of sine and cosine terms (1 constant + 2*M terms).
        """
        super().__init__(domain_range, n_basis)
        if n_basis % 2 == 0:
            raise ValueError("n_basis for Fourier basis must be an odd positive integer.")

    def evaluate(self, eval_points: np.ndarray) -> np.ndarray:
        eval_points = np.asarray(eval_points)
        n_pts = len(eval_points)
        res = np.zeros((n_pts, self.n_basis))
        
        a, b = self.domain_range
        width = b - a
        
        # First basis function is constant (normalized)
        res[:, 0] = 1.0 / np.sqrt(width)
        
        # Remaining basis functions are alternating sine and cosine terms
        # for j = 1, ..., M
        M = (self.n_basis - 1) // 2
        for j in range(1, M + 1):
            omega = 2 * np.pi * j / width
            res[:, 2 * j - 1] = np.sqrt(2.0 / width) * np.sin(omega * (eval_points - a))
            res[:, 2 * j] = np.sqrt(2.0 / width) * np.cos(omega * (eval_points - a))
            
        return res

    def evaluate_derivative(self, eval_points: np.ndarray, order: int=1) -> np.ndarray:
        eval_points = np.asarray(eval_points)
        if order < 0:
            raise ValueError("Derivative order must be non-negative.")
        if order == 0:
            return self.evaluate(eval_points)
            
        n_pts = len(eval_points)
        res = np.zeros((n_pts, self.n_basis))
        
        a, b = self.domain_range
        width = b - a
        
        # Constant basis function derivative is 0
        res[:, 0] = 0.0
        
        M = (self.n_basis - 1) // 2
        for j in range(1, M + 1):
            omega = 2 * np.pi * j / width
            # For derivative of sin(w*(t-a)) and cos(w*(t-a)):
            # We can use standard trigonometric derivative rules, or general form:
            # d^k/dt^k sin(w*t) = w^k * sin(w*t + k * pi / 2)
            # d^k/dt^k cos(w*t) = w^k * cos(w*t + k * pi / 2)
            phase = order * np.pi / 2
            scale = (omega) ** order
            
            val_sin = np.sqrt(2.0 / width) * np.sin(omega * (eval_points - a) + phase)
            val_cos = np.sqrt(2.0 / width) * np.cos(omega * (eval_points - a) + phase)
            
            res[:, 2 * j - 1] = scale * val_sin
            res[:, 2 * j] = scale * val_cos
            
        return res
