import numpy as np
import matplotlib.pyplot as plt
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
        r"""Compute the penalty matrix of a given derivative order.
        
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
        self.cache = None
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

    def _b_spline_value(self, x: float, j: int, p: int):
        """Calculates the B-spline value at x for a single basis function j of degree p.
        
        Parameters
        ----------
        x : float
            The point at which to evaluate the basis function.
        j : int
            The index of the basis function.
        p: int
            The degree of the B-spline basis function.
            
        Returns
        -------
        float
            The B-spline value at x for a single basis function j of degree p.
        """

        # Initialize cache on the first call
        if self.cache is None:
            self.cache = {}

        # Create unique key to identify current recursive state
        key = (x, j, p)

        # Check if result is already in cache
        if key in self.cache:
            return self.cache[key]

        # Base case: degree 0 (step functions)
        if p == 0:
            result = 1.0 if self.knots[j] <= x < self.knots[j+1] else 0.0
            self.cache[key] = result
            return result

        # Recursive step
        denom1 = self.knots[j+p] - self.knots[j]
        denom2 = self.knots[j+p+1] - self.knots[j+1]

        val = 0.0
        if denom1 > 0:
            term1_val = self._b_spline_value(x, j, p-1)
            val += ((x - self.knots[j]) / denom1) * term1_val
            
        if denom2 > 0:
            term2_val = self._b_spline_value(x, j+1, p-1)
            val += ((self.knots[j+p+1] - x) / denom2) * term2_val
        
        # Store result in cache
        self.cache[key] = val
        return val

    def evaluate(self, eval_points: np.ndarray) -> np.ndarray:
        
        # Convert to numpy array
        eval_points = np.asarray(eval_points)

        # Initialize array to store basis values
        basis_vals = np.zeros((len(eval_points), self.n_basis))

        # Evaluate each basis function
        for j in range(self.n_basis):

            # Define a helper function that evaluates a single basis function
            def basis_step(x):
                return self._b_spline_value(x, j, self.degree)
            
            # Vectorize the helper function
            vectorized_func = np.vectorize(basis_step, otypes=[float])
            
            # Evaluate the vectorized function over the evaluation points
            basis_vals[:,j] = vectorized_func(eval_points)
            
        return basis_vals

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

    def plot_b_spline(self,  eval_points: np.ndarray):
        """Create a plot of the B-spline basis functions.
        
        Args:
            eval_points: Points at which to evaluate the basis functions
        """

        # Evaluate the basis functions
        basis_matrix = self.evaluate(eval_points)

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(12,5))

        # Plot each basis function
        for j in range(self.n_basis):
            ax.plot(
                eval_points,
                basis_matrix[:,j],
                linewidth=2,
                label=f"B-spline {j+1}",
                color=plt.cm.get_cmap("Set3", self.n_basis)(j)
            )
        
        # Apply formatting
        ax.set_title(
            f"B-Spline Basis (n_basis={self.n_basis}, degree={self.degree})", 
            fontsize=10, 
            fontweight="bold", 
            pad=10
        )
        ax.set_xlabel("$x$", fontsize=9)
        ax.set_ylabel(r"$\phi_j(x)$", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Display the plot
        plt.tight_layout()
        plt.show()
        
    def b_spline_math_notation(self, x, j):
        """Generates the math notation for a B-spline basis function evaluation.
        
        Parameters
        ----------
        x : float
            The point at which to evaluate the basis function.
        j : int
            The index of the basis function.
            
        Returns
        -------
        str
            The math notation for the B-spline basis function evaluation.
        """

        def make_subscript(number):
            """Convert a number to a subscript string."""
            subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
            return str(number).translate(subscript_map)

        # Convert to subscript
        j_subscr = make_subscript(j)
        p_subscr = make_subscript(self.degree)
        j_plus_p_subscr = make_subscript(j+self.degree)
        p_minus_1_subscr = make_subscript(self.degree-1)
        j_plus_p_plus_1_subscr = make_subscript(j+self.degree+1)
        j_plus_1_subscr = make_subscript(j+1)

        # Line 1 math notation
        part1 = f"({x} - t{j_subscr}/t{j_plus_p_subscr} - t{j_subscr}) * B{j_subscr},{p_minus_1_subscr}(x={x})"
        part2 = f"(t{j_plus_p_plus_1_subscr} - {x}/t{j_plus_p_plus_1_subscr} - t{j_plus_1_subscr}) * B{j_plus_1_subscr},{p_minus_1_subscr}(x={x})"
        math_notation_line1 = f"B{j_subscr},{p_subscr}(x=1.5) = {part1} + {part2}"

        # Line 2 math notation
        part1 = f"({x} - {self.knots[j]}/{self.knots[j+self.degree]} - {self.knots[j]}) * B{j_subscr},{p_minus_1_subscr}(x={x})"
        part2 = f"({self.knots[j+self.degree+1]} - {x}/{self.knots[j+self.degree+1]} - {self.knots[j+1]}) * B{j_plus_1_subscr},{p_minus_1_subscr}(x={x})"
        math_notation_line2 = f"B{j_subscr},{p_subscr}(x=1.5) = {part1} + {part2}"

        # Line 3 math notation
        weight1 = (x - self.knots[j])/(self.knots[j+self.degree] - self.knots[j])
        weight2 = (self.knots[j+self.degree+1] - x)/(self.knots[j+self.degree+1] - self.knots[j+1])
        part1 = f"{weight1:.4f} * B{j_subscr},{p_minus_1_subscr}(x={x})"
        part2 = f"{weight2:.4f} * B{j_plus_1_subscr},{p_minus_1_subscr}(x={x})"
        math_notation_line3 = f"B{j_subscr},{p_subscr}(x=1.5) = {part1} + {part2}"

        # Line 4 math notation
        b_val1 = self._b_spline_step(x, j)
        b_val2 = self._b_spline_step(x, j+1)
        part1 = f"{weight1:.4f} * {b_val1}"
        part2 = f"{weight2:.4f} * {b_val2}"
        math_notation_line4 = f"B{j_subscr},{p_subscr}(x=1.5) = {part1} + {part2}"

        # Line 5 math notation
        math_notation_line5 = f"B{j_subscr},{p_subscr}(x=1.5) = {(weight1 * b_val1) + (weight2 * b_val2):.4f}"

        # Combine all notation lines together
        final_math_notation = "\n".join([
            math_notation_line1,
            math_notation_line2,
            math_notation_line3,
            math_notation_line4,
            math_notation_line5
        ])
        return final_math_notation


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
