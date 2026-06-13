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
            raise ValueError(
                "domain_range must be a tuple of two floats (a, b) with a < b."
            )
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
    def evaluate_derivative(
        self, eval_points: np.ndarray, order: int = 1
    ) -> np.ndarray:
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

    def penalty_matrix(self, order: int = 2, n_points: int = 1000) -> np.ndarray:
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
        deriv = self.evaluate_derivative(t, order=order)  # Shape (n_points, n_basis)

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

    def __init__(
        self, domain_range: tuple[float, float], n_basis: int, degree: int = 3
    ):
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
            raise ValueError(
                f"n_basis ({n_basis}) must be greater than degree ({degree})."
            )

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
        self.knots = np.concatenate(
            [
                np.repeat(a, self.degree + 1),
                internal_knots,
                np.repeat(b, self.degree + 1),
            ]
        )

    def evaluate(self, eval_points: np.ndarray) -> np.ndarray:

        # Convert to numpy array
        eval_points = np.asarray(eval_points)
        n_pts = len(eval_points)

        # Total number of basis functions at degree 0 is len(knots) - 1
        n_knots = len(self.knots)

        # Initialize Level 0 Basis Matrix (Degree = 0)
        # B_shape will contract or adjust as degree increases, but we start with all structural intervals
        current_basis = np.zeros((n_pts, n_knots - 1))

        for i in range(n_knots - 1):
            # Normal half-open interval: [t_i, t_{i+1})
            is_in_interval = (eval_points >= self.knots[i]) & (
                eval_points < self.knots[i + 1]
            )

            # Include the right-most boundary point (x = b) in the last valid interval
            if i == self.n_basis - 1:
                is_in_interval |= eval_points == self.knots[i + 1]

            current_basis[:, i] = is_in_interval.astype(float)

        # Iteratively compute higher degrees up to self.degree
        for p in range(1, self.degree + 1):
            next_basis = np.zeros((n_pts, n_knots - 1 - p))

            for i in range(n_knots - 1 - p):
                # Left Term Calculations
                denom1 = self.knots[i + p] - self.knots[i]
                if denom1 > 0:
                    left_factor = (eval_points - self.knots[i]) / denom1
                    term1 = left_factor * current_basis[:, i]
                else:
                    term1 = 0.0

                # Right Term Calculations
                denom2 = self.knots[i + p + 1] - self.knots[i + 1]
                if denom2 > 0:
                    right_factor = (self.knots[i + p + 1] - eval_points) / denom2
                    term2 = right_factor * current_basis[:, i + 1]
                else:
                    term2 = 0.0

                next_basis[:, i] = term1 + term2

            current_basis = next_basis

        # Return only the first n_basis columns to ensure correct output dimension
        return current_basis[:, : self.n_basis]

    def evaluate_derivative(
        self, eval_points: np.ndarray, order: int = 1
    ) -> np.ndarray:
        r"""
        Evaluates the derivative of all B-spline basis functions at points x for a given degree order.

        The derivative for the i-th B-spline of degree p is given by:
        $$\frac{d}{dx}N_{i,p}(x) = p \cdot \left( \frac{N_{i,p-1}(x)}{t_{i+p} - t_i} - \frac{N_{i+1,p-1}(x)}{t_{i+p+1} - t_{i+1}} \right)$$

        Parameters:
        -----------
        x : np.ndarray
            1D array of points where the derivatives are evaluated.
        order : int
            The order of the derivative to evaluate.

        Returns:
        --------
        np.ndarray
            A matrix of shape (len(x), n_basis) containing the derivatives.
        """

        if order < 0:
            raise ValueError("Derivative order must be non-negative.")

        # Set up variables
        x = np.asarray(eval_points)
        n_samples = len(x)

        # Base Case 1: Order 0 is just evaluating the basis functions themselves
        if order == 0:
            return self.evaluate(x)

        # Base Case 2: Order exceeds degree, derivative is 0 everywhere
        if order > self.degree:
            return np.zeros((n_samples, self.n_basis))

        # Compute higher-order derivatives recursively
        t = self.knots

        def _evaluate_raw_basis(x, degree, n_basis_funcs, knots):
            """Helper method to evaluate arbitrary degree splines on a raw knot vector."""

            n_pts = len(x)
            n_knots = len(knots)

            current_basis = np.zeros((n_pts, n_knots - 1))
            for i in range(n_knots - 1):
                is_in_interval = (x >= knots[i]) & (x < knots[i + 1])
                if i == n_knots - 2:
                    is_in_interval |= x == knots[i + 1]
                current_basis[:, i] = is_in_interval.astype(float)

            for p in range(1, degree + 1):
                next_basis = np.zeros((n_pts, n_knots - 1 - p))
                for i in range(n_knots - 1 - p):
                    denom1 = knots[i + p] - knots[i]
                    term1 = (
                        ((x - knots[i]) / denom1) * current_basis[:, i]
                        if denom1 > 0
                        else 0.0
                    )

                    denom2 = knots[i + p + 1] - knots[i + 1]
                    term2 = (
                        ((knots[i + p + 1] - x) / denom2) * current_basis[:, i + 1]
                        if denom2 > 0
                        else 0.0
                    )

                    next_basis[:, i] = term1 + term2
                current_basis = next_basis

            return current_basis[:, :n_basis_funcs]

        def _compute_deriv(p_current, current_n_basis):
            """Helper that recursively matches the derivative definitionon the underlying structural knot indices."""

            # Base case: evaluate the basis functions at the correct lower degree
            if p_current == self.degree - order:
                # Evaluate the lower degree basis on our EXACT same knot vector.
                return _evaluate_raw_basis(x, p_current, current_n_basis, t)

            # Initialize array for this recursive layer
            deriv = np.zeros((n_samples, current_n_basis))

            # Evaluate the next lower degree layer required for this step
            lower_deriv = _compute_deriv(p_current - 1, current_n_basis + 1)

            for i in range(current_n_basis):
                # Term 1 denominator
                denom1 = t[i + p_current] - t[i]
                term1 = (lower_deriv[:, i] / denom1) if denom1 > 0 else 0.0

                # Term 2 denominator
                denom2 = t[i + p_current + 1] - t[i + 1]
                term2 = (lower_deriv[:, i + 1] / denom2) if denom2 > 0 else 0.0

                deriv[:, i] = p_current * (term1 - term2)

            return deriv

        return _compute_deriv(self.degree, self.n_basis)

    def plot_b_spline(self, eval_points: np.ndarray):
        """Create a plot of the B-spline basis functions.

        Args:
            eval_points: Points at which to evaluate the basis functions
        """

        # Evaluate the basis functions
        basis_matrix = self.evaluate(eval_points)

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot each basis function
        for j in range(self.n_basis):
            ax.plot(
                eval_points,
                basis_matrix[:, j],
                linewidth=2,
                label=f"B-spline {j + 1}",
                color=plt.cm.get_cmap("Set3", self.n_basis)(j),
            )

        # Apply formatting
        ax.set_title(
            f"B-Spline Basis (n_basis={self.n_basis}, degree={self.degree})",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("$x$", fontsize=9)
        ax.set_ylabel(r"$\phi_j(x)$", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
        ax.grid(True, linestyle="--", alpha=0.7)

        # Display the plot
        plt.tight_layout()
        plt.show()


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
            raise ValueError(
                "n_basis for Fourier basis must be an odd positive integer."
            )

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

    def evaluate_derivative(
        self, eval_points: np.ndarray, order: int = 1
    ) -> np.ndarray:
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
