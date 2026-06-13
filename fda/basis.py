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
        self,
        domain_range: tuple[float, float],
        n_basis: int,
        degree: int = 3,
        weight_type: str = "linear",
        p_func=None,
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
        weight_type : str, default="linear"
            Type of weights to use for the B-spline basis functions.
            - "linear": Linear weights.
            - "trigonometric": Trigonometric weights.
            - "hyperbolic": Hyperbolic weights
            - "variable_degree": Variable degree or fractional weighting where the degree p is a function of x.
        p_func : callable, default=None
            Function that computes the degree as a function of x. This is only used when weight_type is "variable_degree".
        """

        super().__init__(domain_range, n_basis)
        self.degree = int(degree)
        self._setup_knots()
        self.weight_type = weight_type.lower()
        self.p_func = p_func

        # Scale frequency so the total width fits nicely within pi radians
        width = self.domain_range[1] - self.domain_range[0]
        self.omega = np.pi / width if width > 0 else 1.0

        if degree < 0:
            raise ValueError("degree must be a non-negative integer.")

        if n_basis <= degree:
            raise ValueError(
                f"n_basis ({n_basis}) must be greater than degree ({degree})."
            )

        if self.weight_type not in [
            "linear",
            "trigonometric",
            "hyperbolic",
            "variable_degree",
        ]:
            raise ValueError(
                f"weight_type ({self.weight_type}) must be one of 'linear', 'trigonometric', 'hyperbolic', or 'variable_degree'."
            )

        if self.weight_type == "variable_degree" and self.p_func is None:
            raise ValueError(
                "p_func must be provided when weight_type is 'variable_degree'."
            )

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

    def _evaluate_raw_basis(self, x, degree, n_basis_funcs, knots):
        """
        Helper method to evaluate arbitrary degree splines on a raw knot vector,
        supporting linear, trigonometric, and hyperbolic weighting structures.
        """

        n_pts = len(x)
        n_knots = len(knots)
        w = self.omega  # Local reference to the domain frequency scale factor

        # --- Layer 0: Characteristic Step Functions ---
        current_basis = np.zeros((n_pts, n_knots - 1))
        for i in range(n_knots - 1):
            is_in_interval = (x >= knots[i]) & (x < knots[i + 1])
            if i == n_knots - 2:
                is_in_interval |= x == knots[i + 1]
            current_basis[:, i] = is_in_interval.astype(float)

        # --- Layers 1 to Degree: Pyramid Recurrence Iteration ---
        for p in range(1, degree + 1):
            next_basis = np.zeros((n_pts, n_knots - 1 - p))
            for i in range(n_knots - 1 - p):
                # --- Branch A: Standard Linear Weights ---
                if self.weight_type == "linear":
                    denom1 = knots[i + p] - knots[i]
                    left_factor = (x - knots[i]) / denom1 if denom1 > 0 else 0.0

                    denom2 = knots[i + p + 1] - knots[i + 1]
                    right_factor = (
                        (knots[i + p + 1] - x) / denom2 if denom2 > 0 else 0.0
                    )

                # --- Branch B: Trigonometric (GB-Spline) Weights ---
                elif self.weight_type == "trigonometric":
                    denom1 = np.sin(w * (knots[i + p] - knots[i]) / 2.0)
                    left_factor = (
                        np.sin(w * (x - knots[i]) / 2.0) / denom1 if denom1 > 0 else 0.0
                    )

                    denom2 = np.sin(w * (knots[i + p + 1] - knots[i + 1]) / 2.0)
                    right_factor = (
                        np.sin(w * (knots[i + p + 1] - x) / 2.0) / denom2
                        if denom2 > 0
                        else 0.0
                    )

                # --- Branch C: Hyperbolic Weights ---
                elif self.weight_type == "hyperbolic":
                    denom1 = (
                        np.path.sinh(w * (knots[i + p] - knots[i]) / 2.0)
                        if hasattr(np.path, "sinh")
                        else np.sinh(w * (knots[i + p] - knots[i]) / 2.0)
                    )
                    left_factor = (
                        np.sinh(w * (x - knots[i]) / 2.0) / denom1
                        if denom1 > 0
                        else 0.0
                    )

                    denom2 = np.sinh(w * (knots[i + p + 1] - knots[i + 1]) / 2.0)
                    right_factor = (
                        np.sinh(w * (knots[i + p + 1] - x) / 2.0) / denom2
                        if denom2 > 0
                        else 0.0
                    )
                # --- Fallback safety branch ---
                else:
                    denom1 = knots[i + p] - knots[i]
                    left_factor = (x - knots[i]) / denom1 if denom1 > 0 else 0.0
                    denom2 = knots[i + p + 1] - knots[i + 1]
                    right_factor = (
                        (knots[i + p + 1] - x) / denom2 if denom2 > 0 else 0.0
                    )

                # Blend the adjacent lower degree basis states
                term1 = left_factor * current_basis[:, i]
                term2 = right_factor * current_basis[:, i + 1]
                next_basis[:, i] = term1 + term2

            current_basis = next_basis

        return current_basis[:, :n_basis_funcs]

    def _linear_weights(self, eval_points: np.ndarray, i: int, p: int):
        """Calculate the linear weights for the B-spline basis functions.

        Parameters
        ----------
        eval_points : np.ndarray
            Points at which to evaluate the basis functions.
        i : int
            The index of the basis function.
        p : int
            The degree of the basis function.

        Returns
        -------
        np.ndarray
            The linear weights for the basis function.
        """

        # Component 1: Left Factor Blending
        denom1 = self.knots[i + p] - self.knots[i]
        left_factor = (eval_points - self.knots[i]) / denom1 if denom1 > 0 else 0.0

        # Component 2: Right Factor Blending
        denom2 = self.knots[i + p + 1] - self.knots[i + 1]
        right_factor = (
            (self.knots[i + p + 1] - eval_points) / denom2 if denom2 > 0 else 0.0
        )

        return left_factor, right_factor

    def _trigonometric_weights(self, eval_points: np.ndarray, i: int, p: int):
        """Calculate the trigonometric weights for the B-spline basis functions.

        Parameters
        ----------
        eval_points : np.ndarray
            Points at which to evaluate the basis functions.
        i : int
            The index of the basis function.
        p : int
            The degree of the basis function.

        Returns
        -------
        np.ndarray
            The trigonometric weights for the basis function.
        """

        t = self.knots
        w = self.omega

        # Component 1: Left Factor Blending
        denom1 = np.sin(w * (t[i + p] - t[i]) / 2.0)
        left_factor = (
            np.sin(w * (eval_points - t[i]) / 2.0) / denom1 if denom1 > 0 else 0.0
        )

        # Component 2: Right Factor Blending
        denom2 = np.sin(w * (t[i + p + 1] - t[i + 1]) / 2.0)
        right_factor = (
            np.sin(w * (t[i + p + 1] - eval_points) / 2.0) / denom2
            if denom2 > 0
            else 0.0
        )
        return left_factor, right_factor

    def _hyperbolic_weights(self, eval_points: np.ndarray, i: int, p: int):
        """Calculate the hyperbolic weights for the B-spline basis functions.

        Parameters
        ----------
        eval_points : np.ndarray
            Points at which to evaluate the basis functions.
        i : int
            The index of the basis function.
        p : int
            The degree of the basis function.

        Returns
        -------
        np.ndarray
            The hyperbolic weights for the basis function.
        """

        t = self.knots
        w = self.omega

        # Component 1: Left Factor Blending
        denom1 = np.sinh(w * (t[i + p] - t[i]) / 2.0)
        left_factor = (
            np.sinh(w * (eval_points - t[i]) / 2.0) / denom1 if denom1 > 0 else 0.0
        )

        # Component 2: Right Factor Blending
        denom2 = np.sinh(w * (t[i + p + 1] - t[i + 1]) / 2.0)
        right_factor = (
            np.sinh(w * (t[i + p + 1] - eval_points) / 2.0) / denom2
            if denom2 > 0
            else 0.0
        )
        return left_factor, right_factor

    def evaluate(self, eval_points: np.ndarray) -> np.ndarray:

        # Convert to numpy array
        eval_points = np.asarray(eval_points)
        n_pts = len(eval_points)

        # --- Handle Variable Degree Splines via Continuous Blending ---

        if self.weight_type == "variable_degree":
            p_values = self.p_func(eval_points)
            p_floor = np.floor(p_values).astype(int)
            p_ceil = np.ceil(p_values).astype(int)
            alpha = p_values - p_floor

            # Use the robust recursive raw basis evaluator built for derivatives
            basis_cache = {}
            for deg in np.unique(np.concatenate([p_floor, p_ceil])):
                if deg <= self.degree:
                    basis_cache[deg] = self._evaluate_raw_basis(
                        eval_points, deg, self.n_basis, self.knots
                    )
                else:
                    raise ValueError(
                        f"Requested variable degree {deg} exceeds maximum configured degree {self.degree}"
                    )

            # Linearly blend spatial states
            res = np.zeros((n_pts, self.n_basis))
            for idx in range(n_pts):
                f_deg = p_floor[idx]
                c_deg = p_ceil[idx]
                a = alpha[idx]
                res[idx, :] = (1.0 - a) * basis_cache[f_deg][idx, :] + a * basis_cache[
                    c_deg
                ][idx, :]
            return res

        # --- Standard Recurrence Pipeline (Linear, Trigonometric, Hyperbolic) ---

        n_knots = len(self.knots)
        current_basis = np.zeros((n_pts, n_knots - 1))

        for i in range(n_knots - 1):
            is_in_interval = (eval_points >= self.knots[i]) & (
                eval_points < self.knots[i + 1]
            )
            if i == n_knots - 2:
                is_in_interval |= eval_points == self.knots[i + 1]
            current_basis[:, i] = is_in_interval.astype(float)

        for p in range(1, self.degree + 1):
            next_basis = np.zeros((n_pts, n_knots - 1 - p))

            for i in range(n_knots - 1 - p):
                if self.weight_type == "linear":
                    left_factor, right_factor = self._linear_weights(eval_points, i, p)
                elif self.weight_type == "trigonometric":
                    left_factor, right_factor = self._trigonometric_weights(
                        eval_points, i, p
                    )
                elif self.weight_type == "hyperbolic":
                    left_factor, right_factor = self._hyperbolic_weights(
                        eval_points, i, p
                    )

                term1 = left_factor * current_basis[:, i]
                term2 = right_factor * current_basis[:, i + 1]
                next_basis[:, i] = term1 + term2

            current_basis = next_basis

        return current_basis[:, : self.n_basis]

    def evaluate_derivative(
        self, eval_points: np.ndarray, order: int = 1
    ) -> np.ndarray:
        """
        Evaluates the derivative of all B-spline basis functions at points x for a given order.
        """

        if order < 0:
            raise ValueError("Derivative order must be non-negative.")

        x = np.asarray(eval_points)
        n_samples = len(x)

        # Base Case: Order 0 is just standard evaluation
        if order == 0:
            return self.evaluate(x)

        # --- Path A: High-Precision Central Difference for Non-Linear / Variable Weights ---
        if self.weight_type in ["trigonometric", "hyperbolic", "variable_degree"]:
            # Small step size optimized for 64-bit floating point precision
            h = 1e-5

            if order == 1:
                # 4th-order central difference schema for smooth slope evaluation
                f_inf2 = self.evaluate(x - 2 * h)
                f_inf1 = self.evaluate(x - h)
                f_sup1 = self.evaluate(x + h)
                f_sup2 = self.evaluate(x + 2 * h)
                return (-f_sup2 + 8 * f_sup1 - 8 * f_inf1 + f_inf2) / (12 * h)

            elif order == 2:
                # 4th-order central difference schema for curvature evaluation
                f_inf2 = self.evaluate(x - 2 * h)
                f_inf1 = self.evaluate(x - h)
                f_mid = self.evaluate(x)
                f_sup1 = self.evaluate(x + h)
                f_sup2 = self.evaluate(x + 2 * h)
                return (-f_sup2 + 16 * f_sup1 - 30 * f_mid + 16 * f_inf1 - f_inf2) / (
                    12 * (h**2)
                )

            else:
                # Recursive fallback for higher orders if needed
                deriv_minus = self.evaluate_derivative(x - h, order=order - 1)
                deriv_plus = self.evaluate_derivative(x + h, order=order - 1)
                return (deriv_plus - deriv_minus) / (2 * h)

        # --- Path B: Exact Analytical Recurrence for Standard Linear Splines ---
        if order > self.degree:
            return np.zeros((n_samples, self.n_basis))

        t = self.knots

        def _compute_deriv(p_current, current_n_basis):
            if p_current == self.degree - order:
                return self._evaluate_raw_basis(x, p_current, current_n_basis, t)

            deriv = np.zeros((n_samples, current_n_basis))
            lower_deriv = _compute_deriv(p_current - 1, current_n_basis + 1)

            for i in range(current_n_basis):
                denom1 = t[i + p_current] - t[i]
                term1 = (lower_deriv[:, i] / denom1) if denom1 > 0 else 0.0

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
