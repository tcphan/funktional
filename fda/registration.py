import numpy as np
from scipy.integrate import simpson, cumtrapz
from scipy.optimize import minimize

class warpingFunction:
    """
    Base class for warping functions.
    """
    def __init__(self, t_grid):
        self.t_grid = t_grid
        self.t_min = t_grid[0]
        self.t_max = t_grid[-1]

    def __call__(self, coefficients):
        return self.warp(coefficients)

class powerWarp(warpingFunction):
    """
    Power transformation warping function.

    Parameters
    ----------
    t_grid : array_like
        The time grid.
    """

    def __init__(self, t_grid):
        super().__init__(t_grid)

    def warp(self, gamma: float):
        """
        Power transformation warping function.

        Parameters
        ----------
        gamma : float
            The warping function parameter.
            Must be strictly positive (gamma > 0).
            
        Interpretations:
        ----------------
            - gamma > 1: expands the beginning of the curve and compresses the end (shifts curve left)
            - gamma < 1: compresses the beginning of the curve and expands the end (shifts curve right)

        Returns
        -------
        h_t : ndarray
            The warped time grid.
        """

        # Ensure gamma is positive
        gamma = max(1e-3, gamma)
        
        # Normalize the time grid to [0, 1]
        t_norm = (self.t_grid - self.t_min) / (self.t_max - self.t_min)
        
        # Apply power transformation
        h_norm = t_norm ** gamma
        
        # Rescale back to original time domain
        return self.t_min + (self.t_max - self.t_min) * h_norm

class mobiusWarp(warpingFunction):
    """
    Möbius transformation warping function.

    Parameters
    ----------
    t_grid : array_like
        The time grid.
    """
    def __init__(self, t_grid):
        super().__init__(t_grid)

    def warp(self, f: float):
        """
        Möbius transformation warping function.

        Parameters
        ----------
        f : float
            The warping function parameter.
            Must be strictly greater than -1 to avoid division by zero/negative slopes.

        Interpretations:
        ----------------
            - f > 0: shifts the curve to the left
            - f < 0: shifts the curve to the right

        Returns
        -------
        h_t : ndarray
            The warped time grid.
        """
        # Ensure f is strictly monotonic
        f = np.clip(f, -0.99, 0.99)
        
        # Normalize the time grid to [0, 1]
        t_norm = (self.t_grid - self.t_min) / (self.t_max - self.t_min)
        
        # Apply Möbius transformation
        h_norm = t_norm / (f * t_norm + (1 - f))
        
        # Rescale back to original time domain
        return self.t_min + (self.t_max - self.t_min) * h_norm

class ramsayWarp(warpingFunction):
    """
    Ramsay's basis transformation warping function. Rather than forcing the warping function 
    to be strictly monotonic through constraints, Ramsay's method models the transformation as the 
    cumulative integral of a positive function. This ensures monotonicity without explicit constraints.

    Parameters
    ----------
    t_grid : array_like
        The time grid.
    """
    
    def __init__(self, t_grid, basis_object):
        super().__init__(t_grid)
        self.basis_object = basis_object

    def warp(self, coefficients):
        """
        Ramsay's basis transformation warping function.

        Parameters
        ----------
        coefficients : ndarray
            The basis function coefficients.

        Returns
        -------
        h_t : ndarray
            The warped time grid.
        """

        # Ensure coefficients are a numpy array
        coefficients = np.asarray(coefficients)
        
        # Get the basis function values
        basis_values = self.basis_object.evaluate(self.t_grid)
        
        # W(t) is an unconstrained linear combination of basis functions
        W = basis_values @ coefficients
        
        # exp(W(t)) is guaranteed to be strictly positive
        exp_W = np.exp(W)
        
        # Cumulative integral for the numerator, total integral for denominator
        # cumtrapz is used to get the running integral value at each point on the grid
        numerator = cumtrapz(exp_W, self.t_grid, initial=0)
        denominator = simpson(exp_W, x=self.t_grid)
        
        # Normalize and scale to boundaries
        h_t = self.t_min + (self.t_max - self.t_min) * (numerator / denominator)
        return h_t


class curveRegistration:
    """
    Class for registering curves to a target curve.
    """
    def __init__(self, t_grid, x_basis, target, warping_method="ramsay"):
        self.t_grid = t_grid
        self.x_basis = x_basis
        self.target = target
        self.warping_method = warping_method.lower()

    def _get_default_initial_guess(self):
        """
        Generates the default baseline (identity/no-warp) coefficients depending on the selected warping method.
        """

        if self.warping_method == "power":
            # gamma = 1.0 means t^1 = t (Identity)
            return np.array([1.0])
            
        elif self.warping_method == "moebius":
            # f = 0.0 means t / (0 + 1) = t (Identity)
            return np.array([0.0])
            
        elif self.warping_method == "ramsay":
            # Ramsay's basis matrix shape depends on how many basis functions x_basis has
            num_basis = self.x_basis.evaluate(self.t_grid).shape[1]
            # Zero coefficients mean exp(0) = 1 (constant), integrating to a linear line (Identity)
            return np.zeros(num_basis)

        else:
            raise ValueError(f"Unknown warping method: {self.warping_method}")

    def _warping_function(self, coefficients):
        """
        Apply the warping function based on the specified method.
        """

        valid_methods = [
            "power",
            "moebius",
            "ramsay"
        ]

        coef_val = coefficients[0] if isinstance(coefficients, (np.ndarray, list)) else coefficients

        if self.warping_method == "power":
            return self.power_warp(coef_val)
        elif self.warping_method == "moebius":
            return self.moebius_warp(coef_val)
        elif self.warping_method == "ramsay":
            return self.ramsay_warp(coefficients)
        else:
            raise ValueError(f"Invalid warping method '{self.warping_method}'. Choose from {valid_methods}.")

    def regsse(self, coefficients):
        r"""
        Calculates the Registration Sum of Squared Errors (REGSSE) for a single curve.
        The REGSSE measures how well a warping function h(t) aligns a subject curve x(t) with the target curve \mu(t).
        
        REGSSE = integral_{t_a}^{t_b} [x(h(t)) - \mu(t)]^2 dt

        Parameters
        ----------
        coefficients : array_like
            The weights used by the warping function to transform the time grid.
            
        Returns:
        --------
        sse : float
            The integrated sum of squared errors after registration.
        """

        # Evaluate warping function on the fine grid
        h_t = self._warping_function(coefficients)
        
        # Ensure h(t) is strictly increasing and within bounds
        h_t = np.clip(h_t, self.t_grid[0], self.t_grid[-1])
        
        # Evaluate the registered (warped) curve: x_i(h_i(t))
        x_registered = self.x_basis.evaluate(h_t)
        mu_t = self.target
        
        # Compute the squared differences
        squared_errors = (x_registered - mu_t) ** 2
        
        # Integrate over the domain using Simpson's rule to get the SSE
        sse = simpson(squared_errors, x=self.t_grid)
        return sse

    def optimize(self, initial_guess=None, optimization_method='L-BFGS-B'):
        """
        Optimizes the warping function to minimize the REGSSE.
        
        Parameters:
        -----------
        initial_guess : array-like
            The initial guess for the warping function coefficients.
        optimization_method : str
            The optimization method to use.
            
        Returns:
        --------
        result : OptimizeResult
            The optimization result.
        """

        if initial_guess is None:
            initial_guess = self._get_default_initial_guess()

        # For certain warping methods (e.g. power, moebius), it helps to bound the optimizer step-sizes so it doesn't try negative exponents or asymptotic boundaries
        bounds = None
        if optimization_method in ["L-BFGS-B", "SLSQP", "Trust-Region"]:
            if self.warping_method == "power":
                bounds = [(1e-3, 10.0)]
            elif self.warping_method == "moebius":
                bounds = [(-0.95, 0.95)]

        # Minimize the REGSSE
        result = minimize(
            fun=self.regsse,         # Registration SSE objective function
            x0=initial_guess,        # Starting guess
            method=optimization_method,           # Optimization method
            bounds=bounds            # Bounds for the coefficients
        )
        return result
    