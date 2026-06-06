import numpy as np
from scipy.integrate import simpson, cumtrapz
from scipy.optimize import minimize





class curveRegistration:
    """
    Class for registering curves to a target curve.
    """
    def __init__(self, t_grid, x_basis, target_basis, warping_function):
        self.t_grid = t_grid
        self.x_basis = x_basis
        self.target_basis = target_basis
        self.warping_function = warping_function

    def power_warp(self, gamma: float):
        """
        Power transformation warping function.

        Parameters
        -----------
        gamma : float
            The warping function parameter. Must be strictly positive (gamma > 0).
            - gamma > 1: expands the beginning of the curve and compresses the end (shifts curve left)
            - gamma < 1: compresses the beginning of the curve and expands the end (shifts curve right)
            
        Returns:
        --------
        h_t : ndarray
            The warped time grid.
        """
        
        # Ensure gamma is positive
        gamma = max(1e-3, gamma)
        
        # Normalize the time grid to [0, 1]
        t_min, t_max = self.t_grid[0], self.t_grid[-1]  
        t_norm = (self.t_grid - t_min) / (t_max - t_min)
        
        # Apply power transformation
        h_norm = t_norm ** gamma
        
        # Rescale back to original time domain
        return t_min + (t_max - t_min) * h_norm

    def moebius_warp(self, f: float):
        """
        Möbius warping function.

        Parameters
        ----------
        f : float
            The warping function parameter. Must be strictly greater than -1 to avoid division by zero/negative slopes.
            - f > 0: shifts the curve to the left
            - f < 0: shifts the curve to the right

        Returns
        -------
        h_t : ndarray
            The warped time grid.
        """

        # Force parameter into a safe strictly monotonic range (-0.99, 0.99)
        f = np.clip(f, -0.99, 0.99)

        # Normalize t_grid to [0, 1]
        t_min, t_max = self.t_grid[0], self.t_grid[-1]
        t_norm = (self.t_grid - t_min) / (t_max - t_min)
        
        # Apply transformation
        h_norm = t_norm / (f * t_norm + (1 - f))
        
        # Scale back to original domain
        return t_min + (t_max - t_min) * h_norm

    def ramsay_warp(self, coefficients):
        """
        Strictly monotonic warping function as defined by Ramsay. Rather than forcing the warping function 
        to be strictly monotonic through constraints, Ramsay's method models the transformation as the 
        cumulative integral of a positive function. This ensures monotonicity without explicit constraints.

        Parameters
        ----------
        coefficients : array_like
            The weights for the basis functions.
        
        Returns
        -------
        h_t : ndarray
            The warped time grid.
        """
        
        # Get domain range
        t_min, t_max = self.t_grid[0], self.t_grid[-1]
        
        # Compute the B-spline basis matrix for the current time grid
        # The B-spline basis matrix has shape (len(t_grid), num_basis_functions)
        basis_matrix = self.x_basis.evaluate(self.t_grid)

        # W(t) is an unconstrained linear combination of basis functions
        W = basis_matrix @ coefficients
        
        # exp(W(t)) is guaranteed to be strictly positive
        exp_W = np.exp(W)
        
        # Cumulative integral for the numerator, total integral for denominator
        # cumtrapz is used to get the running integral value at each point on the grid
        numerator = cumtrapz(exp_W, t_grid, initial=0)
        denominator = simpson(exp_W, x=t_grid)
        
        # Normalize and scale to boundaries
        h_t = t_min + (t_max - t_min) * (numerator / denominator)
        return h_t

    def regsse(self):
        r"""
        Calculates the Registration Sum of Squared Errors (REGSSE) for a single curve.
        The REGSSE measures how well a warping function h(t) aligns a subject curve x(t) with the target curve \mu(t).
        
        REGSSE = integral_{t_a}^{t_b} [x(h(t)) - \mu(t)]^2 dt
            
        Returns:
        --------
        sse : float
            The integrated sum of squared errors after registration.
        """

        # Evaluate warping function on the fine grid
        h_t = self.warping_function(self.t_grid)
        
        # Ensure h(t) is strictly increasing and within bounds
        h_t = np.clip(h_t, self.t_grid[0], self.t_grid[-1])
        
        # Evaluate the registered (warped) curve: x_i(h_i(t))
        x_registered = self.x_basis.evaluate(h_t)
        mu_t = self.target_basis.evaluate(self.t_grid)
        
        # Compute the squared differences
        squared_errors = (x_registered - mu_t) ** 2
        
        # Integrate over the domain using Simpson's rule to get the SSE
        sse = simpson(squared_errors, x=self.t_grid)
        return sse

    def optimize(self, initial_guess, method='L-BFGS-B'):
        """
        Optimizes the warping function to minimize the REGSSE.
        
        Parameters:
        -----------
        initial_guess : array-like
            The initial guess for the warping function coefficients.
        method : str
            The optimization method to use.
            
        Returns:
        --------
        result : OptimizeResult
            The optimization result.
        """
        result = minimize(self.regsse, initial_guess, method=method)
        return result
    