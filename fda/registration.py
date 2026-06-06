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
            - gamma > 1: expands the beginning of the curve and compresses the end (shifts curve left or delays features)
            - gamma < 1: compresses the beginning of the curve and expands the end (shifts curve right or accelerates features)
            
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
    