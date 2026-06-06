import numpy as np
from scipy.integrate import simpson


class curveRegistration:
    """
    Class for registering curves to a target curve.
    """
    def __init__(self, t_grid, x_basis, target_basis, warping_function):
        self.t_grid = t_grid
        self.x_basis = x_basis
        self.target_basis = target_basis
        self.warping_function = warping_function

        
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