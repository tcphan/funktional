import numpy as np
import matplotlib.pyplot as plt

class BezierCurve:

    def __init__(self, control_points):
        """
        Initializes the BezierCurve object.
        
        Args:
            control_points: Array containing the coordinates of the control points (# control points x # dimensions)
        """
        
        self.control_points = np.array(control_points)
        self.t_vals = np.linspace(0, 1, 100)
        self.n_dim = self.control_points.shape[1]

        # Raise error if less than 2 control points are provided
        if len(control_points) < 2:
            raise ValueError("At least 2 control points are required to define a Bezier curve.")
    
    def _linear_interpolation(self, a, b, t):
        """
        Interpolates between two points.
        
        Args:
            a: Array containing coordinates for control point A
            b: Array containing coordinates for control point B
            t: Value between 0 and 1 that determines the weight of the two control points
            
        Returns:
            Interpolated point
        """

        return a * (1 - t) + b * t

    def _get_bezier_coordinate(self, t):
        """
        Calculates the coordinates of a Bezier curve for a given parameter t.

        Args:
            t: Value between 0 and 1 that determines the weight of the two control points
            
        Returns:
            Coordinates of the Bezier curve for the given parameter t
        """

        n_lerp_points = len(self.control_points)
        points = self.control_points.copy()

        # Loop through each control points and determine the interpolated points
        while n_lerp_points > 1:

            lerp_points = []

            # Interpolate each adjacent pair of control points
            for p in range(n_lerp_points-1):
                start_point = points[p]
                end_point = points[p+1]
                lerp_points.append(self._linear_interpolation(start_point, end_point, t))

            # Update the list of points for the next iteration
            points = lerp_points
            n_lerp_points -= 1

        # Return the final interpolated point
        return lerp_points[0]

    def get_bezier_curve(self):
        """
        Calculates the coordinates of a Bezier curve for all t in t_vals.
            
        Returns:
            Array of coordinates for the Bezier curve for all values of t in t_vals
        """
        
        coordinates = [self._get_bezier_coordinate(t) for t in self.t_vals]
        return np.array(coordinates)

    def plot_component_curve(self):
        """
        Plots the component coordinates of the Bezier curve.
        
        Returns:
            Axis with the plotted components
        """

        # Get the figure and axes
        fig, ax = plt.subplots(figsize=(12,5))
        
        # Plot the components
        for i in range(self.n_dim):
            
            # Normalize the bezier values
            bezier_values = self.get_bezier_curve()[:, i]
            min_bezier_val = np.min(bezier_values)
            max_bezier_val = np.max(bezier_values)
            
            # Avoid division by zero
            if (max_bezier_val - min_bezier_val) == 0:
                normalized_bezier_values = np.zeros(len(bezier_values))
            else:
                normalized_bezier_values = (bezier_values - min_bezier_val) / (max_bezier_val - min_bezier_val)
            
            # Plot the curve
            ax.plot(
                self.t_vals,
                normalized_bezier_values,
                linewidth=2,
                label=f"Dimension {i+1}",
                color=plt.cm.get_cmap("Set3", self.n_dim)(i)
            )

        # Apply formatting
        ax.set_title("Component Curves", fontsize=10, fontweight="bold", pad=10)
        ax.set_xlabel("$t$", fontsize=9)
        ax.set_ylabel("Normalized B$_i$(t)", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()
        plt.show()


    def plot_bezier_curve(self):
        """
        Plots the Bezier curve defined by the control points.
        
        Returns:
            Axis with the plotted curve
        """

        # Get the figure and axes
        if self.n_dim >= 3:
            fig = plt.figure(figsize=(12,6))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(12,5))
        
        # Identify how many axes to plot
        # If number of dimensions for the Bezier curve exceeds 3, then will only plot the first 3 dimensions
        n_axes = min(3, self.n_dim)

        # Get the coordinates for the curve
        coordinates = [self.get_bezier_curve()[:, i] for i in range(n_axes)]
        
        # Print statement if the number of dimensions for the Bezier curve exceeds 3
        if self.n_dim > 3:
            print(f"Number of dimensions is {self.n_dim}. Displaying only the first 3 dimensions of the Bezier curve.")
        
        # Plot the curve
        ax.plot(*coordinates, linewidth=2, color="#4F46E5", label="Bezier Curve")
        
        # Apply formatting
        ax.set_title("Bezier Curve", fontsize=10, fontweight="bold", pad=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.show()
