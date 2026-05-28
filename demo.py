import numpy as np
import matplotlib.pyplot as plt
from fda import BSplineBasis, smooth_data, functional_mean, functional_variance, FPCA

def run_demo():
    # Set seed for reproducibility
    np.random.seed(42)
    
    # -------------------------------------------------------------
    # 1. Generate Synthetic Functional Data
    # -------------------------------------------------------------
    # True curves: y_i(t) = a_i * sin(t) + b_i * cos(2*t)
    # where a_i ~ N(3, 1), b_i ~ N(1, 0.3)
    n_curves = 40
    n_points = 100
    t = np.linspace(0, 2 * np.pi, n_points)
    
    a = np.random.normal(3.0, 1.0, size=(n_curves, 1))
    b = np.random.normal(1.0, 0.3, size=(n_curves, 1))
    
    # Clean curves
    curves_clean = a * np.sin(t) + b * np.cos(2 * t)
    
    # Add observational noise
    noise = np.random.normal(0, 0.4, size=curves_clean.shape)
    curves_noisy = curves_clean + noise
    
    # -------------------------------------------------------------
    # 2. Smooth the Data using B-Spline Basis
    # -------------------------------------------------------------
    # Create a B-spline basis with 12 basis functions and degree 3 (cubic)
    basis = BSplineBasis(domain_range=(0, 2 * np.pi), n_basis=12, degree=3)
    
    # Project and smooth discrete observations using penalized least squares (lambda = 0.05)
    fd = smooth_data(t, curves_noisy, basis, lmbda=0.05, penalty_order=2)
    
    # Evaluate smoothed curves on a dense grid
    t_dense = np.linspace(0, 2 * np.pi, 200)
    curves_smooth = fd(t_dense)
    
    # -------------------------------------------------------------
    # 3. Compute Functional Mean and Variance
    # -------------------------------------------------------------
    fd_mean = functional_mean(fd)
    mean_curve = fd_mean(t_dense)[0]
    
    var_vals = functional_variance(fd, t_dense)
    std_curve = np.sqrt(var_vals)
    
    # -------------------------------------------------------------
    # 4. Perform Functional PCA (FPCA)
    # -------------------------------------------------------------
    fpca = FPCA(n_components=2)
    scores = fpca.fit_transform(fd)
    
    # Evaluate principal components (eigenfunctions)
    pcs = fpca.components_(t_dense)
    
    # -------------------------------------------------------------
    # 5. Create a Beautiful, Sleek Visualization
    # -------------------------------------------------------------
    # Use a modern, high-quality style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#EEEEEE'
    plt.rcParams['grid.linewidth'] = 0.5
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    fig.suptitle("Functional Data Analysis (FDA) Module Capabilities", fontsize=18, fontweight='bold', color='#1A1A1A', y=0.96)
    
    # Subplot 1: Noisy vs Smoothed curves
    ax = axs[0, 0]
    ax.set_facecolor('#FAFAFA')
    ax.grid(True)
    # Plot first 5 curves for clarity
    for i in range(5):
        line, = ax.plot(t_dense, curves_smooth[i], lw=2.5, alpha=0.95, label=f"Curve {i+1} Smoothed" if i == 0 else "")
        ax.scatter(t, curves_noisy[i], color=line.get_color(), s=15, alpha=0.4, label=f"Curve {i+1} Noisy" if i == 0 else "")
    ax.set_title("1. Noisy Data Smoothing (B-Splines)", fontsize=13, fontweight='semibold', color='#333333', pad=10)
    ax.set_xlabel("Time (t)", fontsize=10)
    ax.set_ylabel("x(t)", fontsize=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='#E0E0E0')
    
    # Subplot 2: Mean and Variance
    ax = axs[0, 1]
    ax.set_facecolor('#FAFAFA')
    ax.grid(True)
    # Plot all smoothed curves in light grey background
    for i in range(n_curves):
        ax.plot(t_dense, curves_smooth[i], color='#CCCCCC', lw=0.5, alpha=0.5)
    # Plot mean and std band
    ax.plot(t_dense, mean_curve, color='#D32F2F', lw=3.0, label='Functional Mean')
    ax.fill_between(t_dense, mean_curve - std_curve, mean_curve + std_curve, 
                    color='#D32F2F', alpha=0.15, label='Mean ± 1 Std Dev')
    ax.set_title("2. Functional Descriptive Statistics", fontsize=13, fontweight='semibold', color='#333333', pad=10)
    ax.set_xlabel("Time (t)", fontsize=10)
    ax.set_ylabel("x(t)", fontsize=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='#E0E0E0')
    
    # Subplot 3: FPCA Principal Components
    ax = axs[1, 0]
    ax.set_facecolor('#FAFAFA')
    ax.grid(True)
    ax.plot(t_dense, pcs[0], color='#1976D2', lw=2.5, label=f"FPC 1 ({fpca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.plot(t_dense, pcs[1], color='#388E3C', lw=2.5, label=f"FPC 2 ({fpca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.set_title("3. Principal Component Functions (Harmonics)", fontsize=13, fontweight='semibold', color='#333333', pad=10)
    ax.set_xlabel("Time (t)", fontsize=10)
    ax.set_ylabel("Eigenfunction Value", fontsize=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='#E0E0E0')
    
    # Subplot 4: Score Scatter Plot color-coded by True Amplitude 'a'
    ax = axs[1, 1]
    ax.set_facecolor('#FAFAFA')
    ax.grid(True)
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=a.squeeze(), cmap='plasma', s=60, edgecolor='white', linewidth=0.5)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("True Sine Amplitude (a)", fontsize=10)
    ax.set_title("4. FPCA Scores Space", fontsize=13, fontweight='semibold', color='#333333', pad=10)
    ax.set_xlabel("FPC 1 Score", fontsize=10)
    ax.set_ylabel("FPC 2 Score", fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save the output image
    output_path = "fda_demo_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Demo run completed successfully. Visualization saved to {output_path}.")
    print(f"Explained variance of first 2 components: {fpca.explained_variance_ratio_ * 100}%")

if __name__ == "__main__":
    run_demo()
