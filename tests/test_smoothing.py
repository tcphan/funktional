import pytest
import numpy as np
from fda.basis import BSplineBasis
from fda.representation import FunctionalData
from fda.smoothing import smooth_data
from fda.stats import functional_mean, functional_variance, functional_covariance

def test_functional_data_basic():
    basis = BSplineBasis((0, 1), 5)
    coef = np.random.randn(3, 5)  # 3 curves, 5 basis coefficients
    
    fd = FunctionalData(basis, coef)
    
    assert fd.n_curves == 3
    assert fd.n_basis == 5
    assert fd.domain_range == (0.0, 1.0)
    
    # Evaluate at points
    t = np.linspace(0, 1, 10)
    res = fd(t)
    assert res.shape == (3, 10)
    
    # Evaluate derivative
    deriv = fd.evaluate_derivative(t, order=1)
    assert deriv.shape == (3, 10)

def test_functional_data_arithmetic():
    basis = BSplineBasis((0, 1), 5)
    coef1 = np.ones((2, 5))
    coef2 = np.ones((2, 5)) * 2
    
    fd1 = FunctionalData(basis, coef1)
    fd2 = FunctionalData(basis, coef2)
    
    # Test addition
    fd_add = fd1 + fd2
    assert np.allclose(fd_add.coef, 3.0)
    
    # Test subtraction
    fd_sub = fd2 - fd1
    assert np.allclose(fd_sub.coef, 1.0)
    
    # Test scalar multiplication
    fd_mul = fd1 * 5
    assert np.allclose(fd_mul.coef, 5.0)

def test_data_smoothing_shared_grid():
    basis = BSplineBasis((0, 1), 6, degree=3)
    t = np.linspace(0, 1, 50)
    
    # Create a true curve: sine wave
    true_y = np.sin(2 * np.pi * t)
    
    # Add noise
    np.random.seed(42)
    noisy_y = true_y + np.random.normal(0, 0.1, size=t.shape)
    
    # Smooth (unpenalized least squares)
    fd = smooth_data(t, noisy_y, basis, lmbda=0.0)
    assert fd.n_curves == 1
    assert fd.coef.shape == (1, 6)
    
    # Reconstructed values should be close to true sine wave
    recon_y = fd(t)[0]
    assert np.allclose(recon_y, true_y, atol=0.2)

def test_data_smoothing_penalized():
    basis = BSplineBasis((0, 1), 15, degree=3)
    t = np.linspace(0, 1, 100)
    true_y = np.sin(2 * np.pi * t)
    
    np.random.seed(42)
    noisy_y = true_y + np.random.normal(0, 0.3, size=t.shape)
    
    # High lambda should lead to very smooth, almost straight line curves (over-smoothed)
    fd_smooth = smooth_data(t, noisy_y, basis, lmbda=1.0)
    fd_rough = smooth_data(t, noisy_y, basis, lmbda=0.0)
    
    # Double derivative integral of smooth curve should be smaller than that of rough curve
    R = basis.penalty_matrix(order=2)
    int_deriv_smooth = fd_smooth.coef[0] @ R @ fd_smooth.coef[0]
    int_deriv_rough = fd_rough.coef[0] @ R @ fd_rough.coef[0]
    
    assert int_deriv_smooth < int_deriv_rough

def test_data_smoothing_individual_grids():
    basis = BSplineBasis((0, 1), 6)
    
    # 2 curves with different grids
    t_list = [np.linspace(0, 1, 30), np.linspace(0.1, 0.9, 20)]
    y_list = [np.sin(2 * np.pi * t_list[0]), np.cos(2 * np.pi * t_list[1])]
    
    fd = smooth_data(t_list, y_list, basis, lmbda=0.01)
    
    assert fd.n_curves == 2
    assert fd.coef.shape == (2, 6)

def test_functional_statistics():
    basis = BSplineBasis((0, 1), 5)
    coef = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0, 7.0],
        [2.0, 3.0, 4.0, 5.0, 6.0]
    ])
    fd = FunctionalData(basis, coef)
    
    # 1. Mean
    fd_mean = functional_mean(fd)
    assert fd_mean.n_curves == 1
    assert np.allclose(fd_mean.coef[0], [2.0, 3.0, 4.0, 5.0, 6.0])
    
    # 2. Variance at grid
    t = np.linspace(0, 1, 10)
    var_vals = functional_variance(fd, t)
    assert len(var_vals) == 10
    assert np.all(var_vals >= 0)
    
    # Since variance of coeffs is: [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], etc.] -> let's compute directly:
    # Var(coef) for each column is 1.0 (mean values: 1, 3, 2 -> var = ((1-2)^2 + (3-2)^2 + (2-2)^2)/2 = 1.0)
    # The actual variance function should be positive
    assert np.all(var_vals > 0)
    
    # 3. Covariance surface
    cov_surface = functional_covariance(fd, eval_points1=t, eval_points2=t)
    assert cov_surface.shape == (10, 10)
    assert np.allclose(cov_surface, cov_surface.T)  # Auto-covariance is symmetric
