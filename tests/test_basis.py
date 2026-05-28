import pytest
import numpy as np
from fda.basis import BSplineBasis, FourierBasis

def test_bspline_basis_evaluation():
    domain = (0, 1)
    n_basis = 10
    degree = 3
    basis = BSplineBasis(domain, n_basis, degree=degree)
    
    # Test evaluation points
    t = np.linspace(0, 1, 100)
    Phi = basis.evaluate(t)
    
    assert Phi.shape == (100, 10)
    assert not np.any(np.isnan(Phi))
    # B-spline partition of unity (for open knot sequences with coincident endpoints)
    # Sum of basis functions is 1 inside (0, 1)
    t_internal = t[1:-1]
    assert np.allclose(np.sum(basis.evaluate(t_internal), axis=1), 1.0)

def test_bspline_basis_derivatives():
    basis = BSplineBasis((0, 10), 10, degree=3)
    t = np.linspace(0, 10, 50)
    
    # 1st and 2nd derivatives
    d1 = basis.evaluate_derivative(t, order=1)
    d2 = basis.evaluate_derivative(t, order=2)
    
    assert d1.shape == (50, 10)
    assert d2.shape == (50, 10)
    
    # High order derivative beyond degree should be all zeros
    d4 = basis.evaluate_derivative(t, order=4)
    assert np.allclose(d4, 0.0)

def test_fourier_basis_evaluation():
    domain = (0, 2 * np.pi)
    n_basis = 5  # Must be odd
    basis = FourierBasis(domain, n_basis)
    
    t = np.linspace(0, 2 * np.pi, 100)
    Phi = basis.evaluate(t)
    
    assert Phi.shape == (100, 5)
    assert not np.any(np.isnan(Phi))
    
    # Test orthogonality of the basis numerically
    W = basis.penalty_matrix(order=0, n_points=5000)
    # Since it is orthonormal, W should be close to identity
    assert np.allclose(W, np.eye(5), atol=1e-3)

def test_fourier_basis_derivatives():
    domain = (0, 1)
    n_basis = 3
    basis = FourierBasis(domain, n_basis)
    t = np.linspace(0, 1, 20)
    
    d1 = basis.evaluate_derivative(t, order=1)
    d2 = basis.evaluate_derivative(t, order=2)
    
    assert d1.shape == (20, 3)
    assert d2.shape == (20, 3)
    
    # Verify analytical derivative of sine/cosine manually at t = 0
    # First basis is constant -> derivative = 0
    assert np.allclose(d1[0, 0], 0.0)
    
    # Second basis: \sqrt{2} * sin(2*pi*t) -> derivative at t=0 is 2*pi*\sqrt{2}
    expected_d1 = 2 * np.pi * np.sqrt(2.0)
    assert np.allclose(d1[0, 1], expected_d1)

def test_penalty_matrix_properties():
    basis = BSplineBasis((0, 1), 6, degree=3)
    W = basis.penalty_matrix(order=0)
    R = basis.penalty_matrix(order=2)
    
    # Check shapes
    assert W.shape == (6, 6)
    assert R.shape == (6, 6)
    
    # Check symmetry
    assert np.allclose(W, W.T)
    assert np.allclose(R, R.T)
    
    # Check that basis overlap matrix W is positive definite (eigenvalues > 0)
    evals = np.linalg.eigvalsh(W)
    assert np.all(evals > 0)
