import pytest
import numpy as np
from fda.basis import BSplineBasis
from fda.representation import FunctionalData
from fda.fpca import FPCA
from fda.smoothing import smooth_data

def test_fpca_basic():
    # Generate some simple curves: y_i(t) = a_i * sin(t) + b_i * cos(t)
    basis = BSplineBasis((0, 2 * np.pi), 10)
    t = np.linspace(0, 2 * np.pi, 100)
    
    np.random.seed(42)
    n_curves = 15
    a = np.random.normal(0, 2, size=(n_curves, 1))
    b = np.random.normal(0, 0.5, size=(n_curves, 1))
    
    y = a * np.sin(t) + b * np.cos(t)
    fd = smooth_data(t, y, basis)
    
    # Run FPCA
    fpca = FPCA(n_components=2)
    fpca.fit(fd)
    
    # Verify properties
    assert fpca.eigenvalues_.shape == (2,)
    assert fpca.components_.n_curves == 2
    assert fpca.components_.basis.n_basis == 10
    
    # First component eigenvalue should be much larger than the second,
    # because 'a' variance is 4, 'b' variance is 0.25
    assert fpca.eigenvalues_[0] > fpca.eigenvalues_[1]
    
    # Explained variance ratios
    assert len(fpca.explained_variance_ratio_) == 2
    assert np.sum(fpca.explained_variance_ratio_) <= 1.0
    
    # Test scores shape
    scores = fpca.transform(fd)
    assert scores.shape == (n_curves, 2)
    
    # Test fit_transform equivalent
    scores2 = fpca.fit_transform(fd)
    assert np.allclose(scores, scores2)
    
    # Test reconstruction
    fd_recon = fpca.inverse_transform(scores)
    assert fd_recon.n_curves == n_curves
    assert fd_recon.coef.shape == fd.coef.shape
    
    # Reconstructed curves should be close to original curves
    assert np.allclose(fd_recon(t), fd(t), atol=0.2)

def test_fpca_exceptions():
    basis = BSplineBasis((0, 1), 5)
    coef = np.random.randn(1, 5)
    fd_single = FunctionalData(basis, coef)
    
    fpca = FPCA(n_components=2)
    
    # FPCA should fail with a single curve
    with pytest.raises(ValueError, match="FPCA requires at least 2 curves"):
        fpca.fit(fd_single)
        
    # FPCA should fail if calling transform before fitting
    with pytest.raises(RuntimeError, match="must be fitted"):
        fpca.transform(fd_single)
        
    # FPCA should fail if calling inverse_transform before fitting
    with pytest.raises(RuntimeError, match="must be fitted"):
        fpca.inverse_transform(np.zeros((1, 2)))
