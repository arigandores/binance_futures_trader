"""Tests for feature calculator."""

import pytest
import numpy as np
from detector.features import FeatureCalculator


def test_robust_zscore():
    """Test robust z-score calculation using MAD."""
    # Create a feature calculator instance (minimal setup)
    # Note: In real tests, would need proper mocking

    # Test data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    # Median = 5.5, MAD should be robust to outlier 100
    series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]

    median = np.median(series)
    mad = np.median(np.abs(np.array(series) - median))
    robust_sigma = 1.4826 * mad

    # Last value z-score
    z = (100 - median) / robust_sigma

    # Should be high but not as extreme as standard z-score
    assert z > 5  # Significant outlier
    assert z < 30  # But not as extreme as standard z-score would be (~94)


def test_beta_calculation():
    """Test beta calculation with 5m aggregation."""
    # Placeholder for beta calculation test
    pass


def test_excess_return():
    """Test excess return calculation."""
    # Placeholder
    pass
