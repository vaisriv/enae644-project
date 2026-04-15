"""Tests for metrics computation."""

import pytest
import jax.numpy as jnp
from src.simulation import metrics


class TestComputeObserverAccuracy:
    """Tests for compute_observer_accuracy function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_observer_accuracy_calculation(self, mock_observer_net, straight_line_trajectory):
        """Test observer accuracy computation."""
        # This will test the metrics once implemented
        pass


class TestComputePathLengthRatio:
    """Tests for compute_path_length_ratio function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_path_length_ratio_straight_line(self, straight_line_trajectory):
        """Test path length ratio for straight line (should be 1.0)."""
        # Optimal path is straight line, actual is straight line
        # Ratio should be 1.0
        pass


class TestComputeBeliefEntropy:
    """Tests for compute_belief_entropy function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_entropy_uniform_distribution(self):
        """Test entropy of uniform distribution."""
        # Uniform distribution over 3 goals should have max entropy
        belief = jnp.array([1/3, 1/3, 1/3])
        # entropy = -sum(p * log(p)) = -3 * (1/3 * log(1/3)) = log(3)
        pass

    @pytest.mark.skip(reason="Not implemented yet")
    def test_entropy_deterministic_distribution(self):
        """Test entropy of deterministic distribution is zero."""
        belief = jnp.array([1.0, 0.0, 0.0])
        # entropy should be 0
        pass


class TestComputeInterceptionDistance:
    """Tests for compute_interception_distance function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_min_distance_between_trajectories(self):
        """Test minimum distance calculation between two trajectories."""
        pass


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_all_metrics_dict_structure(self):
        """Test that compute_all_metrics returns complete dict."""
        # Should return dict with all expected keys
        pass
