"""Tests for geometric utility functions."""

import pytest
import jax.numpy as jnp
from src.shared import geometry


class TestEuclideanDistance:
    """Tests for euclidean_distance function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_distance_between_points(self):
        """Test distance calculation between two points."""
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([3.0, 4.0])
        # Expected: 5.0 (3-4-5 triangle)
        assert jnp.isclose(geometry.euclidean_distance(p1, p2), 5.0)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_distance_same_point(self):
        """Test distance from point to itself is zero."""
        p = jnp.array([2.0, 3.0])
        assert jnp.isclose(geometry.euclidean_distance(p, p), 0.0)


class TestAngleBetween:
    """Tests for angle_between function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_perpendicular_vectors(self):
        """Test angle between perpendicular vectors is π/2."""
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([0.0, 1.0])
        assert jnp.isclose(geometry.angle_between(v1, v2), jnp.pi / 2)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_parallel_vectors(self):
        """Test angle between parallel vectors is 0."""
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([2.0, 0.0])
        assert jnp.isclose(geometry.angle_between(v1, v2), 0.0)


class TestNormalizeVector:
    """Tests for normalize_vector function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_normalize_unit_vector(self):
        """Test normalizing already unit vector."""
        v = jnp.array([1.0, 0.0])
        normalized = geometry.normalize_vector(v)
        assert jnp.allclose(normalized, v)
        assert jnp.isclose(jnp.linalg.norm(normalized), 1.0)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_normalize_arbitrary_vector(self):
        """Test normalizing arbitrary vector."""
        v = jnp.array([3.0, 4.0])
        normalized = geometry.normalize_vector(v)
        expected = jnp.array([3.0 / 5.0, 4.0 / 5.0])
        assert jnp.allclose(normalized, expected)


class TestPointToSegmentDistance:
    """Tests for point_to_segment_distance function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_on_segment(self):
        """Test distance is zero when point is on segment."""
        point = jnp.array([2.0, 0.0])
        seg_start = jnp.array([0.0, 0.0])
        seg_end = jnp.array([4.0, 0.0])
        assert jnp.isclose(
            geometry.point_to_segment_distance(point, seg_start, seg_end),
            0.0
        )

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_perpendicular_to_segment(self):
        """Test perpendicular distance calculation."""
        point = jnp.array([2.0, 3.0])
        seg_start = jnp.array([0.0, 0.0])
        seg_end = jnp.array([4.0, 0.0])
        # Point is 3 units above segment
        assert jnp.isclose(
            geometry.point_to_segment_distance(point, seg_start, seg_end),
            3.0
        )


class TestCrossProduct2D:
    """Tests for cross_product_2d function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_cross_product_perpendicular(self):
        """Test cross product of perpendicular vectors."""
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([0.0, 1.0])
        # Should be positive (counter-clockwise)
        assert geometry.cross_product_2d(v1, v2) > 0

    @pytest.mark.skip(reason="Not implemented yet")
    def test_cross_product_parallel(self):
        """Test cross product of parallel vectors is zero."""
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([2.0, 0.0])
        assert jnp.isclose(geometry.cross_product_2d(v1, v2), 0.0)
