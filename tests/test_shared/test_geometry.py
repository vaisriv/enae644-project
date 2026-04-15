"""Tests for geometric utility functions."""

import jax.numpy as jnp
from src.shared import geometry


class TestEuclideanDistance:
    """Tests for euclidean_distance function."""

    def test_distance_between_points(self):
        """Test distance calculation between two points."""
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([3.0, 4.0])
        # Expected: 5.0 (3-4-5 triangle)
        assert jnp.isclose(geometry.euclidean_distance(p1, p2), 5.0)

    def test_distance_same_point(self):
        """Test distance from point to itself is zero."""
        p = jnp.array([2.0, 3.0])
        assert jnp.isclose(geometry.euclidean_distance(p, p), 0.0)

    def test_distance_negative_coordinates(self):
        """Test distance with negative coordinates."""
        p1 = jnp.array([-1.0, -1.0])
        p2 = jnp.array([2.0, 3.0])
        expected = jnp.sqrt(9.0 + 16.0)  # sqrt((2-(-1))^2 + (3-(-1))^2)
        assert jnp.isclose(geometry.euclidean_distance(p1, p2), expected)


class TestAngleBetween:
    """Tests for angle_between function."""

    def test_perpendicular_vectors(self):
        """Test angle between perpendicular vectors is π/2."""
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([0.0, 1.0])
        assert jnp.isclose(geometry.angle_between(v1, v2), jnp.pi / 2)

    def test_parallel_vectors(self):
        """Test angle between parallel vectors is 0."""
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([2.0, 0.0])
        assert jnp.isclose(geometry.angle_between(v1, v2), 0.0)

    def test_opposite_vectors(self):
        """Test angle between opposite vectors is π."""
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([-1.0, 0.0])
        assert jnp.isclose(geometry.angle_between(v1, v2), jnp.pi)

    def test_45_degree_angle(self):
        """Test 45-degree angle between vectors."""
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([1.0, 1.0])
        assert jnp.isclose(geometry.angle_between(v1, v2), jnp.pi / 4)


class TestNormalizeVector:
    """Tests for normalize_vector function."""

    def test_normalize_unit_vector(self):
        """Test normalizing already unit vector."""
        v = jnp.array([1.0, 0.0])
        normalized = geometry.normalize_vector(v)
        assert jnp.allclose(normalized, v)
        assert jnp.isclose(jnp.linalg.norm(normalized), 1.0)

    def test_normalize_arbitrary_vector(self):
        """Test normalizing arbitrary vector."""
        v = jnp.array([3.0, 4.0])
        normalized = geometry.normalize_vector(v)
        expected = jnp.array([3.0 / 5.0, 4.0 / 5.0])
        assert jnp.allclose(normalized, expected)

    def test_normalize_zero_vector(self):
        """Test normalizing zero vector returns zero vector."""
        v = jnp.array([0.0, 0.0])
        normalized = geometry.normalize_vector(v)
        assert jnp.allclose(normalized, v)


class TestPointToSegmentDistance:
    """Tests for point_to_segment_distance function."""

    def test_point_on_segment(self):
        """Test distance is zero when point is on segment."""
        point = jnp.array([2.0, 0.0])
        seg_start = jnp.array([0.0, 0.0])
        seg_end = jnp.array([4.0, 0.0])
        assert jnp.isclose(
            geometry.point_to_segment_distance(point, seg_start, seg_end), 0.0
        )

    def test_point_perpendicular_to_segment(self):
        """Test perpendicular distance calculation."""
        point = jnp.array([2.0, 3.0])
        seg_start = jnp.array([0.0, 0.0])
        seg_end = jnp.array([4.0, 0.0])
        # Point is 3 units above segment
        assert jnp.isclose(
            geometry.point_to_segment_distance(point, seg_start, seg_end), 3.0
        )

    def test_point_beyond_segment_end(self):
        """Test distance when point is beyond segment endpoint."""
        point = jnp.array([5.0, 0.0])
        seg_start = jnp.array([0.0, 0.0])
        seg_end = jnp.array([4.0, 0.0])
        # Closest point is segment end at (4, 0)
        assert jnp.isclose(
            geometry.point_to_segment_distance(point, seg_start, seg_end), 1.0
        )

    def test_point_before_segment_start(self):
        """Test distance when point is before segment start."""
        point = jnp.array([-1.0, 0.0])
        seg_start = jnp.array([0.0, 0.0])
        seg_end = jnp.array([4.0, 0.0])
        # Closest point is segment start at (0, 0)
        assert jnp.isclose(
            geometry.point_to_segment_distance(point, seg_start, seg_end), 1.0
        )


class TestCrossProduct2D:
    """Tests for cross_product_2d function."""

    def test_cross_product_perpendicular(self):
        """Test cross product of perpendicular vectors."""
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([0.0, 1.0])
        # Should be positive (counter-clockwise)
        assert geometry.cross_product_2d(v1, v2) > 0

    def test_cross_product_parallel(self):
        """Test cross product of parallel vectors is zero."""
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([2.0, 0.0])
        assert jnp.isclose(geometry.cross_product_2d(v1, v2), 0.0)

    def test_cross_product_clockwise(self):
        """Test cross product for clockwise rotation is negative."""
        v1 = jnp.array([0.0, 1.0])
        v2 = jnp.array([1.0, 0.0])
        # Should be negative (clockwise)
        assert geometry.cross_product_2d(v1, v2) < 0


class TestSquaredDistance:
    """Tests for squared_distance function."""

    def test_squared_distance_calculation(self):
        """Test squared distance calculation."""
        p1 = jnp.array([0.0, 0.0])
        p2 = jnp.array([3.0, 4.0])
        # Expected: 25.0 (3^2 + 4^2)
        assert jnp.isclose(geometry.squared_distance(p1, p2), 25.0)

    def test_squared_distance_same_point(self):
        """Test squared distance from point to itself is zero."""
        p = jnp.array([2.0, 3.0])
        assert jnp.isclose(geometry.squared_distance(p, p), 0.0)


class TestAngleToPoint:
    """Tests for angle_to_point function."""

    def test_angle_to_right(self):
        """Test angle to point directly to the right."""
        from_point = jnp.array([0.0, 0.0])
        to_point = jnp.array([1.0, 0.0])
        assert jnp.isclose(geometry.angle_to_point(from_point, to_point), 0.0)

    def test_angle_to_upper_right(self):
        """Test angle to point in upper right (45 degrees)."""
        from_point = jnp.array([0.0, 0.0])
        to_point = jnp.array([1.0, 1.0])
        assert jnp.isclose(geometry.angle_to_point(from_point, to_point), jnp.pi / 4)

    def test_angle_to_left(self):
        """Test angle to point directly to the left."""
        from_point = jnp.array([0.0, 0.0])
        to_point = jnp.array([-1.0, 0.0])
        assert jnp.isclose(geometry.angle_to_point(from_point, to_point), jnp.pi)


class TestClosestPointOnSegment:
    """Tests for closest_point_on_segment function."""

    def test_closest_point_on_segment_middle(self):
        """Test closest point when projection is in middle of segment."""
        point = jnp.array([2.0, 1.0])
        seg_start = jnp.array([0.0, 0.0])
        seg_end = jnp.array([4.0, 0.0])
        closest = geometry.closest_point_on_segment(point, seg_start, seg_end)
        expected = jnp.array([2.0, 0.0])
        assert jnp.allclose(closest, expected)

    def test_closest_point_is_endpoint(self):
        """Test closest point when it's the segment endpoint."""
        point = jnp.array([5.0, 1.0])
        seg_start = jnp.array([0.0, 0.0])
        seg_end = jnp.array([4.0, 0.0])
        closest = geometry.closest_point_on_segment(point, seg_start, seg_end)
        expected = jnp.array([4.0, 0.0])
        assert jnp.allclose(closest, expected)


class TestRotateVector:
    """Tests for rotate_vector function."""

    def test_rotate_90_degrees(self):
        """Test rotating vector 90 degrees counter-clockwise."""
        v = jnp.array([1.0, 0.0])
        rotated = geometry.rotate_vector(v, jnp.pi / 2)
        expected = jnp.array([0.0, 1.0])
        assert jnp.allclose(rotated, expected, atol=1e-6)

    def test_rotate_180_degrees(self):
        """Test rotating vector 180 degrees."""
        v = jnp.array([1.0, 0.0])
        rotated = geometry.rotate_vector(v, jnp.pi)
        expected = jnp.array([-1.0, 0.0])
        assert jnp.allclose(rotated, expected, atol=1e-6)

    def test_rotate_zero_degrees(self):
        """Test rotating vector 0 degrees returns same vector."""
        v = jnp.array([1.0, 1.0])
        rotated = geometry.rotate_vector(v, 0.0)
        assert jnp.allclose(rotated, v)


class TestPointInTriangle:
    """Tests for point_in_triangle function."""

    def test_point_inside_triangle(self):
        """Test point inside triangle."""
        point = jnp.array([2.0, 1.0])
        v0 = jnp.array([0.0, 0.0])
        v1 = jnp.array([4.0, 0.0])
        v2 = jnp.array([2.0, 3.0])
        assert geometry.point_in_triangle(point, v0, v1, v2)

    def test_point_outside_triangle(self):
        """Test point outside triangle."""
        point = jnp.array([5.0, 5.0])
        v0 = jnp.array([0.0, 0.0])
        v1 = jnp.array([4.0, 0.0])
        v2 = jnp.array([2.0, 3.0])
        assert not geometry.point_in_triangle(point, v0, v1, v2)

    def test_point_on_triangle_vertex(self):
        """Test point on triangle vertex."""
        point = jnp.array([0.0, 0.0])
        v0 = jnp.array([0.0, 0.0])
        v1 = jnp.array([4.0, 0.0])
        v2 = jnp.array([2.0, 3.0])
        assert geometry.point_in_triangle(point, v0, v1, v2)
