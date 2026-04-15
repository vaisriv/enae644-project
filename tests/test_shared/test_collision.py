"""Tests for collision detection functions."""

import pytest
import jax.numpy as jnp
from src.shared import collision


class TestPointInCircle:
    """Tests for point_in_circle function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_inside_circle(self):
        """Test point clearly inside circle."""
        point = jnp.array([5.0, 5.0])
        center = jnp.array([5.0, 5.0])
        radius = 2.0
        assert collision.point_in_circle(point, center, radius)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_outside_circle(self):
        """Test point clearly outside circle."""
        point = jnp.array([10.0, 10.0])
        center = jnp.array([0.0, 0.0])
        radius = 1.0
        assert not collision.point_in_circle(point, center, radius)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_on_circle_boundary(self):
        """Test point on circle boundary."""
        point = jnp.array([1.0, 0.0])
        center = jnp.array([0.0, 0.0])
        radius = 1.0
        assert collision.point_in_circle(point, center, radius)


class TestPointInPolygon:
    """Tests for point_in_polygon function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_inside_triangle(self):
        """Test point inside triangular polygon."""
        point = jnp.array([2.0, 2.0])
        vertices = jnp.array([[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]])
        assert collision.point_in_polygon(point, vertices)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_outside_triangle(self):
        """Test point outside triangular polygon."""
        point = jnp.array([10.0, 10.0])
        vertices = jnp.array([[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]])
        assert not collision.point_in_polygon(point, vertices)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_inside_square(self):
        """Test point inside square polygon."""
        point = jnp.array([2.5, 2.5])
        vertices = jnp.array([
            [1.0, 1.0],
            [4.0, 1.0],
            [4.0, 4.0],
            [1.0, 4.0]
        ])
        assert collision.point_in_polygon(point, vertices)


class TestSegmentCircleCollision:
    """Tests for segment_circle_collision function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_segment_passes_through_circle(self):
        """Test segment that passes through circle center."""
        seg_start = jnp.array([0.0, 5.0])
        seg_end = jnp.array([10.0, 5.0])
        center = jnp.array([5.0, 5.0])
        radius = 1.0
        assert collision.segment_circle_collision(seg_start, seg_end, center, radius)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_segment_misses_circle(self):
        """Test segment that doesn't intersect circle."""
        seg_start = jnp.array([0.0, 0.0])
        seg_end = jnp.array([2.0, 0.0])
        center = jnp.array([5.0, 5.0])
        radius = 1.0
        assert not collision.segment_circle_collision(seg_start, seg_end, center, radius)


class TestSegmentSegmentIntersection:
    """Tests for segment_segment_intersection function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_intersecting_segments(self):
        """Test two segments that intersect."""
        seg1_start = jnp.array([0.0, 0.0])
        seg1_end = jnp.array([2.0, 2.0])
        seg2_start = jnp.array([0.0, 2.0])
        seg2_end = jnp.array([2.0, 0.0])
        assert collision.segment_segment_intersection(
            seg1_start, seg1_end, seg2_start, seg2_end
        )

    @pytest.mark.skip(reason="Not implemented yet")
    def test_non_intersecting_segments(self):
        """Test two segments that don't intersect."""
        seg1_start = jnp.array([0.0, 0.0])
        seg1_end = jnp.array([1.0, 0.0])
        seg2_start = jnp.array([0.0, 2.0])
        seg2_end = jnp.array([1.0, 2.0])
        assert not collision.segment_segment_intersection(
            seg1_start, seg1_end, seg2_start, seg2_end
        )


class TestBatchCollisionCheck:
    """Tests for batch_collision_check function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_batch_check_all_colliding(self):
        """Test batch check where all points collide."""
        points = jnp.array([[5.0, 5.0], [5.1, 5.0], [4.9, 5.0]])
        centers = jnp.array([[5.0, 5.0]])
        radii = jnp.array([1.0])
        results = collision.batch_collision_check(points, centers, radii)
        assert jnp.all(results)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_batch_check_none_colliding(self):
        """Test batch check where no points collide."""
        points = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        centers = jnp.array([[10.0, 10.0]])
        radii = jnp.array([1.0])
        results = collision.batch_collision_check(points, centers, radii)
        assert not jnp.any(results)
