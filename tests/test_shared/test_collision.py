"""Tests for collision detection functions."""

import jax.numpy as jnp
from src.shared import collision


class TestPointInCircle:
    """Tests for point_in_circle function."""

    def test_point_inside_circle(self):
        """Test point clearly inside circle."""
        point = jnp.array([5.0, 5.0])
        center = jnp.array([5.0, 5.0])
        radius = 2.0
        assert collision.point_in_circle(point, center, radius)

    def test_point_outside_circle(self):
        """Test point clearly outside circle."""
        point = jnp.array([10.0, 10.0])
        center = jnp.array([0.0, 0.0])
        radius = 1.0
        assert not collision.point_in_circle(point, center, radius)

    def test_point_on_circle_boundary(self):
        """Test point on circle boundary."""
        point = jnp.array([1.0, 0.0])
        center = jnp.array([0.0, 0.0])
        radius = 1.0
        assert collision.point_in_circle(point, center, radius)

    def test_point_near_boundary_inside(self):
        """Test point just inside circle boundary."""
        point = jnp.array([0.9, 0.0])
        center = jnp.array([0.0, 0.0])
        radius = 1.0
        assert collision.point_in_circle(point, center, radius)


class TestPointInPolygon:
    """Tests for point_in_polygon function."""

    def test_point_inside_triangle(self):
        """Test point inside triangular polygon."""
        point = jnp.array([2.0, 2.0])
        vertices = jnp.array([[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]])
        assert collision.point_in_polygon(point, vertices)

    def test_point_outside_triangle(self):
        """Test point outside triangular polygon."""
        point = jnp.array([10.0, 10.0])
        vertices = jnp.array([[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]])
        assert not collision.point_in_polygon(point, vertices)

    def test_point_inside_square(self):
        """Test point inside square polygon."""
        point = jnp.array([2.5, 2.5])
        vertices = jnp.array([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])
        assert collision.point_in_polygon(point, vertices)

    def test_point_outside_square(self):
        """Test point outside square polygon."""
        point = jnp.array([0.0, 0.0])
        vertices = jnp.array([[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 4.0]])
        assert not collision.point_in_polygon(point, vertices)


class TestSegmentCircleCollision:
    """Tests for segment_circle_collision function."""

    def test_segment_passes_through_circle(self):
        """Test segment that passes through circle center."""
        seg_start = jnp.array([0.0, 5.0])
        seg_end = jnp.array([10.0, 5.0])
        center = jnp.array([5.0, 5.0])
        radius = 1.0
        assert collision.segment_circle_collision(seg_start, seg_end, center, radius)

    def test_segment_misses_circle(self):
        """Test segment that doesn't intersect circle."""
        seg_start = jnp.array([0.0, 0.0])
        seg_end = jnp.array([2.0, 0.0])
        center = jnp.array([5.0, 5.0])
        radius = 1.0
        assert not collision.segment_circle_collision(
            seg_start, seg_end, center, radius
        )

    def test_segment_grazes_circle(self):
        """Test segment that just touches circle."""
        seg_start = jnp.array([0.0, 1.0])
        seg_end = jnp.array([2.0, 1.0])
        center = jnp.array([1.0, 0.0])
        radius = 1.0
        assert collision.segment_circle_collision(seg_start, seg_end, center, radius)

    def test_segment_endpoint_in_circle(self):
        """Test segment with endpoint inside circle."""
        seg_start = jnp.array([0.0, 0.0])
        seg_end = jnp.array([0.5, 0.0])
        center = jnp.array([0.0, 0.0])
        radius = 1.0
        assert collision.segment_circle_collision(seg_start, seg_end, center, radius)


class TestSegmentSegmentIntersection:
    """Tests for segment_segment_intersection function."""

    def test_intersecting_segments(self):
        """Test two segments that intersect."""
        seg1_start = jnp.array([0.0, 0.0])
        seg1_end = jnp.array([2.0, 2.0])
        seg2_start = jnp.array([0.0, 2.0])
        seg2_end = jnp.array([2.0, 0.0])
        assert collision.segment_segment_intersection(
            seg1_start, seg1_end, seg2_start, seg2_end
        )

    def test_non_intersecting_segments(self):
        """Test two segments that don't intersect."""
        seg1_start = jnp.array([0.0, 0.0])
        seg1_end = jnp.array([1.0, 0.0])
        seg2_start = jnp.array([0.0, 2.0])
        seg2_end = jnp.array([1.0, 2.0])
        assert not collision.segment_segment_intersection(
            seg1_start, seg1_end, seg2_start, seg2_end
        )

    def test_parallel_segments(self):
        """Test two parallel segments that don't intersect."""
        seg1_start = jnp.array([0.0, 0.0])
        seg1_end = jnp.array([2.0, 0.0])
        seg2_start = jnp.array([0.0, 1.0])
        seg2_end = jnp.array([2.0, 1.0])
        assert not collision.segment_segment_intersection(
            seg1_start, seg1_end, seg2_start, seg2_end
        )

    def test_touching_endpoints(self):
        """Test two segments that touch at endpoints."""
        seg1_start = jnp.array([0.0, 0.0])
        seg1_end = jnp.array([1.0, 1.0])
        seg2_start = jnp.array([1.0, 1.0])
        seg2_end = jnp.array([2.0, 0.0])
        assert collision.segment_segment_intersection(
            seg1_start, seg1_end, seg2_start, seg2_end
        )


class TestBatchCollisionCheck:
    """Tests for batch_collision_check function."""

    def test_batch_check_all_colliding(self):
        """Test batch check where all points collide."""
        points = jnp.array([[5.0, 5.0], [5.1, 5.0], [4.9, 5.0]])
        centers = jnp.array([[5.0, 5.0]])
        radii = jnp.array([1.0])
        results = collision.batch_collision_check(points, centers, radii)
        assert jnp.all(results)

    def test_batch_check_none_colliding(self):
        """Test batch check where no points collide."""
        points = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        centers = jnp.array([[10.0, 10.0]])
        radii = jnp.array([1.0])
        results = collision.batch_collision_check(points, centers, radii)
        assert not jnp.any(results)

    def test_batch_check_mixed(self):
        """Test batch check with some colliding and some not."""
        points = jnp.array([[5.0, 5.0], [0.0, 0.0], [5.1, 5.0]])
        centers = jnp.array([[5.0, 5.0]])
        radii = jnp.array([1.0])
        results = collision.batch_collision_check(points, centers, radii)
        assert results[0]  # First point collides
        assert not results[1]  # Second point doesn't collide
        assert results[2]  # Third point collides


class TestSegmentPolygonCollision:
    """Tests for segment_polygon_collision function."""

    def test_segment_crosses_polygon(self):
        """Test segment that crosses through polygon."""
        seg_start = jnp.array([0.0, 2.0])
        seg_end = jnp.array([4.0, 2.0])
        vertices = jnp.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
        assert collision.segment_polygon_collision(seg_start, seg_end, vertices)

    def test_segment_misses_polygon(self):
        """Test segment that doesn't intersect polygon."""
        seg_start = jnp.array([0.0, 0.0])
        seg_end = jnp.array([0.5, 0.0])
        vertices = jnp.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
        assert not collision.segment_polygon_collision(seg_start, seg_end, vertices)

    def test_segment_endpoint_in_polygon(self):
        """Test segment with endpoint inside polygon."""
        seg_start = jnp.array([0.0, 0.0])
        seg_end = jnp.array([2.0, 2.0])
        vertices = jnp.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
        assert collision.segment_polygon_collision(seg_start, seg_end, vertices)


class TestPathCollisionFree:
    """Tests for path_collision_free function."""

    def test_collision_free_path(self):
        """Test path that doesn't collide with obstacles."""
        path = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        circle_obstacles = [(jnp.array([5.0, 5.0]), 1.0)]
        polygon_obstacles = []
        assert collision.path_collision_free(path, circle_obstacles, polygon_obstacles)

    def test_colliding_path(self):
        """Test path that collides with obstacle."""
        path = jnp.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
        circle_obstacles = [(jnp.array([5.0, 5.0]), 1.0)]
        polygon_obstacles = []
        assert not collision.path_collision_free(
            path, circle_obstacles, polygon_obstacles
        )

    def test_path_with_polygon_obstacle(self):
        """Test path collision with polygon obstacle."""
        path = jnp.array([[0.0, 2.0], [4.0, 2.0]])
        circle_obstacles = []
        polygon_obstacles = [
            jnp.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
        ]
        assert not collision.path_collision_free(
            path, circle_obstacles, polygon_obstacles
        )
