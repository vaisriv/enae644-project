"""Tests for workspace utilities."""

import jax.numpy as jnp
from src.shared.workspace import (
    CircleObstacle,
    create_workspace,
    is_in_bounds,
    is_collision_free,
    is_in_workspace,
    sample_collision_free_point,
)


class TestCreateWorkspace:
    """Tests for create_workspace function."""

    def test_create_empty_workspace(self):
        """Test creating workspace with no obstacles."""
        bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
        ws = create_workspace(bounds, [])
        assert jnp.allclose(ws.bounds, bounds)
        assert len(ws.obstacles) == 0

    def test_create_workspace_with_obstacles(self):
        """Test creating workspace with obstacles."""
        bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
        obs = [CircleObstacle(center=jnp.array([5.0, 5.0]), radius=1.0)]
        ws = create_workspace(bounds, obs)
        assert len(ws.obstacles) == 1


class TestIsInBounds:
    """Tests for is_in_bounds function."""

    def test_point_inside_bounds(self):
        """Test point clearly inside bounds."""
        point = jnp.array([5.0, 5.0])
        bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
        assert is_in_bounds(point, bounds)

    def test_point_outside_bounds(self):
        """Test point outside bounds."""
        point = jnp.array([15.0, 5.0])
        bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
        assert not is_in_bounds(point, bounds)

    def test_point_on_boundary(self):
        """Test point on boundary is considered inside."""
        point = jnp.array([0.0, 5.0])
        bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
        assert is_in_bounds(point, bounds)

    def test_point_outside_y_bounds(self):
        """Test point outside y bounds."""
        point = jnp.array([5.0, 15.0])
        bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
        assert not is_in_bounds(point, bounds)

    def test_point_at_corner(self):
        """Test point at corner of bounds."""
        point = jnp.array([10.0, 10.0])
        bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
        assert is_in_bounds(point, bounds)


class TestIsCollisionFree:
    """Tests for is_collision_free function."""

    def test_point_collision_free(self):
        """Test point with no obstacle collision."""
        point = jnp.array([1.0, 1.0])
        obstacles = [CircleObstacle(center=jnp.array([5.0, 5.0]), radius=1.0)]
        assert is_collision_free(point, obstacles)

    def test_point_colliding_with_circle(self):
        """Test point colliding with circle obstacle."""
        point = jnp.array([5.0, 5.0])
        obstacles = [CircleObstacle(center=jnp.array([5.0, 5.0]), radius=1.0)]
        assert not is_collision_free(point, obstacles)

    def test_point_collision_free_multiple_obstacles(self):
        """Test point collision-free with multiple obstacles."""
        point = jnp.array([0.0, 0.0])
        obstacles = [
            CircleObstacle(center=jnp.array([5.0, 5.0]), radius=1.0),
            CircleObstacle(center=jnp.array([8.0, 8.0]), radius=0.5),
        ]
        assert is_collision_free(point, obstacles)

    def test_point_colliding_with_one_of_many(self):
        """Test point colliding with one obstacle among many."""
        point = jnp.array([5.0, 5.0])
        obstacles = [
            CircleObstacle(center=jnp.array([5.0, 5.0]), radius=1.0),
            CircleObstacle(center=jnp.array([8.0, 8.0]), radius=0.5),
        ]
        assert not is_collision_free(point, obstacles)


class TestIsInWorkspace:
    """Tests for is_in_workspace function."""

    def test_valid_point_in_workspace(self, simple_workspace):
        """Test point that is both in bounds and collision-free."""
        point = jnp.array([5.0, 5.0])
        assert is_in_workspace(point, simple_workspace)

    def test_point_out_of_bounds(self, simple_workspace):
        """Test point outside bounds."""
        point = jnp.array([15.0, 5.0])
        assert not is_in_workspace(point, simple_workspace)

    def test_point_colliding(self, workspace_with_circle_obstacle):
        """Test point in bounds but colliding with obstacle."""
        point = jnp.array([5.0, 5.0])  # Center of circle
        assert not is_in_workspace(point, workspace_with_circle_obstacle)

    def test_point_collision_free_in_workspace(self, workspace_with_circle_obstacle):
        """Test valid point in workspace with obstacles."""
        point = jnp.array([1.0, 1.0])  # Away from obstacle
        assert is_in_workspace(point, workspace_with_circle_obstacle)


class TestSampleCollisionFreePoint:
    """Tests for sample_collision_free_point function."""

    def test_sample_in_empty_workspace(self, simple_workspace, jax_key):
        """Test sampling in workspace with no obstacles."""
        point = sample_collision_free_point(simple_workspace, jax_key)
        assert is_in_workspace(point, simple_workspace)

    def test_sample_with_obstacles(self, workspace_with_circle_obstacle, jax_key):
        """Test sampling in workspace with obstacles."""
        point = sample_collision_free_point(workspace_with_circle_obstacle, jax_key)
        assert is_in_workspace(point, workspace_with_circle_obstacle)

    def test_sample_multiple_points(self, simple_workspace, jax_key_sequence):
        """Test sampling multiple collision-free points."""
        for key in jax_key_sequence[:5]:
            point = sample_collision_free_point(simple_workspace, key)
            assert is_in_workspace(point, simple_workspace)

    def test_sampled_point_in_bounds(self, simple_workspace, jax_key):
        """Test that sampled point is within workspace bounds."""
        point = sample_collision_free_point(simple_workspace, jax_key)
        bounds = simple_workspace.bounds
        assert bounds[0, 0] <= point[0] <= bounds[0, 1]
        assert bounds[1, 0] <= point[1] <= bounds[1, 1]
