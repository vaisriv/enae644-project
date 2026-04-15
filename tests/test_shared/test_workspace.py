"""Tests for workspace utilities."""

import pytest
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

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_inside_bounds(self):
        """Test point clearly inside bounds."""
        point = jnp.array([5.0, 5.0])
        bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
        assert is_in_bounds(point, bounds)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_outside_bounds(self):
        """Test point outside bounds."""
        point = jnp.array([15.0, 5.0])
        bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
        assert not is_in_bounds(point, bounds)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_on_boundary(self):
        """Test point on boundary is considered inside."""
        point = jnp.array([0.0, 5.0])
        bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
        assert is_in_bounds(point, bounds)


class TestIsCollisionFree:
    """Tests for is_collision_free function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_collision_free(self):
        """Test point with no obstacle collision."""
        point = jnp.array([1.0, 1.0])
        obstacles = [CircleObstacle(center=jnp.array([5.0, 5.0]), radius=1.0)]
        assert is_collision_free(point, obstacles)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_colliding_with_circle(self):
        """Test point colliding with circle obstacle."""
        point = jnp.array([5.0, 5.0])
        obstacles = [CircleObstacle(center=jnp.array([5.0, 5.0]), radius=1.0)]
        assert not is_collision_free(point, obstacles)


class TestIsInWorkspace:
    """Tests for is_in_workspace function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_valid_point_in_workspace(self, simple_workspace):
        """Test point that is both in bounds and collision-free."""
        point = jnp.array([5.0, 5.0])
        assert is_in_workspace(point, simple_workspace)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_out_of_bounds(self, simple_workspace):
        """Test point outside bounds."""
        point = jnp.array([15.0, 5.0])
        assert not is_in_workspace(point, simple_workspace)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_point_colliding(self, workspace_with_circle_obstacle):
        """Test point in bounds but colliding with obstacle."""
        point = jnp.array([5.0, 5.0])  # Center of circle
        assert not is_in_workspace(point, workspace_with_circle_obstacle)


class TestSampleCollisionFreePoint:
    """Tests for sample_collision_free_point function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_sample_in_empty_workspace(self, simple_workspace, jax_key):
        """Test sampling in workspace with no obstacles."""
        point = sample_collision_free_point(simple_workspace, jax_key)
        assert is_in_workspace(point, simple_workspace)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_sample_with_obstacles(self, workspace_with_circle_obstacle, jax_key):
        """Test sampling in workspace with obstacles."""
        point = sample_collision_free_point(workspace_with_circle_obstacle, jax_key)
        assert is_in_workspace(point, workspace_with_circle_obstacle)
