"""Workspace representation and utilities for 2D planning environment.

This module provides the Workspace dataclass and related functions for:
- Representing 2D bounded workspaces with obstacles
- Checking point-in-workspace collision
- Creating workspaces from configuration
"""

from dataclasses import dataclass
from typing import List, Union
import jax
import jax.numpy as jnp

from src.shared.collision import point_in_circle, point_in_polygon


@dataclass
class CircleObstacle:
    """Circle obstacle in 2D workspace."""

    center: jnp.ndarray  # (2,) position
    radius: float


@dataclass
class PolygonObstacle:
    """Polygon obstacle in 2D workspace."""

    vertices: jnp.ndarray  # (N, 2) vertices in counter-clockwise order


Obstacle = Union[CircleObstacle, PolygonObstacle]


@dataclass
class Workspace:
    """2D workspace with bounds and obstacles.

    Attributes:
        bounds: (2, 2) array [[x_min, x_max], [y_min, y_max]]
        obstacles: List of CircleObstacle and/or PolygonObstacle
    """

    bounds: jnp.ndarray  # (2, 2)
    obstacles: List[Obstacle]


def create_workspace(bounds: jnp.ndarray, obstacles: List[Obstacle]) -> Workspace:
    """Create a Workspace from bounds and obstacles.

    Args:
        bounds: (2, 2) array [[x_min, x_max], [y_min, y_max]]
        obstacles: List of obstacle objects

    Returns:
        Workspace object

    Example:
        >>> bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
        >>> obs = [CircleObstacle(center=jnp.array([5.0, 5.0]), radius=1.0)]
        >>> ws = create_workspace(bounds, obs)
    """
    # TODO: Add validation:
    #   - bounds shape is (2, 2)
    #   - x_min < x_max, y_min < y_max
    #   - obstacles have valid shapes
    return Workspace(bounds=bounds, obstacles=obstacles)


def is_in_bounds(point: jnp.ndarray, bounds: jnp.ndarray) -> bool:
    """Check if a point is within workspace bounds.

    Args:
        point: (2,) array [x, y]
        bounds: (2, 2) array [[x_min, x_max], [y_min, y_max]]

    Returns:
        True if point is within bounds, False otherwise
    """
    x_in_bounds = (bounds[0, 0] <= point[0]) & (point[0] <= bounds[0, 1])
    y_in_bounds = (bounds[1, 0] <= point[1]) & (point[1] <= bounds[1, 1])
    return x_in_bounds & y_in_bounds  # type: ignore[return-value]


def is_collision_free(point: jnp.ndarray, obstacles: List[Obstacle]) -> bool:
    """Check if a point is collision-free with respect to all obstacles.

    Args:
        point: (2,) array [x, y]
        obstacles: List of obstacle objects

    Returns:
        True if point collides with no obstacles, False otherwise
    """
    for obstacle in obstacles:
        if isinstance(obstacle, CircleObstacle):
            if point_in_circle(point, obstacle.center, obstacle.radius):
                return False
        elif isinstance(obstacle, PolygonObstacle):
            if point_in_polygon(point, obstacle.vertices):
                return False
    return True


def is_in_workspace(point: jnp.ndarray, workspace: Workspace) -> bool:
    """Check if a point is valid (in bounds and collision-free).

    Args:
        point: (2,) array [x, y]
        workspace: Workspace object

    Returns:
        True if point is in bounds and collision-free, False otherwise
    """
    return is_in_bounds(point, workspace.bounds) and is_collision_free(
        point, workspace.obstacles
    )


def sample_collision_free_point(workspace: Workspace, key: jnp.ndarray) -> jnp.ndarray:
    """Sample a random collision-free point in the workspace using rejection sampling.

    Args:
        workspace: Workspace object
        key: JAX PRNG key

    Returns:
        (2,) array representing a valid point

    Note:
        Uses a Python-level rejection loop (not JIT-compiled). Tries up to 10 000
        candidates before falling back to the workspace centre.
    """
    bounds = workspace.bounds
    for _ in range(10_000):
        key, x_key, y_key = jax.random.split(key, 3)
        x = jax.random.uniform(x_key, minval=bounds[0, 0], maxval=bounds[0, 1])
        y = jax.random.uniform(y_key, minval=bounds[1, 0], maxval=bounds[1, 1])
        point = jnp.array([x, y])
        if is_in_workspace(point, workspace):
            return point
    # Fallback: workspace centre (always collision-free in reasonable configs)
    cx = (float(bounds[0, 0]) + float(bounds[0, 1])) / 2.0
    cy = (float(bounds[1, 0]) + float(bounds[1, 1])) / 2.0
    return jnp.array([cx, cy])
