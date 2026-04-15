"""Sample workspace generators for testing."""

import jax.numpy as jnp
from src.shared.workspace import Workspace, CircleObstacle, PolygonObstacle


def empty_workspace() -> Workspace:
    """Empty 10x10 workspace with no obstacles."""
    bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
    return Workspace(bounds=bounds, obstacles=[])


def sparse_obstacle_workspace() -> Workspace:
    """Workspace with 2-3 well-separated obstacles."""
    bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
    obstacles = [
        CircleObstacle(
            center=jnp.array([2.0, 2.0]),
            radius=0.5
        ),
        CircleObstacle(
            center=jnp.array([8.0, 8.0]),
            radius=0.7
        ),
        PolygonObstacle(
            vertices=jnp.array([
                [5.0, 1.0],
                [6.0, 1.0],
                [5.5, 2.0]
            ])
        )
    ]
    return Workspace(bounds=bounds, obstacles=obstacles)


def dense_obstacle_workspace() -> Workspace:
    """Workspace with many obstacles creating a maze-like environment."""
    bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
    obstacles = []

    # Create a grid of circular obstacles
    for x in range(2, 9, 2):
        for y in range(2, 9, 2):
            obstacles.append(CircleObstacle(
                center=jnp.array([float(x), float(y)]),
                radius=0.4
            ))

    # Add some polygon obstacles
    obstacles.append(PolygonObstacle(
        vertices=jnp.array([
            [0.5, 4.5],
            [1.5, 4.5],
            [1.5, 5.5],
            [0.5, 5.5]
        ])
    ))

    obstacles.append(PolygonObstacle(
        vertices=jnp.array([
            [8.5, 4.5],
            [9.5, 4.5],
            [9.5, 5.5],
            [8.5, 5.5]
        ])
    ))

    return Workspace(bounds=bounds, obstacles=obstacles)


def corridor_workspace() -> Workspace:
    """Narrow corridor workspace for testing path planning."""
    bounds = jnp.array([[0.0, 20.0], [0.0, 5.0]])
    obstacles = [
        # Top wall obstacles
        PolygonObstacle(vertices=jnp.array([[0.0, 3.5], [8.0, 3.5], [8.0, 5.0], [0.0, 5.0]])),
        PolygonObstacle(vertices=jnp.array([[12.0, 3.5], [20.0, 3.5], [20.0, 5.0], [12.0, 5.0]])),

        # Bottom wall obstacles
        PolygonObstacle(vertices=jnp.array([[0.0, 0.0], [8.0, 0.0], [8.0, 1.5], [0.0, 1.5]])),
        PolygonObstacle(vertices=jnp.array([[12.0, 0.0], [20.0, 0.0], [20.0, 1.5], [12.0, 1.5]])),
    ]
    return Workspace(bounds=bounds, obstacles=obstacles)


def single_circle_workspace() -> Workspace:
    """Simple workspace with one circular obstacle in the center."""
    bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
    obstacle = CircleObstacle(
        center=jnp.array([5.0, 5.0]),
        radius=1.5
    )
    return Workspace(bounds=bounds, obstacles=[obstacle])


def single_polygon_workspace() -> Workspace:
    """Simple workspace with one square polygon obstacle."""
    bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
    obstacle = PolygonObstacle(
        vertices=jnp.array([
            [4.0, 4.0],
            [6.0, 4.0],
            [6.0, 6.0],
            [4.0, 6.0]
        ])
    )
    return Workspace(bounds=bounds, obstacles=[obstacle])
