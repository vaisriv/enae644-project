"""Training data generation for observer and IRL models."""

from typing import List

import jax
import jax.numpy as jnp

from src.data.schemas import TrajectoryDataset
from src.deceptive.tree import RRTTree
from src.shared.collision import segment_circle_collision, segment_polygon_collision
from src.shared.trajectory import Trajectory, create_trajectory
from src.shared.workspace import CircleObstacle, PolygonObstacle, Workspace


def generate_optimal_trajectories(
    workspace: Workspace,
    goals: jnp.ndarray,
    samples_per_goal: int,
    key,
) -> TrajectoryDataset:
    """Generate non-deceptive trajectories toward each goal for observer training.

    For each goal, samples `samples_per_goal` random starting positions within the
    workspace and runs a simple RRT to that goal. The resulting (N, 2) position
    arrays are labelled with the goal index.

    Args:
        workspace: Environment with bounds and obstacles
        goals: (num_goals, 2) goal positions
        samples_per_goal: Number of trajectories per goal
        key: JAX PRNG key

    Returns:
        TrajectoryDataset with variable-length trajectories and goal labels
    """
    trajectories: List[jnp.ndarray] = []
    goal_ids: List[int] = []
    num_goals = goals.shape[0]

    for goal_id in range(num_goals):
        goal = goals[goal_id]
        for _ in range(samples_per_goal):
            key, start_key, rrt_key = jax.random.split(key, 3)
            start = _random_start(workspace, goal, start_key)
            path = _simple_rrt(start, goal, workspace, rrt_key, max_iterations=500)
            trajectories.append(path)
            goal_ids.append(goal_id)

    return TrajectoryDataset(
        trajectories=trajectories,
        goal_ids=goal_ids,
        goals=goals,
    )


def generate_irl_demonstrations(
    workspace: Workspace,
    goals: jnp.ndarray,
    num_demonstrations: int,
    key,
) -> List[Trajectory]:
    """Generate expert (goal-directed) demonstrations for IRL training.

    Args:
        workspace: Environment
        goals: (num_goals, 2) candidate goal positions
        num_demonstrations: Total number of demonstrations to generate
        key: JAX PRNG key

    Returns:
        List of Trajectory objects (paths toward randomly-selected goals)
    """
    num_goals = goals.shape[0]
    demonstrations: List[Trajectory] = []

    for i in range(num_demonstrations):
        goal_id = i % num_goals
        goal = goals[goal_id]
        key, start_key, rrt_key = jax.random.split(key, 3)
        start = _random_start(workspace, goal, start_key)
        path_positions = _simple_rrt(
            start, goal, workspace, rrt_key, max_iterations=500
        )
        n = path_positions.shape[0]
        times = jnp.arange(n, dtype=jnp.float32) * 0.1
        traj = create_trajectory(times, path_positions)
        demonstrations.append(traj)

    return demonstrations


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _random_start(workspace: Workspace, goal: jnp.ndarray, key) -> jnp.ndarray:
    """Sample a random start that is in bounds, collision-free, and not at goal."""
    bounds = workspace.bounds
    for _ in range(200):
        key, kx, ky = jax.random.split(key, 3)
        x = float(
            jax.random.uniform(
                kx, minval=float(bounds[0, 0]), maxval=float(bounds[0, 1])
            )
        )
        y = float(
            jax.random.uniform(
                ky, minval=float(bounds[1, 0]), maxval=float(bounds[1, 1])
            )
        )
        p = jnp.array([x, y])
        if _point_free(p, workspace) and float(jnp.linalg.norm(p - goal)) > 1.0:
            return p
    # Fallback: return a corner point
    return jnp.array([bounds[0, 0] + 0.5, bounds[1, 0] + 0.5])


def _point_free(p: jnp.ndarray, workspace: Workspace) -> bool:
    """Check bounds and obstacles."""
    bounds = workspace.bounds
    if not (
        bounds[0, 0] <= p[0] <= bounds[0, 1] and bounds[1, 0] <= p[1] <= bounds[1, 1]
    ):
        return False
    for obs in workspace.obstacles:
        if isinstance(obs, CircleObstacle):
            if float(jnp.linalg.norm(p - obs.center)) <= obs.radius:
                return False
        elif isinstance(obs, PolygonObstacle):
            from src.shared.collision import point_in_polygon

            if point_in_polygon(p, obs.vertices):
                return False
    return True


def _segment_free(p1: jnp.ndarray, p2: jnp.ndarray, workspace: Workspace) -> bool:
    """Return True if segment p1→p2 is free of obstacles."""
    for obs in workspace.obstacles:
        if isinstance(obs, CircleObstacle):
            if segment_circle_collision(p1, p2, obs.center, obs.radius):
                return False
        elif isinstance(obs, PolygonObstacle):
            if segment_polygon_collision(p1, p2, obs.vertices):
                return False
    return True


def _simple_rrt(
    start: jnp.ndarray,
    goal: jnp.ndarray,
    workspace: Workspace,
    key,
    max_iterations: int = 500,
    step_size: float = 0.5,
    goal_radius: float = 0.5,
    goal_bias: float = 0.15,
) -> jnp.ndarray:
    """Simple RRT (no deception) returning (N, 2) waypoints from start to goal.

    Falls back to a straight-line path if no valid path is found within
    max_iterations.
    """
    tree = RRTTree()
    tree.add_node(start, parent_id=None, cost=0.0)

    bounds = workspace.bounds

    for _ in range(max_iterations):
        key, bias_key, sx_key, sy_key = jax.random.split(key, 4)

        if float(jax.random.uniform(bias_key)) < goal_bias:
            sample = goal
        else:
            x = float(
                jax.random.uniform(
                    sx_key, minval=float(bounds[0, 0]), maxval=float(bounds[0, 1])
                )
            )
            y = float(
                jax.random.uniform(
                    sy_key, minval=float(bounds[1, 0]), maxval=float(bounds[1, 1])
                )
            )
            sample = jnp.array([x, y])

        nearest_id = tree.find_nearest(sample)
        x_nearest = tree.nodes[nearest_id].position

        direction = sample - x_nearest
        dist = float(jnp.linalg.norm(direction))
        if dist < 1e-8:
            continue
        x_new = x_nearest + (direction / dist) * min(step_size, dist)

        if not _segment_free(x_nearest, x_new, workspace):
            continue

        new_id = tree.add_node(x_new, parent_id=nearest_id, cost=0.0)

        if float(jnp.linalg.norm(x_new - goal)) < goal_radius:
            return tree.extract_path(new_id)

    # Return best found path
    best_id = 0
    best_dist = float("inf")
    for i, node in enumerate(tree.nodes):
        d = float(jnp.linalg.norm(node.position - goal))
        if d < best_dist:
            best_dist = d
            best_id = i
    return tree.extract_path(best_id)
