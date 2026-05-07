"""Adversarial RRT* planner for deceptive motion planning."""

from typing import List, Tuple

import jax
import jax.numpy as jnp

from src.deceptive.deception_cost import evaluate_deception_cost
from src.deceptive.tree import RRTTree
from src.shared.collision import segment_circle_collision, segment_polygon_collision
from src.shared.trajectory import Trajectory, create_trajectory
from src.shared.workspace import CircleObstacle, PolygonObstacle, Workspace
from src.simulation.config import PlannerConfig


def adversarial_rrt_star(
    start: jnp.ndarray,
    goal: jnp.ndarray,
    workspace: Workspace,
    observer_net,
    true_goal_id: int,
    alpha: float,
    config: PlannerConfig,
    key,
    deception_method: str = "entropy",
) -> Trajectory:
    """Plan a deceptive trajectory using Adversarial RRT*.

    cost(path) = α · path_length + (1 - α) · deception_cost

    Args:
        start: (2,) starting position
        goal: (2,) goal position
        workspace: Environment with bounds and obstacles
        observer_net: Trained TrajectoryClassifier
        true_goal_id: Index of true goal in candidate goals list
        alpha: Deception weight ∈ [0, 1]. alpha=1 → pure path optimisation;
               alpha=0 → pure deception.
        config: RRT* hyperparameters
        key: JAX PRNG key
        deception_method: "entropy" or "accuracy"

    Returns:
        Trajectory from start to (near) goal
    """
    tree = RRTTree()
    tree.add_node(start, parent_id=None, cost=0.0)

    for _ in range(config.max_iterations):
        key, sample_key = jax.random.split(key)
        x_sample = _sample_configuration(workspace, goal, config, sample_key)

        nearest_id = tree.find_nearest(x_sample)
        x_nearest = tree.nodes[nearest_id].position

        x_new = _steer(x_nearest, x_sample, config.step_size)

        if not _segment_collides(x_nearest, x_new, workspace):
            radius = _rewiring_radius(len(tree.nodes), config)
            near_ids = tree.find_near(x_new, radius)
            if not near_ids:
                near_ids = [nearest_id]

            parent_id, min_cost = _choose_parent(
                tree,
                x_new,
                near_ids,
                workspace,
                goal,
                observer_net,
                alpha,
                true_goal_id,
                deception_method,
            )

            new_id = tree.add_node(x_new, parent_id, min_cost)

            _rewire(
                tree,
                new_id,
                near_ids,
                workspace,
                goal,
                observer_net,
                alpha,
                true_goal_id,
                deception_method,
            )

            if float(jnp.linalg.norm(x_new - goal)) < config.goal_radius:
                path = tree.extract_path(new_id)
                return _path_to_trajectory(path)

    best_id = _find_closest_to_goal(tree, goal)
    path = tree.extract_path(best_id)
    return _path_to_trajectory(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_configuration(
    workspace: Workspace,
    goal: jnp.ndarray,
    config: PlannerConfig,
    key,
) -> jnp.ndarray:
    """Goal-biased configuration sampling."""
    key, bias_key, sample_key = jax.random.split(key, 3)
    if float(jax.random.uniform(bias_key)) < config.goal_bias_probability:
        noise_key, _ = jax.random.split(key)
        return goal + jax.random.normal(noise_key, shape=(2,)) * 0.5
    else:
        bounds = workspace.bounds
        x = jax.random.uniform(sample_key, minval=bounds[0, 0], maxval=bounds[0, 1])
        key, y_key = jax.random.split(key)
        y = jax.random.uniform(y_key, minval=bounds[1, 0], maxval=bounds[1, 1])
        return jnp.array([x, y])


def _steer(x_from: jnp.ndarray, x_to: jnp.ndarray, step_size: float) -> jnp.ndarray:
    """Steer from x_from toward x_to by at most step_size."""
    direction = x_to - x_from
    dist = float(jnp.linalg.norm(direction))
    if dist <= step_size or dist < 1e-8:
        return x_to
    return x_from + (direction / dist) * step_size


def _rewiring_radius(num_nodes: int, config: PlannerConfig) -> float:
    """Compute shrinking rewiring radius: min(γ · (log(n)/n)^0.5, η)."""
    if num_nodes <= 1:
        return config.max_radius
    import math

    r = config.gamma * math.sqrt(math.log(num_nodes) / num_nodes)
    return min(r, config.max_radius)


def _segment_collides(p1: jnp.ndarray, p2: jnp.ndarray, workspace: Workspace) -> bool:
    """Return True if the segment p1→p2 intersects any obstacle."""
    for obs in workspace.obstacles:
        if isinstance(obs, CircleObstacle):
            if segment_circle_collision(p1, p2, obs.center, obs.radius):
                return True
        elif isinstance(obs, PolygonObstacle):
            if segment_polygon_collision(p1, p2, obs.vertices):
                return True
    return False


_MAX_OBS_LEN = (
    20  # Limit observer input length to avoid JAX recompilation on growing paths
)


def _combined_cost(
    path_positions: jnp.ndarray,
    goal: jnp.ndarray,
    observer_net,
    alpha: float,
    true_goal_id: int,
    method: str,
) -> float:
    """α · path_length + (1 - α) · deception_cost."""
    j_path = _path_length(path_positions)
    if alpha >= 1.0 or observer_net is None:
        return float(j_path)
    # Truncate to last _MAX_OBS_LEN positions so the observer input has a
    # fixed or bounded shape, preventing JAX from recompiling on every call.
    obs_input = path_positions[-_MAX_OBS_LEN:]
    j_deception = float(
        evaluate_deception_cost(obs_input, observer_net, true_goal_id, method)
    )
    return float(alpha * j_path + (1.0 - alpha) * j_deception)


def _path_length(positions: jnp.ndarray) -> float:
    """Sum of Euclidean segment lengths for an (N, 2) position array."""
    if positions.shape[0] < 2:
        return 0.0
    diffs = jnp.diff(positions, axis=0)
    return float(jnp.sum(jnp.linalg.norm(diffs, axis=1)))


def _choose_parent(
    tree: RRTTree,
    x_new: jnp.ndarray,
    near_ids: List[int],
    workspace: Workspace,
    goal: jnp.ndarray,
    observer_net,
    alpha: float,
    true_goal_id: int,
    method: str,
) -> Tuple[int, float]:
    """Choose parent for x_new that minimises combined cost."""
    best_parent = near_ids[0]
    best_cost = float("inf")

    for near_id in near_ids:
        x_near = tree.nodes[near_id].position
        if _segment_collides(x_near, x_new, workspace):
            continue
        path_to_near = tree.extract_path(near_id)
        candidate = jnp.vstack([path_to_near, x_new[None, :]])
        cost = _combined_cost(
            candidate, goal, observer_net, alpha, true_goal_id, method
        )
        if cost < best_cost:
            best_cost = cost
            best_parent = near_id

    if best_cost == float("inf"):
        # All near nodes blocked by collision — fall back to nearest
        best_parent = near_ids[0]
        path_to_parent = tree.extract_path(best_parent)
        candidate = jnp.vstack([path_to_parent, x_new[None, :]])
        best_cost = _combined_cost(
            candidate, goal, observer_net, alpha, true_goal_id, method
        )

    return best_parent, best_cost


def _rewire(
    tree: RRTTree,
    new_id: int,
    near_ids: List[int],
    workspace: Workspace,
    goal: jnp.ndarray,
    observer_net,
    alpha: float,
    true_goal_id: int,
    method: str,
) -> None:
    """Rewire near nodes through new_id if it reduces their cost.

    Uses path length only for the rewiring decision (avoids repeated observer
    calls on the same path suffix, which would dominate runtime).
    """
    x_new = tree.nodes[new_id].position
    path_to_new = tree.extract_path(new_id)
    length_to_new = _path_length(path_to_new)

    for near_id in near_ids:
        if near_id == tree.nodes[new_id].parent_id:
            continue
        x_near = tree.nodes[near_id].position
        if _segment_collides(x_new, x_near, workspace):
            continue
        # Use path-length cost for rewiring to avoid O(near_ids) observer calls
        segment_len = float(jnp.linalg.norm(x_near - x_new))
        new_cost = length_to_new + segment_len
        if new_cost < tree.nodes[near_id].cost:
            tree.update_node(near_id, new_id, new_cost)


def _find_closest_to_goal(tree: RRTTree, goal: jnp.ndarray) -> int:
    """Return ID of node closest to goal."""
    best_id = 0
    best_dist = float("inf")
    for i, node in enumerate(tree.nodes):
        d = float(jnp.linalg.norm(node.position - goal))
        if d < best_dist:
            best_dist = d
            best_id = i
    return best_id


def _path_to_trajectory(path_positions: jnp.ndarray, dt: float = 0.1) -> Trajectory:
    """Convert (N, 2) waypoints to a Trajectory with uniform time spacing."""
    n = path_positions.shape[0]
    times = jnp.arange(n, dtype=jnp.float32) * dt
    return create_trajectory(times, path_positions)
