"""Performance metrics computation for simulation analysis."""

from typing import List

import jax.numpy as jnp

from src.shared.trajectory import Trajectory, compute_path_length


def compute_observer_accuracy(
    observer_net,
    trajectory: Trajectory,
    true_goal_id: int,
) -> float:
    """Compute observer network's classification accuracy on a trajectory.

    Args:
        observer_net: Trained TrajectoryClassifier
        trajectory: Agent's trajectory
        true_goal_id: Index of true goal in candidate goals list

    Returns:
        Probability assigned to true goal by observer
    """
    goal_probs = observer_net(trajectory.positions)
    return float(goal_probs[true_goal_id])


def compute_path_length_ratio(
    actual_traj: Trajectory,
    optimal_traj: Trajectory,
) -> float:
    """Compute ratio of actual path length to optimal path length.

    Args:
        actual_traj: Agent's actual (possibly deceptive) trajectory
        optimal_traj: Optimal trajectory to same goal

    Returns:
        Path length ratio (>= 1.0, where 1.0 is optimal)
    """
    actual_length = float(compute_path_length(actual_traj))
    optimal_length = float(compute_path_length(optimal_traj))
    if optimal_length < 1e-8:
        return 1.0
    return actual_length / optimal_length


def compute_belief_entropy_over_time(
    belief_history: List[jnp.ndarray],
) -> jnp.ndarray:
    """Compute Shannon entropy of the belief distribution at each timestep.

    Args:
        belief_history: List of (num_goals,) belief distributions

    Returns:
        (T,) array of entropy values
    """
    entropies = []
    for belief in belief_history:
        entropy = -jnp.sum(belief * jnp.log(belief + 1e-10))
        entropies.append(float(entropy))
    return jnp.array(entropies)


def compute_interception_distance(
    trajectory_D: Trajectory,
    trajectory_I: Trajectory,
) -> float:
    """Compute minimum distance between the two agent trajectories.

    Args:
        trajectory_D: Deceptive agent's trajectory
        trajectory_I: Interceptor agent's trajectory

    Returns:
        Minimum distance achieved between agents
    """
    T = min(trajectory_D.positions.shape[0], trajectory_I.positions.shape[0])
    distances = jnp.linalg.norm(
        trajectory_D.positions[:T] - trajectory_I.positions[:T], axis=1
    )
    return float(jnp.min(distances))


def compute_goal_inference_accuracy(
    belief_history: List[jnp.ndarray],
    true_goal_id: int,
    threshold: float = 0.5,
) -> float:
    """Compute fraction of timesteps where true goal was the MAP estimate.

    Args:
        belief_history: List of belief distributions
        true_goal_id: Index of true goal
        threshold: Unused; kept for API compatibility

    Returns:
        Fraction of timesteps where argmax(belief) == true_goal_id
    """
    if not belief_history:
        return 0.0
    correct = sum(
        1 for belief in belief_history if int(jnp.argmax(belief)) == true_goal_id
    )
    return correct / len(belief_history)


def compute_time_to_convergence(
    belief_history: List[jnp.ndarray],
    convergence_threshold: float = 0.8,
) -> float:
    """Compute timestep when belief first converges to a single goal.

    Args:
        belief_history: List of belief distributions
        convergence_threshold: Minimum probability for convergence

    Returns:
        Timestep index of convergence, or -1.0 if belief never converged
    """
    for t, belief in enumerate(belief_history):
        if float(jnp.max(belief)) > convergence_threshold:
            return float(t)
    return -1.0


def compute_deception_effectiveness(
    observer_accuracy: float,
    path_length_ratio: float,
    alpha: float,
) -> float:
    """Compute overall deception effectiveness score.

    Score = α · (1 - observer_accuracy) + (1 - α) · (1 / path_length_ratio)

    Higher is better for Agent D.

    Args:
        observer_accuracy: Probability observer assigned to true goal
        path_length_ratio: Actual / optimal path length
        alpha: Deception weight used in planning

    Returns:
        Deception effectiveness score ∈ [0, 1]
    """
    confusion = 1.0 - observer_accuracy
    efficiency = 1.0 / max(path_length_ratio, 1e-3)
    return float(alpha * confusion + (1.0 - alpha) * efficiency)


def compute_interception_efficiency(
    interception_distance: float,
    time_to_convergence: float,
    simulation_time: float,
) -> float:
    """Compute interceptor agent's performance score.

    Args:
        interception_distance: Minimum distance achieved between agents
        time_to_convergence: Timestep of goal inference convergence (-1 if never)
        simulation_time: Total simulation duration

    Returns:
        Interception efficiency score (higher = better for Agent I)
    """
    closeness = 1.0 / (1.0 + interception_distance)
    ttc = time_to_convergence if time_to_convergence >= 0 else simulation_time
    timeliness = 1.0 - min(1.0, ttc / max(simulation_time, 1.0))
    return float(closeness * (1.0 + timeliness) / 2.0)
