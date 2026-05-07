"""Performance metrics computation for simulation analysis.

This module provides functions to compute various performance metrics
for evaluating the adversarial interaction between agents.
"""

from typing import Dict, List
import jax.numpy as jnp

from src.shared.trajectory import Trajectory


def compute_observer_accuracy(
    observer_net,  # TrajectoryClassifier
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
    # TODO: Implement
    # goal_probs = observer_net(trajectory.positions)
    # return float(goal_probs[true_goal_id])
    raise NotImplementedError("compute_observer_accuracy not implemented")


def compute_path_length_ratio(
    actual_traj: Trajectory,
    optimal_traj: Trajectory,
) -> float:
    """Compute ratio of actual path length to optimal path length.

    Args:
        actual_traj: Agent's actual (possibly deceptive) trajectory
        optimal_traj: Optimal (straight-line or RRT*) trajectory to same goal

    Returns:
        Path length ratio (>= 1.0, where 1.0 is optimal)
    """
    # TODO: Implement
    # from src.shared.trajectory import compute_path_length
    # actual_length = compute_path_length(actual_traj)
    # optimal_length = compute_path_length(optimal_traj)
    # return float(actual_length / optimal_length)
    raise NotImplementedError("compute_path_length_ratio not implemented")


def compute_belief_entropy_over_time(
    belief_history: List[jnp.ndarray],
) -> jnp.ndarray:
    """Compute Shannon entropy of the belief distribution at each timestep.

    Args:
        belief_history: List of (num_goals,) belief distributions over time

    Returns:
        (T,) array of entropy values, one per timestep
    """
    # TODO: Implement
    # entropies = []
    # for belief in belief_history:
    #     entropy = -jnp.sum(belief * jnp.log(belief + 1e-10))
    #     entropies.append(entropy)
    # return jnp.array(entropies)
    raise NotImplementedError("compute_belief_entropy_over_time not implemented")


def compute_interception_distance(
    trajectory_D: Trajectory,
    trajectory_I: Trajectory,
) -> float:
    """Compute minimum distance between two agent trajectories.

    Args:
        trajectory_D: Deceptive agent's trajectory
        trajectory_I: Interceptor agent's trajectory

    Returns:
        Minimum distance achieved between agents during simulation
    """
    # TODO: Implement
    # T = min(len(trajectory_D.positions), len(trajectory_I.positions))
    # distances = jnp.linalg.norm(
    #     trajectory_D.positions[:T] - trajectory_I.positions[:T], axis=1
    # )
    # return float(jnp.min(distances))
    raise NotImplementedError("compute_interception_distance not implemented")


def compute_goal_inference_accuracy(
    belief_history: List[jnp.ndarray],
    true_goal_id: int,
    threshold: float = 0.5,
) -> float:
    """Compute fraction of timesteps where true goal was most likely.

    Args:
        belief_history: List of belief distributions
        true_goal_id: Index of true goal
        threshold: Probability threshold for "correct" inference

    Returns:
        Fraction of timesteps where true goal had highest belief
    """
    # TODO: Implement
    # correct = sum(
    #     belief[true_goal_id] == belief.max() for belief in belief_history
    # )
    # return correct / len(belief_history)
    raise NotImplementedError("compute_goal_inference_accuracy not implemented")


def compute_time_to_convergence(
    belief_history: List[jnp.ndarray],
    convergence_threshold: float = 0.8,
) -> float:
    """Compute timestep when belief first converges to a single goal.

    Args:
        belief_history: List of belief distributions
        convergence_threshold: Probability threshold for convergence

    Returns:
        Timestep index when belief first exceeds threshold, or -1.0 if never
    """
    # TODO: Implement
    # for t, belief in enumerate(belief_history):
    #     if belief.max() > convergence_threshold:
    #         return float(t)
    # return -1.0
    raise NotImplementedError("compute_time_to_convergence not implemented")


def compute_deception_effectiveness(
    observer_accuracy: float,
    path_length_ratio: float,
    alpha: float,
) -> float:
    """Compute overall deception effectiveness score.

    Combines observer confusion with path efficiency to measure how well the
    deceptive agent balanced concealment and optimality.

    Args:
        observer_accuracy: Probability observer assigned to true goal
        path_length_ratio: Actual path length / optimal path length
        alpha: Deception weight used in planning

    Returns:
        Deception score (higher = more effective)
    """
    # TODO: Implement
    # e.g. alpha * (1 - observer_accuracy) + (1 - alpha) * (1 / path_length_ratio)
    raise NotImplementedError("compute_deception_effectiveness not implemented")


def compute_interception_efficiency(
    interception_distance: float,
    time_to_convergence: float,
    simulation_time: float,
) -> float:
    """Compute interceptor agent's performance score.

    Args:
        interception_distance: Minimum distance achieved between agents
        time_to_convergence: Timestep of goal inference convergence
        simulation_time: Total simulation duration

    Returns:
        Interception efficiency score (higher = better)
    """
    # TODO: Implement
    # e.g. (1 / interception_distance) * (1 - time_to_convergence / simulation_time)
    raise NotImplementedError("compute_interception_efficiency not implemented")
