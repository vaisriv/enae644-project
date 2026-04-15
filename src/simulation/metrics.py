"""Performance metrics computation for simulation analysis.

This module provides functions to compute various performance metrics
for evaluating the adversarial interaction between agents.
"""

from typing import Dict, List
import jax.numpy as jnp

from src.shared.trajectory import Trajectory


def compute_all_metrics(
    trajectory_D: Trajectory,
    trajectory_I: Trajectory,
    belief_history: List[jnp.ndarray],
    observer_net,  # TrajectoryClassifier
    true_goal: jnp.ndarray,
) -> Dict[str, float]:
    """Compute all performance metrics for a simulation run.

    Args:
        trajectory_D: Deceptive agent's trajectory
        trajectory_I: Interceptor agent's trajectory
        belief_history: List of belief distributions over time
        observer_net: Trained observer network for classification
        true_goal: True goal position (2,)

    Returns:
        Dictionary containing all computed metrics
    """
    metrics = {}

    # TODO: Implement all metric computations
    # metrics['observer_accuracy_final'] = compute_observer_accuracy(
    #     trajectory_D, observer_net, true_goal
    # )
    # metrics['path_length_ratio'] = compute_path_length_ratio(trajectory_D)
    # metrics['belief_entropy_final'] = compute_belief_entropy(belief_history[-1])
    # metrics['belief_entropy_mean'] = compute_mean_belief_entropy(belief_history)
    # metrics['interception_distance_min'] = compute_interception_distance(
    #     trajectory_D, trajectory_I
    # )

    raise NotImplementedError("compute_all_metrics not implemented")


def compute_observer_accuracy(
    trajectory: Trajectory, observer_net, true_goal: jnp.ndarray, goal_id: int
) -> float:
    """Compute observer network's classification accuracy.

    Args:
        trajectory: Agent's trajectory
        observer_net: Trained observer network
        true_goal: True goal position
        goal_id: Index of true goal in candidate goals list

    Returns:
        Probability assigned to true goal by observer
    """
    # TODO: Implement
    # Run observer network on full trajectory
    # goal_probs = observer_net(trajectory.positions)
    # return float(goal_probs[goal_id])
    raise NotImplementedError("compute_observer_accuracy not implemented")


def compute_path_length_ratio(trajectory: Trajectory) -> float:
    """Compute ratio of actual path length to optimal (straight-line) path length.

    Args:
        trajectory: Agent's trajectory

    Returns:
        Path length ratio (>= 1.0, where 1.0 is optimal)
    """
    # TODO: Implement
    # actual_length = compute_path_length(trajectory)
    # optimal_length = euclidean_distance(
    #     trajectory.positions[0],
    #     trajectory.positions[-1]
    # )
    # return actual_length / optimal_length
    raise NotImplementedError("compute_path_length_ratio not implemented")


def compute_belief_entropy(belief: jnp.ndarray) -> float:
    """Compute Shannon entropy of a belief distribution.

    Args:
        belief: (num_goals,) probability distribution

    Returns:
        Entropy in nats: -sum(p * log(p))

    Note:
        Higher entropy = more uncertain/deceptive
        Lower entropy = more confident
    """
    # TODO: Implement
    # Add small epsilon to avoid log(0)
    # entropy = -jnp.sum(belief * jnp.log(belief + 1e-10))
    # return float(entropy)
    raise NotImplementedError("compute_belief_entropy not implemented")


def compute_mean_belief_entropy(belief_history: List[jnp.ndarray]) -> float:
    """Compute mean entropy across all timesteps.

    Args:
        belief_history: List of belief distributions

    Returns:
        Mean entropy over time
    """
    # TODO: Implement
    # entropies = [compute_belief_entropy(belief) for belief in belief_history]
    # return float(jnp.mean(jnp.array(entropies)))
    raise NotImplementedError("compute_mean_belief_entropy not implemented")


def compute_interception_distance(
    trajectory_D: Trajectory, trajectory_I: Trajectory
) -> float:
    """Compute minimum distance between two trajectories.

    Args:
        trajectory_D: Deceptive agent's trajectory
        trajectory_I: Interceptor agent's trajectory

    Returns:
        Minimum distance achieved during simulation
    """
    # TODO: Implement
    # For each timestep, compute distance between agents
    # min_dist = jnp.inf
    # T = min(len(trajectory_D.positions), len(trajectory_I.positions))
    # for i in range(T):
    #     dist = jnp.linalg.norm(trajectory_D.positions[i] - trajectory_I.positions[i])
    #     min_dist = min(min_dist, dist)
    # return float(min_dist)
    raise NotImplementedError("compute_interception_distance not implemented")


def compute_goal_inference_accuracy(
    belief_history: List[jnp.ndarray], true_goal_id: int, threshold: float = 0.5
) -> float:
    """Compute fraction of time true goal was most likely.

    Args:
        belief_history: List of belief distributions
        true_goal_id: Index of true goal
        threshold: Probability threshold for "correct" inference

    Returns:
        Fraction of timesteps where true goal was most likely
    """
    # TODO: Implement
    # correct_inferences = sum(
    #     belief[true_goal_id] == belief.max() for belief in belief_history
    # )
    # return correct_inferences / len(belief_history)
    raise NotImplementedError("compute_goal_inference_accuracy not implemented")


def compute_time_to_convergence(
    belief_history: List[jnp.ndarray], convergence_threshold: float = 0.8
) -> float:
    """Compute time until belief converges to a single goal.

    Args:
        belief_history: List of belief distributions
        convergence_threshold: Probability threshold for convergence

    Returns:
        Timestep index when belief first exceeds threshold, or -1 if never
    """
    # TODO: Implement
    # for t, belief in enumerate(belief_history):
    #     if belief.max() > convergence_threshold:
    #         return float(t)
    # return -1.0
    raise NotImplementedError("compute_time_to_convergence not implemented")


def compute_deception_effectiveness(
    observer_accuracy: float, path_length_ratio: float, alpha: float
) -> float:
    """Compute overall deception effectiveness score.

    This combines observer confusion with path efficiency to measure
    how well the deceptive agent balanced concealment and optimality.

    Args:
        observer_accuracy: Probability observer assigned to true goal
        path_length_ratio: Actual path length / optimal path length
        alpha: Deception weight used in planning

    Returns:
        Deception score (higher = more effective)
    """
    # TODO: Implement
    # Lower observer accuracy = better deception
    # Lower path ratio = better efficiency
    # Score could be: (1 - observer_accuracy) / path_length_ratio
    # Or: alpha * (1 - observer_accuracy) + (1 - alpha) * (1 / path_length_ratio)
    raise NotImplementedError("compute_deception_effectiveness not implemented")


def compute_interception_efficiency(
    interception_distance: float, time_to_convergence: float, simulation_time: float
) -> float:
    """Compute interceptor agent's performance score.

    Args:
        interception_distance: Minimum distance achieved
        time_to_convergence: Time to goal inference convergence
        simulation_time: Total simulation duration

    Returns:
        Interception efficiency score (higher = better)
    """
    # TODO: Implement
    # Closer distance = better
    # Faster convergence = better
    # Score could be: (1 / interception_distance) * (1 - time_to_convergence / simulation_time)
    raise NotImplementedError("compute_interception_efficiency not implemented")
