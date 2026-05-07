"""Deception cost evaluation functions."""

import jax.numpy as jnp


def entropy_based_deception_cost(goal_probs: jnp.ndarray) -> float:
    """Compute negative entropy (higher value = more revealing).

    Negated Shannon entropy: deception_cost = -H(p) = Σ p log p
    Minimizing this cost maximizes observer uncertainty about the true goal.

    Args:
        goal_probs: (num_goals,) probability distribution over goals

    Returns:
        Scalar deception cost
    """
    entropy = -jnp.sum(goal_probs * jnp.log(goal_probs + 1e-8))
    return float(-entropy)


def accuracy_based_deception_cost(
    goal_probs: jnp.ndarray,
    true_goal_id: int,
) -> float:
    """Return probability assigned to the true goal.

    Minimizing this cost reduces the observer's classification probability.

    Args:
        goal_probs: (num_goals,) probability distribution
        true_goal_id: Index of the true goal

    Returns:
        Scalar deception cost equal to observer's confidence on true goal
    """
    return float(goal_probs[true_goal_id])


def evaluate_deception_cost(
    partial_path: jnp.ndarray,
    observer_net,
    true_goal_id: int,
    method: str = "entropy",
) -> float:
    """Evaluate deception cost for a partial path.

    Args:
        partial_path: (N, 2) position array
        observer_net: Trained TrajectoryClassifier
        true_goal_id: Index of true goal (used only for accuracy method)
        method: "entropy" or "accuracy"

    Returns:
        Scalar deception cost
    """
    goal_probs = observer_net(partial_path)

    if method == "entropy":
        return entropy_based_deception_cost(goal_probs)
    elif method == "accuracy":
        return accuracy_based_deception_cost(goal_probs, true_goal_id)
    else:
        raise ValueError(f"Unknown deception cost method: {method!r}")
