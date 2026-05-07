"""Bayesian belief update utilities."""

import jax
import jax.numpy as jnp


@jax.jit
def bayesian_update(
    prior: jnp.ndarray,
    likelihoods: jnp.ndarray,
) -> jnp.ndarray:
    """Bayesian belief update: posterior ∝ likelihood × prior.

    Args:
        prior: (num_goals,) prior belief distribution
        likelihoods: (num_goals,) likelihood P(obs | goal) for each goal

    Returns:
        (num_goals,) normalised posterior distribution
    """
    posterior = prior * likelihoods
    total = jnp.sum(posterior)
    return jnp.where(
        total > 1e-10, posterior / total, jnp.ones_like(posterior) / len(prior)
    )


def compute_likelihood(
    observation: jnp.ndarray,
    goal_hypothesis: jnp.ndarray,
    learned_model,
) -> float:
    """Compute P(observation | goal_hypothesis) via the learned reward model.

    An observation (position) is more likely under a goal hypothesis if the
    action consistent with that hypothesis has high reward.

    Args:
        observation: (2,) observed position of Agent D
        goal_hypothesis: (2,) candidate goal position
        learned_model: LearnedRewardFunction

    Returns:
        Scalar likelihood (> 0)
    """
    direction = goal_hypothesis - observation
    dist = jnp.linalg.norm(direction)
    direction = jnp.where(dist > 1e-6, direction / dist, jnp.array([1.0, 0.0]))
    reward = learned_model(observation, direction * 0.5)
    return float(jnp.exp(jnp.clip(reward, -20.0, 20.0)))
