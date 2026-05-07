"""Particle filter for online goal inference."""

from dataclasses import dataclass
from typing import List, Tuple

import jax
import jax.numpy as jnp

from src.interceptor.belief_update import compute_likelihood


@dataclass
class Particle:
    """A single particle representing a goal hypothesis."""

    goal_id: int
    weight: float


class ParticleFilter:
    """Particle filter for online Bayesian inference of Agent D's goal.

    Maintains a weighted particle set over candidate goals. Weights are
    updated each timestep using the learned reward model as a likelihood proxy.
    """

    def __init__(
        self,
        num_particles: int,
        candidate_goals: jnp.ndarray,
        learned_model,
        key,
    ):
        self.num_particles = num_particles
        self.goals = candidate_goals
        self.model = learned_model

        # Initialise particles uniformly over goals
        num_goals = candidate_goals.shape[0]
        self.particles: List[Particle] = [
            Particle(goal_id=i % num_goals, weight=1.0 / num_particles)
            for i in range(num_particles)
        ]

    def update(self, observation: jnp.ndarray, key) -> None:
        """Update particle weights given a new observation of Agent D's position.

        Args:
            observation: (2,) latest observed position of Agent D
            key: JAX PRNG key (used for resampling)
        """
        for p in self.particles:
            likelihood = self._compute_likelihood(observation, p.goal_id)
            p.weight *= max(likelihood, 1e-30)

        total = sum(p.weight for p in self.particles)
        if total > 1e-30:
            for p in self.particles:
                p.weight /= total
        else:
            # Reset to uniform if all weights vanished
            for p in self.particles:
                p.weight = 1.0 / self.num_particles

        ess = self._effective_sample_size()
        if ess < self.num_particles * 0.5:
            self._resample(key)

    def _compute_likelihood(self, obs: jnp.ndarray, goal_id: int) -> float:
        """Compute P(obs | goal) using the learned reward as a proxy."""
        goal = self.goals[goal_id]
        return compute_likelihood(obs, goal, self.model)

    def _effective_sample_size(self) -> float:
        """Compute ESS = 1 / Σ w²."""
        sq_sum = sum(p.weight**2 for p in self.particles)
        return 1.0 / (sq_sum + 1e-30)

    def _resample(self, key) -> None:
        """Systematic resampling proportional to particle weights."""
        weights = jnp.array([p.weight for p in self.particles])
        indices = jax.random.choice(
            key,
            self.num_particles,
            shape=(self.num_particles,),
            p=weights,
            replace=True,
        )
        new_particles = [
            Particle(
                goal_id=self.particles[int(idx)].goal_id,
                weight=1.0 / self.num_particles,
            )
            for idx in indices
        ]
        self.particles = new_particles

    def estimate_goal(self) -> Tuple[int, float]:
        """Return MAP goal estimate and confidence.

        Returns:
            (goal_id, confidence) where confidence is total weight for MAP goal
        """
        num_goals = self.goals.shape[0]
        counts = jnp.zeros(num_goals)
        for p in self.particles:
            counts = counts.at[p.goal_id].add(p.weight)
        best_id = int(jnp.argmax(counts))
        return best_id, float(counts[best_id])

    def get_belief_distribution(self) -> jnp.ndarray:
        """Return normalised belief distribution over goals.

        Returns:
            (num_goals,) probability array
        """
        num_goals = self.goals.shape[0]
        belief = jnp.zeros(num_goals)
        for p in self.particles:
            belief = belief.at[p.goal_id].add(p.weight)
        total = jnp.sum(belief)
        return jnp.where(total > 1e-10, belief / total, jnp.ones(num_goals) / num_goals)
