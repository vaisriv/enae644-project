"""Data schemas for training datasets."""

from dataclasses import dataclass
from typing import List

import jax.numpy as jnp


@dataclass
class TrajectoryDataset:
    """Training dataset for observer and IRL.

    Attributes:
        trajectories: List of (T_i, 2) position arrays, variable-length per sample
        goal_ids: Integer goal label for each trajectory
        goals: (num_goals, 2) goal positions
    """

    trajectories: List[jnp.ndarray]
    goal_ids: List[int]
    goals: jnp.ndarray
