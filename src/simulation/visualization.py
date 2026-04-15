"""Visualization utilities for simulation results.

This module provides functions for creating plots and saving data outputs
for analysis and presentation.
"""

from typing import List, Dict, Optional
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import jax.numpy as jnp

from src.shared.workspace import Workspace, CircleObstacle, PolygonObstacle
from src.shared.trajectory import Trajectory


def plot_trajectories(
    workspace: Workspace,
    trajectory_D: Trajectory,
    trajectory_I: Trajectory,
    candidate_goals: List[jnp.ndarray],
    true_goal: jnp.ndarray,
    save_path: Optional[str] = None
):
    """Plot workspace with agent trajectories and obstacles.

    Args:
        workspace: Workspace with bounds and obstacles
        trajectory_D: Deceptive agent's trajectory
        trajectory_I: Interceptor agent's trajectory
        candidate_goals: List of candidate goal positions
        true_goal: True goal position (highlighted differently)
        save_path: Optional path to save figure (PNG/PDF)
    """
    # TODO: Implement visualization
    # 1. Create figure and axis
    # fig, ax = plt.subplots(figsize=(10, 10))

    # 2. Plot workspace bounds
    # bounds = workspace.bounds
    # ax.set_xlim(bounds[0, 0], bounds[0, 1])
    # ax.set_ylim(bounds[1, 0], bounds[1, 1])

    # 3. Plot obstacles
    # for obstacle in workspace.obstacles:
    #     if isinstance(obstacle, CircleObstacle):
    #         circle = patches.Circle(obstacle.center, obstacle.radius, ...)
    #         ax.add_patch(circle)
    #     elif isinstance(obstacle, PolygonObstacle):
    #         polygon = patches.Polygon(obstacle.vertices, ...)
    #         ax.add_patch(polygon)

    # 4. Plot trajectories
    # ax.plot(trajectory_D.positions[:, 0], trajectory_D.positions[:, 1], ...)
    # ax.plot(trajectory_I.positions[:, 0], trajectory_I.positions[:, 1], ...)

    # 5. Plot goals
    # for goal in candidate_goals:
    #     ax.scatter(goal[0], goal[1], marker='x', ...)
    # ax.scatter(true_goal[0], true_goal[1], marker='*', ...)

    # 6. Add legend and labels
    # ax.legend()
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")

    # 7. Save or show
    # if save_path:
    #     plt.savefig(save_path)
    # else:
    #     plt.show()

    raise NotImplementedError("plot_trajectories not implemented")


def plot_belief_evolution(
    belief_history: List[jnp.ndarray],
    times: jnp.ndarray,
    true_goal_id: int,
    save_path: Optional[str] = None
):
    """Plot belief distribution evolution over time.

    Args:
        belief_history: List of belief distributions (each is num_goals,)
        times: Timestamps corresponding to belief_history
        true_goal_id: Index of true goal (highlighted)
        save_path: Optional path to save figure
    """
    # TODO: Implement
    # 1. Stack beliefs into array (T, num_goals)
    # beliefs = jnp.stack(belief_history, axis=0)

    # 2. Create figure
    # fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Plot each goal's probability over time
    # for goal_id in range(beliefs.shape[1]):
    #     label = f"Goal {goal_id}" + (" (true)" if goal_id == true_goal_id else "")
    #     ax.plot(times, beliefs[:, goal_id], label=label)

    # 4. Add labels and legend
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Belief Probability")
    # ax.legend()

    # 5. Save or show
    # if save_path:
    #     plt.savefig(save_path)

    raise NotImplementedError("plot_belief_evolution not implemented")


def plot_metrics(
    metrics: Dict[str, float],
    save_path: Optional[str] = None
):
    """Create multi-panel visualization of metrics.

    Args:
        metrics: Dictionary of metric names and values
        save_path: Optional path to save figure
    """
    # TODO: Implement
    # Create bar chart or multi-panel figure showing all metrics
    raise NotImplementedError("plot_metrics not implemented")


def plot_distance_over_time(
    trajectory_D: Trajectory,
    trajectory_I: Trajectory,
    intercept_threshold: float,
    save_path: Optional[str] = None
):
    """Plot distance between agents over time.

    Args:
        trajectory_D: Deceptive agent's trajectory
        trajectory_I: Interceptor agent's trajectory
        intercept_threshold: Threshold distance for interception
        save_path: Optional path to save figure
    """
    # TODO: Implement
    # 1. Compute distance at each timestep
    # 2. Plot distance vs time
    # 3. Add horizontal line for threshold
    raise NotImplementedError("plot_distance_over_time not implemented")


def save_trajectories_csv(
    trajectory_D: Trajectory,
    trajectory_I: Trajectory,
    output_path: str
):
    """Save trajectories to CSV file.

    Args:
        trajectory_D: Deceptive agent's trajectory
        trajectory_I: Interceptor agent's trajectory
        output_path: Path for output CSV file

    CSV Format:
        time, x_D, y_D, vx_D, vy_D, x_I, y_I, vx_I, vy_I
    """
    # TODO: Implement
    # with open(output_path, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['time', 'x_D', 'y_D', 'vx_D', 'vy_D', 'x_I', 'y_I', 'vx_I', 'vy_I'])
    #     T = min(len(trajectory_D.times), len(trajectory_I.times))
    #     for i in range(T):
    #         writer.writerow([
    #             trajectory_D.times[i],
    #             trajectory_D.positions[i, 0], trajectory_D.positions[i, 1],
    #             trajectory_D.velocities[i, 0], trajectory_D.velocities[i, 1],
    #             trajectory_I.positions[i, 0], trajectory_I.positions[i, 1],
    #             trajectory_I.velocities[i, 0], trajectory_I.velocities[i, 1]
    #         ])
    raise NotImplementedError("save_trajectories_csv not implemented")


def save_metrics_csv(
    metrics: Dict[str, float],
    output_path: str
):
    """Save metrics to CSV file.

    Args:
        metrics: Dictionary of metric names and values
        output_path: Path for output CSV file

    CSV Format:
        metric_name, value
    """
    # TODO: Implement
    # with open(output_path, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['metric', 'value'])
    #     for metric_name, value in metrics.items():
    #         writer.writerow([metric_name, value])
    raise NotImplementedError("save_metrics_csv not implemented")


def save_belief_history_csv(
    belief_history: List[jnp.ndarray],
    times: jnp.ndarray,
    output_path: str
):
    """Save belief history to CSV file.

    Args:
        belief_history: List of belief distributions
        times: Timestamps
        output_path: Path for output CSV file

    CSV Format:
        time, goal_0_prob, goal_1_prob, ..., goal_N_prob
    """
    # TODO: Implement
    # with open(output_path, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     num_goals = len(belief_history[0])
    #     header = ['time'] + [f'goal_{i}_prob' for i in range(num_goals)]
    #     writer.writerow(header)
    #     for t, belief in zip(times, belief_history):
    #         writer.writerow([t] + belief.tolist())
    raise NotImplementedError("save_belief_history_csv not implemented")


def create_animation(
    workspace: Workspace,
    trajectory_D: Trajectory,
    trajectory_I: Trajectory,
    belief_history: List[jnp.ndarray],
    save_path: str,
    fps: int = 10
):
    """Create animated visualization of simulation.

    Args:
        workspace: Workspace with obstacles
        trajectory_D: Deceptive agent's trajectory
        trajectory_I: Interceptor agent's trajectory
        belief_history: Belief evolution
        save_path: Path to save animation (MP4/GIF)
        fps: Frames per second
    """
    # TODO: Implement using matplotlib.animation
    # This would create an animated plot showing:
    # - Agents moving along trajectories
    # - Belief distribution updating in side panel
    raise NotImplementedError("create_animation not implemented")
