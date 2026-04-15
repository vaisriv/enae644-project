"""Visualization utilities for simulation results.

This module provides functions for creating plots and saving data outputs
for analysis and presentation.
"""

from typing import List, Dict, Optional
import csv

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import jax.numpy as jnp

from src.shared.workspace import Workspace, CircleObstacle, PolygonObstacle
from src.shared.trajectory import Trajectory


def plot_trajectories(
    workspace: Workspace,
    trajectory_D: Trajectory,
    trajectory_I: Trajectory,
    candidate_goals: List[jnp.ndarray],
    true_goal: jnp.ndarray,
    save_path: Optional[str] = None,
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
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot workspace bounds
    bounds = workspace.bounds
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_aspect("equal")

    # Plot obstacles
    for obstacle in workspace.obstacles:
        if isinstance(obstacle, CircleObstacle):
            circle = patches.Circle(
                obstacle.center,
                obstacle.radius,
                color="gray",
                alpha=0.5,
                label="Obstacle",
            )
            ax.add_patch(circle)
        elif isinstance(obstacle, PolygonObstacle):
            polygon = patches.Polygon(
                obstacle.vertices, color="gray", alpha=0.5, label="Obstacle"
            )
            ax.add_patch(polygon)

    # Plot trajectories
    ax.plot(
        trajectory_D.positions[:, 0],
        trajectory_D.positions[:, 1],
        "b-",
        linewidth=2,
        label="Agent D (Deceptive)",
        alpha=0.7,
    )
    ax.plot(
        trajectory_I.positions[:, 0],
        trajectory_I.positions[:, 1],
        "r-",
        linewidth=2,
        label="Agent I (Interceptor)",
        alpha=0.7,
    )

    # Plot start positions
    ax.scatter(
        trajectory_D.positions[0, 0],
        trajectory_D.positions[0, 1],
        marker="o",
        s=150,
        color="blue",
        edgecolors="black",
        linewidths=2,
        label="D Start",
        zorder=5,
    )
    ax.scatter(
        trajectory_I.positions[0, 0],
        trajectory_I.positions[0, 1],
        marker="o",
        s=150,
        color="red",
        edgecolors="black",
        linewidths=2,
        label="I Start",
        zorder=5,
    )

    # Plot candidate goals
    for i, goal in enumerate(candidate_goals):
        ax.scatter(
            goal[0],
            goal[1],
            marker="x",
            s=200,
            color="orange",
            linewidths=3,
            label="Candidate Goal" if i == 0 else None,
        )

    # Plot true goal
    ax.scatter(
        true_goal[0],
        true_goal[1],
        marker="*",
        s=500,
        color="gold",
        edgecolors="black",
        linewidths=2,
        label="True Goal",
        zorder=5,
    )

    # Add legend and labels
    ax.legend(loc="best", fontsize=10)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Agent Trajectories", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_belief_evolution(
    belief_history: List[jnp.ndarray],
    times: jnp.ndarray,
    true_goal_id: int,
    save_path: Optional[str] = None,
):
    """Plot belief distribution evolution over time.

    Args:
        belief_history: List of belief distributions (each is num_goals,)
        times: Timestamps corresponding to belief_history
        true_goal_id: Index of true goal (highlighted)
        save_path: Optional path to save figure
    """
    # Stack beliefs into array (T, num_goals)
    beliefs = jnp.stack(belief_history, axis=0)
    num_goals = beliefs.shape[1]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each goal's probability over time
    colors = plt.cm.tab10(range(num_goals))
    for goal_id in range(num_goals):
        is_true = goal_id == true_goal_id
        label = f"Goal {goal_id}" + (" (TRUE)" if is_true else "")
        linewidth = 3 if is_true else 2
        linestyle = "-" if is_true else "--"
        ax.plot(
            times,
            beliefs[:, goal_id],
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            color=colors[goal_id],
            alpha=0.8,
        )

    # Add horizontal line at 0.5 for reference
    ax.axhline(
        y=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="p=0.5"
    )

    # Add labels and legend
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Belief Probability", fontsize=12)
    ax.set_title("Goal Belief Evolution", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_metrics(metrics: Dict[str, float], save_path: Optional[str] = None):
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
    save_path: Optional[str] = None,
):
    """Plot distance between agents over time.

    Args:
        trajectory_D: Deceptive agent's trajectory
        trajectory_I: Interceptor agent's trajectory
        intercept_threshold: Threshold distance for interception
        save_path: Optional path to save figure
    """
    # Compute distance at each timestep
    T = min(len(trajectory_D.positions), len(trajectory_I.positions))
    distances = []
    for i in range(T):
        dist = jnp.linalg.norm(trajectory_D.positions[i] - trajectory_I.positions[i])
        distances.append(float(dist))

    times = trajectory_D.times[:T]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot distance vs time
    ax.plot(times, distances, "b-", linewidth=2, label="Agent Distance")

    # Add horizontal line for threshold
    ax.axhline(
        y=intercept_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Interception Threshold ({intercept_threshold}m)",
    )

    # Add labels and legend
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Distance (m)", fontsize=12)
    ax.set_title("Distance Between Agents Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_trajectories_csv(
    trajectory_D: Trajectory, trajectory_I: Trajectory, output_path: str
):
    """Save trajectories to CSV file.

    Args:
        trajectory_D: Deceptive agent's trajectory
        trajectory_I: Interceptor agent's trajectory
        output_path: Path for output CSV file

    CSV Format:
        time, x_D, y_D, vx_D, vy_D, x_I, y_I, vx_I, vy_I
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time", "x_D", "y_D", "vx_D", "vy_D", "x_I", "y_I", "vx_I", "vy_I"]
        )
        T = min(len(trajectory_D.times), len(trajectory_I.times))
        for i in range(T):
            writer.writerow(
                [
                    float(trajectory_D.times[i]),
                    float(trajectory_D.positions[i, 0]),
                    float(trajectory_D.positions[i, 1]),
                    float(trajectory_D.velocities[i, 0]),
                    float(trajectory_D.velocities[i, 1]),
                    float(trajectory_I.positions[i, 0]),
                    float(trajectory_I.positions[i, 1]),
                    float(trajectory_I.velocities[i, 0]),
                    float(trajectory_I.velocities[i, 1]),
                ]
            )


def save_metrics_csv(metrics: Dict[str, float], output_path: str):
    """Save metrics to CSV file.

    Args:
        metrics: Dictionary of metric names and values
        output_path: Path for output CSV file

    CSV Format:
        metric_name, value
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric_name, value in metrics.items():
            writer.writerow([metric_name, float(value)])


def save_belief_history_csv(
    belief_history: List[jnp.ndarray], times: jnp.ndarray, output_path: str
):
    """Save belief history to CSV file.

    Args:
        belief_history: List of belief distributions
        times: Timestamps
        output_path: Path for output CSV file

    CSV Format:
        time, goal_0_prob, goal_1_prob, ..., goal_N_prob
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        num_goals = len(belief_history[0])
        header = ["time"] + [f"goal_{i}_prob" for i in range(num_goals)]
        writer.writerow(header)
        for t, belief in zip(times, belief_history):
            writer.writerow([float(t)] + [float(p) for p in belief.tolist()])


def create_animation(
    workspace: Workspace,
    trajectory_D: Trajectory,
    trajectory_I: Trajectory,
    belief_history: List[jnp.ndarray],
    save_path: str,
    fps: int = 10,
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
    # Setup figure with two subplots: workspace and belief
    fig = plt.figure(figsize=(16, 8))
    ax_workspace = plt.subplot(1, 2, 1)
    ax_belief = plt.subplot(1, 2, 2)

    # ========================================================================
    # Setup workspace subplot
    # ========================================================================
    bounds = workspace.bounds
    ax_workspace.set_xlim(bounds[0, 0], bounds[0, 1])
    ax_workspace.set_ylim(bounds[1, 0], bounds[1, 1])
    ax_workspace.set_aspect("equal")
    ax_workspace.set_xlabel("X (m)", fontsize=12)
    ax_workspace.set_ylabel("Y (m)", fontsize=12)
    ax_workspace.set_title("Agent Trajectories", fontsize=14, fontweight="bold")
    ax_workspace.grid(True, alpha=0.3)

    # Plot obstacles
    for obstacle in workspace.obstacles:
        if isinstance(obstacle, CircleObstacle):
            circle = patches.Circle(
                obstacle.center, obstacle.radius, color="gray", alpha=0.5
            )
            ax_workspace.add_patch(circle)
        elif isinstance(obstacle, PolygonObstacle):
            polygon = patches.Polygon(obstacle.vertices, color="gray", alpha=0.5)
            ax_workspace.add_patch(polygon)

    # Initialize trajectory lines (will show history)
    (line_D,) = ax_workspace.plot(
        [], [], "b-", linewidth=2, alpha=0.5, label="Agent D Path"
    )
    (line_I,) = ax_workspace.plot(
        [], [], "r-", linewidth=2, alpha=0.5, label="Agent I Path"
    )

    # Initialize agent markers (current positions)
    (marker_D,) = ax_workspace.plot([], [], "bo", markersize=15, label="Agent D")
    (marker_I,) = ax_workspace.plot([], [], "ro", markersize=15, label="Agent I")

    ax_workspace.legend(loc="upper right", fontsize=10)

    # ========================================================================
    # Setup belief subplot
    # ========================================================================
    num_goals = len(belief_history[0])
    ax_belief.set_xlim(0, num_goals - 1)
    ax_belief.set_ylim(0, 1)
    ax_belief.set_xlabel("Goal Index", fontsize=12)
    ax_belief.set_ylabel("Belief Probability", fontsize=12)
    ax_belief.set_title("Goal Belief Distribution", fontsize=14, fontweight="bold")
    ax_belief.grid(True, alpha=0.3)

    # Initialize belief bars
    x_pos = list(range(num_goals))
    bars = ax_belief.bar(x_pos, [0] * num_goals, color="skyblue", edgecolor="black")

    # Time text
    time_text = ax_workspace.text(
        0.02,
        0.98,
        "",
        transform=ax_workspace.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # ========================================================================
    # Animation update function
    # ========================================================================
    def update(frame):
        # Update trajectory history
        line_D.set_data(
            trajectory_D.positions[: frame + 1, 0],
            trajectory_D.positions[: frame + 1, 1],
        )
        line_I.set_data(
            trajectory_I.positions[: frame + 1, 0],
            trajectory_I.positions[: frame + 1, 1],
        )

        # Update current agent positions
        marker_D.set_data(
            [trajectory_D.positions[frame, 0]], [trajectory_D.positions[frame, 1]]
        )
        marker_I.set_data(
            [trajectory_I.positions[frame, 0]], [trajectory_I.positions[frame, 1]]
        )

        # Update belief distribution
        if frame < len(belief_history):
            belief = belief_history[frame]
            for bar, prob in zip(bars, belief):
                bar.set_height(float(prob))

        # Update time text
        time_text.set_text(f"Time: {trajectory_D.times[frame]:.2f}s")

        return line_D, line_I, marker_D, marker_I, *bars, time_text

    # ========================================================================
    # Create animation
    # ========================================================================
    num_frames = min(len(trajectory_D.positions), len(trajectory_I.positions))
    interval = 1000 / fps  # milliseconds per frame

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=interval, blit=True, repeat=True
    )

    # Save animation
    if save_path.endswith(".gif"):
        anim.save(save_path, writer="pillow", fps=fps)
    else:
        # Save as MP4 (requires ffmpeg)
        anim.save(save_path, writer="ffmpeg", fps=fps, dpi=100)

    plt.close(fig)
