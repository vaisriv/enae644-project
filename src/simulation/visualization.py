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


def plot_workspace_with_trajectories(
    workspace: Workspace,
    trajectories: List[Trajectory],
    goals: jnp.ndarray,
    save_path: Optional[str] = None,
):
    """Plot workspace with agent trajectories and obstacles.

    Args:
        workspace: Workspace with bounds and obstacles
        trajectories: List of agent trajectories; first is Agent D (blue),
                      second is Agent I (red)
        goals: (num_goals, 2) array of candidate goal positions
        save_path: Optional path to save figure (PNG/PDF/SVG)
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    bounds = workspace.bounds
    ax.set_xlim(float(bounds[0, 0]), float(bounds[0, 1]))
    ax.set_ylim(float(bounds[1, 0]), float(bounds[1, 1]))
    ax.set_aspect("equal")

    # Plot obstacles
    for obstacle in workspace.obstacles:
        if isinstance(obstacle, CircleObstacle):
            circle = patches.Circle(
                (float(obstacle.center[0]), float(obstacle.center[1])),
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

    # Plot trajectories (first = Agent D, second = Agent I)
    colors = ["blue", "red"]
    labels = ["Agent D (Deceptive)", "Agent I (Interceptor)"]
    for i, traj in enumerate(trajectories):
        color = colors[i] if i < len(colors) else f"C{i}"
        label = labels[i] if i < len(labels) else f"Agent {i}"
        ax.plot(
            traj.positions[:, 0],
            traj.positions[:, 1],
            color=color,
            linewidth=2,
            label=label,
            alpha=0.7,
        )
        ax.scatter(
            traj.positions[0, 0],
            traj.positions[0, 1],
            marker="o",
            s=150,
            color=color,
            edgecolors="black",
            linewidths=2,
            zorder=5,
        )

    # Plot candidate goals
    for i in range(len(goals)):
        ax.scatter(
            float(goals[i, 0]),
            float(goals[i, 1]),
            marker="x",
            s=200,
            color="orange",
            linewidths=3,
            label="Candidate Goal" if i == 0 else None,
        )

    ax.legend(loc="best", fontsize=10)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Agent Trajectories", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_belief_evolution(
    belief_history: List[jnp.ndarray],
    goals: jnp.ndarray,
    observer_belief_history: Optional[List[jnp.ndarray]] = None,
    save_path: Optional[str] = None,
):
    """Plot belief distribution evolution over simulation steps.

    When ``observer_belief_history`` is provided, renders two side-by-side panels:
    the particle-filter belief (left) and the RNN observer goal probabilities (right).
    Otherwise renders a single panel.

    Args:
        belief_history: List of (num_goals,) particle-filter belief distributions
        goals: (num_goals, 2) array of candidate goal positions (for labels)
        observer_belief_history: Optional list of (num_goals,) RNN observer
            probability vectors, one per simulation step
        save_path: Optional path to save figure
    """
    has_observer = bool(observer_belief_history)
    ncols = 2 if has_observer else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]  # type: ignore[list-item]

    colors = plt.cm.tab10(range(len(belief_history[0])))  # type: ignore[attr-defined]
    num_goals = int(jnp.array(belief_history[0]).shape[0])

    def _draw_panel(ax, history: List[jnp.ndarray], title: str) -> None:
        data = jnp.stack(history, axis=0)
        steps = jnp.arange(len(history))
        for goal_id in range(num_goals):
            label = f"Goal {goal_id} ({float(goals[goal_id, 0]):.1f}, {float(goals[goal_id, 1]):.1f})"
            ax.plot(
                steps,
                data[:, goal_id],
                label=label,
                linewidth=2,
                color=colors[goal_id],
                alpha=0.85,
            )
        ax.axhline(
            y=1 / num_goals,
            color="gray",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
            label="uniform",
        )
        ax.set_xlabel("Simulation Step", fontsize=11)
        ax.set_ylabel("Probability", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    _draw_panel(axes[0], belief_history, "Particle Filter Belief")
    if has_observer:
        _draw_panel(axes[1], observer_belief_history, "RNN Observer Probabilities")  # type: ignore[arg-type]

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_trajectories(
    trajectories: Dict[str, Trajectory],
    save_path: str,
):
    """Save trajectories to CSV file.

    Args:
        trajectories: Dict mapping agent name to Trajectory
                      (e.g. {"Agent_D": traj_D, "Agent_I": traj_I})
        save_path: Path for output CSV file

    CSV Format:
        agent, time, x, y, vx, vy
    """
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "time", "x", "y", "vx", "vy"])
        for agent_name, traj in trajectories.items():
            for i in range(len(traj.times)):
                writer.writerow(
                    [
                        agent_name,
                        float(traj.times[i]),
                        float(traj.positions[i, 0]),
                        float(traj.positions[i, 1]),
                        float(traj.velocities[i, 0]),
                        float(traj.velocities[i, 1]),
                    ]
                )


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
    T = min(len(trajectory_D.positions), len(trajectory_I.positions))
    distances = []
    for i in range(T):
        dist = jnp.linalg.norm(trajectory_D.positions[i] - trajectory_I.positions[i])
        distances.append(float(dist))

    times = trajectory_D.times[:T]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, distances, "b-", linewidth=2, label="Agent Distance")
    ax.axhline(
        y=intercept_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Interception Threshold ({intercept_threshold}m)",
    )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Distance (m)", fontsize=12)
    ax.set_title("Distance Between Agents Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_metrics_csv(metrics: Dict[str, float], output_path: str):
    """Save metrics to CSV file.

    Args:
        metrics: Dictionary of metric names and values
        output_path: Path for output CSV file

    CSV Format:
        metric, value
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric_name, value in metrics.items():
            writer.writerow([metric_name, float(value)])


def save_belief_history_csv(
    belief_history: List[jnp.ndarray],
    times: jnp.ndarray,
    output_path: str,
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
    fig = plt.figure(figsize=(16, 8))
    ax_workspace = plt.subplot(1, 2, 1)
    ax_belief = plt.subplot(1, 2, 2)

    bounds = workspace.bounds
    ax_workspace.set_xlim(float(bounds[0, 0]), float(bounds[0, 1]))
    ax_workspace.set_ylim(float(bounds[1, 0]), float(bounds[1, 1]))
    ax_workspace.set_aspect("equal")
    ax_workspace.set_xlabel("X (m)", fontsize=12)
    ax_workspace.set_ylabel("Y (m)", fontsize=12)
    ax_workspace.set_title("Agent Trajectories", fontsize=14, fontweight="bold")
    ax_workspace.grid(True, alpha=0.3)

    for obstacle in workspace.obstacles:
        if isinstance(obstacle, CircleObstacle):
            circle = patches.Circle(
                (float(obstacle.center[0]), float(obstacle.center[1])),
                obstacle.radius,
                color="gray",
                alpha=0.5,
            )
            ax_workspace.add_patch(circle)
        elif isinstance(obstacle, PolygonObstacle):
            polygon = patches.Polygon(obstacle.vertices, color="gray", alpha=0.5)
            ax_workspace.add_patch(polygon)

    (line_D,) = ax_workspace.plot(
        [], [], "b-", linewidth=2, alpha=0.5, label="Agent D Path"
    )
    (line_I,) = ax_workspace.plot(
        [], [], "r-", linewidth=2, alpha=0.5, label="Agent I Path"
    )
    (marker_D,) = ax_workspace.plot([], [], "bo", markersize=15, label="Agent D")
    (marker_I,) = ax_workspace.plot([], [], "ro", markersize=15, label="Agent I")
    ax_workspace.legend(loc="upper right", fontsize=10)

    num_goals = len(belief_history[0])
    ax_belief.set_xlim(0, num_goals - 1)
    ax_belief.set_ylim(0, 1)
    ax_belief.set_xlabel("Goal Index", fontsize=12)
    ax_belief.set_ylabel("Belief Probability", fontsize=12)
    ax_belief.set_title("Goal Belief Distribution", fontsize=14, fontweight="bold")
    ax_belief.grid(True, alpha=0.3)

    x_pos = list(range(num_goals))
    bars = ax_belief.bar(x_pos, [0] * num_goals, color="skyblue", edgecolor="black")

    time_text = ax_workspace.text(
        0.02,
        0.98,
        "",
        transform=ax_workspace.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    def update(frame):
        line_D.set_data(
            trajectory_D.positions[: frame + 1, 0],
            trajectory_D.positions[: frame + 1, 1],
        )
        line_I.set_data(
            trajectory_I.positions[: frame + 1, 0],
            trajectory_I.positions[: frame + 1, 1],
        )
        marker_D.set_data(
            [trajectory_D.positions[frame, 0]], [trajectory_D.positions[frame, 1]]
        )
        marker_I.set_data(
            [trajectory_I.positions[frame, 0]], [trajectory_I.positions[frame, 1]]
        )
        if frame < len(belief_history):
            belief = belief_history[frame]
            for bar, prob in zip(bars, belief):
                bar.set_height(float(prob))
        time_text.set_text(f"Time: {trajectory_D.times[frame]:.2f}s")
        return line_D, line_I, marker_D, marker_I, *bars, time_text

    num_frames = min(len(trajectory_D.positions), len(trajectory_I.positions))
    interval = 1000 / fps

    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=interval, blit=True, repeat=True
    )

    if save_path.endswith(".gif"):
        anim.save(save_path, writer="pillow", fps=fps)
    else:
        anim.save(save_path, writer="ffmpeg", fps=fps, dpi=100)

    plt.close(fig)
