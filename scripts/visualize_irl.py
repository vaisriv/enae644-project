"""Visualize the expert demonstration paths used for IRL training.

The demonstrations are not cached anywhere — they are regenerated on every
training run. This script reproduces training's PRNG split chain so the
plotted paths are exactly the ones the IRL model was trained on.

Usage:
    uv run demo-irl
    uv run demo-irl --num-demos 30
    uv run demo-irl --save outputs/figures/irl_demonstrations.png
    uv run demo-irl --config <path-to-yaml>
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generators import generate_irl_demonstrations
from src.shared.workspace import CircleObstacle, PolygonObstacle
from src.simulation.config import create_workspace_from_config, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--config",
        default="data/configs/experiment_simple_obstacle.yaml",
        help="Path to experiment YAML (default: %(default)s)",
    )
    p.add_argument(
        "--num-demos",
        type=int,
        default=None,
        help="Override number of demos to plot (default: config.training.irl.num_demonstrations)",
    )
    p.add_argument(
        "--save",
        default=None,
        help="Save PNG to this path instead of showing interactively",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    workspace = create_workspace_from_config(config.workspace)
    goals = jnp.array(config.deceptive_agent_config.candidate_goals)
    num_demos = args.num_demos or config.training.irl.num_demonstrations

    # Mirror the PRNG split chain used in src/training.py so the demos plotted
    # here are the exact ones the IRL model trained on.
    key = jax.random.PRNGKey(config.simulation_params.random_seed)
    _obs_key, irl_key = jax.random.split(key)
    _train_key, data_key = jax.random.split(irl_key)

    print(f"Generating {num_demos} demonstrations toward {goals.shape[0]} goals...")
    demos = generate_irl_demonstrations(workspace, goals, num_demos, data_key)

    fig, ax = plt.subplots(figsize=(8, 8))
    bounds = workspace.bounds
    ax.set_xlim(float(bounds[0, 0]), float(bounds[0, 1]))
    ax.set_ylim(float(bounds[1, 0]), float(bounds[1, 1]))
    ax.set_aspect("equal")

    for obs in workspace.obstacles:
        if isinstance(obs, CircleObstacle):
            ax.add_patch(
                patches.Circle(
                    (float(obs.center[0]), float(obs.center[1])),
                    obs.radius,
                    color="gray",
                    alpha=0.5,
                )
            )
        elif isinstance(obs, PolygonObstacle):
            ax.add_patch(patches.Polygon(obs.vertices, color="gray", alpha=0.5))

    num_goals = int(goals.shape[0])
    cmap = plt.get_cmap("tab10")
    seen_goals: set[int] = set()
    for i, traj in enumerate(demos):
        goal_id = i % num_goals  # matches generate_irl_demonstrations cycling
        color = cmap(goal_id)
        label = f"Goal {goal_id}" if goal_id not in seen_goals else None
        seen_goals.add(goal_id)
        ax.plot(
            traj.positions[:, 0],
            traj.positions[:, 1],
            color=color,
            linewidth=1.2,
            alpha=0.6,
            label=label,
        )
        ax.scatter(
            float(traj.positions[0, 0]),
            float(traj.positions[0, 1]),
            color=color,
            s=15,
            alpha=0.8,
            zorder=3,
        )

    for g in range(num_goals):
        ax.scatter(
            float(goals[g, 0]),
            float(goals[g, 1]),
            marker="x",
            s=200,
            color="orange",
            linewidths=3,
            zorder=4,
            label="Candidate Goal" if g == 0 else None,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"IRL Expert Demonstrations  ({num_demos} paths, {num_goals} goals)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
