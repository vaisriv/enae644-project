"""Main entry point for adversarial motion planning simulation.

Runs the simulation using pre-trained model checkpoints. Training is performed
separately via: uv run adversarial-planning-train

Pipeline:
1. Load configuration from YAML
2. Verify model checkpoints exist (run training first if missing)
3. Initialize JAX random key
4. Run simulation (loads checkpoints internally)
5. Save outputs (trajectories, metrics, visualizations)
6. Run α-sweep for ablation study
"""

from pathlib import Path

import jax
import jax.numpy as jnp

from src.simulation.config import SimulationConfig, load_config
from src.simulation.controller import run_simulation
from src.simulation.visualization import (
    plot_belief_evolution,
    plot_distance_over_time,
    plot_workspace_with_trajectories,
    save_belief_history_csv,
    save_metrics_csv,
    save_trajectories,
)


def _check_checkpoints(config: SimulationConfig) -> None:
    """Verify that required model checkpoints exist before running simulation.

    Args:
        config: SimulationConfig with checkpoint paths in agent configs

    Raises:
        FileNotFoundError: If any required checkpoint is missing
    """
    observer_path = config.deceptive_agent_config.observer.checkpoint_path
    irl_path = config.interceptor_agent_config.irl.checkpoint_path
    for path in [observer_path, irl_path]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}\n"
                "Run 'uv run adversarial-planning-train' to train models first."
            )


def main():
    """Main execution function."""
    print("Adversarial Motion Planning Simulation")
    print("=" * 50)

    config_path = "data/configs/experiment_simple_obstacle.yaml"
    output_dir = Path("outputs")

    # =========================================================
    # 1. Load configuration
    # =========================================================
    print(f"\nLoading configuration from: {config_path}")
    config = load_config(config_path)
    print(f"  Workspace bounds    : {config.workspace.bounds}")
    print(f"  Random seed         : {config.simulation_params.random_seed}")
    print(f"  Max simulation time : {config.simulation_params.max_time}s")

    # =========================================================
    # 2. Verify checkpoints
    # =========================================================
    _check_checkpoints(config)

    # =========================================================
    # 3. Initialize JAX key
    # =========================================================
    key = jax.random.PRNGKey(config.simulation_params.random_seed)

    # =========================================================
    # 4. Prepare output directories
    # =========================================================
    figures_dir = output_dir / "figures"
    text_dir = output_dir / "text"
    figures_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 5. Run primary simulation
    # =========================================================
    print("\nRunning simulation...")
    key, sim_key = jax.random.split(key)
    result = run_simulation(config, sim_key)

    print(f"\n  Winner            : {result.winner}")
    print(f"  Completion time   : {result.completion_time:.2f}s")
    print(f"  Observer accuracy : {result.metrics['observer_accuracy']:.3f}")
    print(f"  Path length ratio : {result.metrics['path_length_ratio']:.3f}")
    print(f"  Min intercept dist: {result.metrics['min_interception_distance']:.3f}")

    # =========================================================
    # 6. Save primary outputs
    # =========================================================
    print("\nSaving outputs...")

    save_trajectories(
        trajectories={"Agent_D": result.trajectory_D, "Agent_I": result.trajectory_I},
        save_path=str(text_dir / "trajectories.csv"),
    )
    print(f"  Trajectories → {text_dir / 'trajectories.csv'}")

    save_metrics_csv(result.metrics, str(text_dir / "metrics.csv"))
    print(f"  Metrics      → {text_dir / 'metrics.csv'}")

    save_belief_history_csv(
        result.belief_history,
        result.trajectory_I.times,
        str(text_dir / "belief_history.csv"),
    )
    print(f"  Belief hist  → {text_dir / 'belief_history.csv'}")

    # =========================================================
    # 7. Save figures
    # =========================================================
    from src.simulation.config import create_workspace_from_config  # type: ignore[attr-defined]

    workspace = create_workspace_from_config(config.workspace)
    candidate_goals = jnp.array(config.deceptive_agent_config.candidate_goals)

    plot_workspace_with_trajectories(
        workspace=workspace,
        trajectories=[result.trajectory_D, result.trajectory_I],
        goals=candidate_goals,
        save_path=str(figures_dir / "trajectories.png"),
    )
    print(f"  Trajectory plot → {figures_dir / 'trajectories.png'}")

    plot_belief_evolution(
        belief_history=result.belief_history,
        goals=candidate_goals,
        observer_belief_history=result.observer_belief_history,
        save_path=str(figures_dir / "belief_evolution.png"),
    )
    print(f"  Belief plot     → {figures_dir / 'belief_evolution.png'}")

    plot_distance_over_time(
        trajectory_D=result.trajectory_D,
        trajectory_I=result.trajectory_I,
        intercept_threshold=config.simulation_params.intercept_threshold,
        save_path=str(figures_dir / "distance_over_time.png"),
    )
    print(f"  Distance plot   → {figures_dir / 'distance_over_time.png'}")

    # =========================================================
    # 8. Alpha sweep (ablation study)
    # =========================================================
    _run_alpha_sweep(config, figures_dir, text_dir)

    # =========================================================
    # Done
    # =========================================================
    print("\n" + "=" * 50)
    print("Simulation complete!")
    print(f"Outputs saved to: {output_dir}")


def _run_alpha_sweep(
    config: SimulationConfig, figures_dir: Path, text_dir: Path
) -> None:
    """Plan trajectories for a range of α values and save comparison metrics.

    Runs only the RRT* planning and observer evaluation for each α — not the
    full simulation loop — so the sweep completes in reasonable time.
    """
    import csv
    import dataclasses

    from src.deceptive.observer import load_observer
    from src.deceptive.planner import adversarial_rrt_star
    from src.simulation.config import create_workspace_from_config  # type: ignore[attr-defined]
    from src.simulation.metrics import (
        compute_deception_effectiveness,
        compute_observer_accuracy,
        compute_path_length_ratio,
    )
    from src.shared.trajectory import create_trajectory

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"\nRunning α sweep (planning-only) over {alphas}...")

    workspace = create_workspace_from_config(config.workspace)
    d_cfg = config.deceptive_agent_config
    _goals = jnp.array(d_cfg.candidate_goals)
    true_goal = jnp.array(d_cfg.true_goal)
    true_goal_id = d_cfg.candidate_goals.index(d_cfg.true_goal)
    start = jnp.array(d_cfg.initial_position)

    observer_net = load_observer(d_cfg.observer.checkpoint_path, d_cfg.observer)

    # Optimal (straight-line) trajectory for path length ratio baseline
    optimal_positions = jnp.stack([start, true_goal])
    straight_dist = float(jnp.linalg.norm(true_goal - start))
    optimal_traj = create_trajectory(
        jnp.array([0.0, straight_dist * 0.1]),
        optimal_positions,
    )

    rows = []
    traj_by_alpha = {}

    for alpha in alphas:
        print(f"  α = {alpha:.2f} ...", end="", flush=True)
        cfg_planner = dataclasses.replace(d_cfg.planner, deception_weight=alpha)
        sweep_planner = dataclasses.replace(cfg_planner, max_iterations=2000)

        key = jax.random.PRNGKey(
            config.simulation_params.random_seed + int(alpha * 100)
        )
        traj = adversarial_rrt_star(
            start=start,
            goal=true_goal,
            workspace=workspace,
            observer_net=observer_net,
            true_goal_id=true_goal_id,
            alpha=alpha,
            config=sweep_planner,
            key=key,
        )

        obs_acc = compute_observer_accuracy(observer_net, traj, true_goal_id)
        plr = compute_path_length_ratio(traj, optimal_traj)
        deception_eff = compute_deception_effectiveness(obs_acc, plr, alpha)

        rows.append(
            {
                "alpha": alpha,
                "observer_accuracy": obs_acc,
                "path_length_ratio": plr,
                "deception_effectiveness": deception_eff,
                "num_waypoints": traj.positions.shape[0],
            }
        )
        traj_by_alpha[alpha] = traj
        print(f" obs_acc={obs_acc:.3f} plr={plr:.3f}")

    # Save CSV
    csv_path = text_dir / "alpha_sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  α sweep CSV → {csv_path}")

    # Save comparison plot
    _plot_alpha_sweep(alphas, rows, traj_by_alpha, config, figures_dir)


def _plot_alpha_sweep(alphas, rows, traj_by_alpha, config, figures_dir: Path) -> None:
    """Save α sweep comparison figure."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    from src.simulation.config import create_workspace_from_config  # type: ignore[attr-defined]
    from src.shared.workspace import CircleObstacle

    workspace = create_workspace_from_config(config.workspace)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # -- Panel 1: Trajectories for each α
    ax = axes[0]
    bounds = workspace.bounds
    ax.set_xlim(float(bounds[0, 0]), float(bounds[0, 1]))
    ax.set_ylim(float(bounds[1, 0]), float(bounds[1, 1]))
    for obs in workspace.obstacles:
        if isinstance(obs, CircleObstacle):
            circle = plt.Circle(
                (float(obs.center[0]), float(obs.center[1])),
                float(obs.radius),
                color="gray",
                alpha=0.5,
            )
            ax.add_patch(circle)
    cmap = matplotlib.colormaps["plasma"]
    colors = [cmap(i / max(len(alphas) - 1, 1)) for i in range(len(alphas))]
    for i, alpha in enumerate(alphas):
        traj = traj_by_alpha[alpha]
        ax.plot(
            traj.positions[:, 0],
            traj.positions[:, 1],
            color=colors[i],
            linewidth=1.5,
            label=f"α={alpha:.2f}",
        )
    goals = config.deceptive_agent_config.candidate_goals
    for g in goals:
        ax.plot(g[0], g[1], "k*", markersize=10)
    ax.legend(fontsize=7)
    ax.set_title("Agent D Trajectories by α")
    ax.set_aspect("equal")

    # -- Panel 2: Observer accuracy vs α
    ax = axes[1]
    obs_acc = [r["observer_accuracy"] for r in rows]
    ax.plot(alphas, obs_acc, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("α (deception weight)", fontsize=11)
    ax.set_ylabel("Observer accuracy", fontsize=11)
    ax.set_title("Observer Accuracy vs α")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # -- Panel 3: Path length ratio vs α
    ax = axes[2]
    plr = [r["path_length_ratio"] for r in rows]
    ax.plot(alphas, plr, "ro-", linewidth=2, markersize=8)
    ax.set_xlabel("α (deception weight)", fontsize=11)
    ax.set_ylabel("Path length ratio", fontsize=11)
    ax.set_title("Path Length Ratio vs α")
    ax.axhline(y=1.0, color="gray", linestyle="--", label="Optimal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = str(figures_dir / "alpha_sweep_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  α sweep plot  → {save_path}")


if __name__ == "__main__":
    main()
