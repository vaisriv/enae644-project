"""Main entry point for adversarial motion planning simulation.

Runs the simulation using pre-trained model checkpoints. Training is performed
separately via: uv run adversarial-planning-train

Pipeline:
1. Load configuration from YAML
2. Verify model checkpoints exist (run training first if missing)
3. Initialize JAX random key
4. Run simulation (loads checkpoints internally)
5. Save outputs (trajectories, metrics, visualizations)
"""

from pathlib import Path
import jax

# TODO: Uncomment when modules are implemented
# from src.simulation.controller import run_simulation
# from src.simulation.config import load_config
# from src.simulation.visualization import (
#     plot_workspace_with_trajectories,
#     plot_belief_evolution,
#     save_trajectories,
#     save_metrics_csv,
#     save_belief_history_csv,
# )


def _check_checkpoints(config) -> None:
    """Verify that required model checkpoints exist before running simulation.

    Args:
        config: SimulationConfig with checkpoint paths in agent configs

    Raises:
        FileNotFoundError: If any required checkpoint is missing
    """
    # TODO: Implement checkpoint verification
    # observer_path = config.deceptive_agent_config.observer.checkpoint_path
    # irl_path = config.interceptor_agent_config.irl.checkpoint_path
    # for path in [observer_path, irl_path]:
    #     if not Path(path).exists():
    #         raise FileNotFoundError(
    #             f"Model checkpoint not found: {path}\n"
    #             "Run 'uv run adversarial-planning-train' to train models first."
    #         )
    raise NotImplementedError("_check_checkpoints not implemented")


def main():
    """Main execution function."""
    print("Adversarial Motion Planning Simulation")
    print("=" * 50)

    # ========================================================================
    # 1. Parse command-line arguments
    # ========================================================================
    # TODO: Implement argument parsing
    # parser = argparse.ArgumentParser(
    #     description="Run adversarial motion planning simulation"
    # )
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     default="data/configs/experiment_simple_obstacle.yaml",
    #     help="Path to YAML configuration file",
    # )
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=None,
    #     help="Random seed (overrides config)",
    # )
    # args = parser.parse_args()

    # Placeholder values
    config_path = "data/configs/experiment_simple_obstacle.yaml"
    output_dir = "outputs"
    seed = None

    # ========================================================================
    # 2. Load configuration
    # ========================================================================
    print(f"\nLoading configuration from: {config_path}")
    # TODO: Uncomment when load_config is implemented
    # config = load_config(config_path)

    # Override seed if provided
    # if seed is not None:
    #     config.simulation_params.random_seed = seed

    # print(f"  Workspace bounds: {config.workspace.bounds}")
    # print(f"  Random seed: {config.simulation_params.random_seed}")
    # print(f"  Max simulation time: {config.simulation_params.max_time}s")

    # ========================================================================
    # 3. Verify model checkpoints exist
    # ========================================================================
    # TODO: Uncomment when _check_checkpoints and load_config are implemented
    # _check_checkpoints(config)

    # ========================================================================
    # 4. Initialize JAX random key
    # ========================================================================
    # TODO: Use config.simulation_params.random_seed
    key = jax.random.PRNGKey(42)
    print("\nInitialized JAX PRNG with seed: 42")

    # ========================================================================
    # 5. Run simulation
    # ========================================================================
    print("\nRunning simulation...")
    # TODO: Uncomment when run_simulation is implemented
    # result = run_simulation(config, key)

    # print(f"  Winner: {result.winner}")
    # print(f"  Completion time: {result.completion_time:.2f}s")

    # ========================================================================
    # 6. Create output directory
    # ========================================================================
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nCreating output directory: {output_path}")

    figures_dir = output_path / "figures"
    figures_dir.mkdir(exist_ok=True)

    text_dir = output_path / "text"
    text_dir.mkdir(exist_ok=True)

    # ========================================================================
    # 7. Save trajectories
    # ========================================================================
    print("\nSaving trajectories...")
    # TODO: Uncomment when implemented
    # save_trajectories(
    #     trajectories={"Agent_D": result.trajectory_D, "Agent_I": result.trajectory_I},
    #     save_path=str(text_dir / "trajectories.csv"),
    # )
    # print(f"  Saved to: {text_dir / 'trajectories.csv'}")

    # ========================================================================
    # 8. Save metrics
    # ========================================================================
    print("\nSaving metrics...")
    # TODO: Uncomment when implemented
    # save_metrics_csv(result.metrics, str(text_dir / "metrics.csv"))
    # print(f"  Saved to: {text_dir / 'metrics.csv'}")
    # for metric_name, value in result.metrics.items():
    #     print(f"  {metric_name}: {value:.4f}")

    # ========================================================================
    # 9. Save belief history
    # ========================================================================
    print("\nSaving belief history...")
    # TODO: Uncomment when implemented
    # save_belief_history_csv(
    #     result.belief_history,
    #     result.trajectory_I.times,
    #     str(text_dir / "belief_history.csv"),
    # )
    # print(f"  Saved to: {text_dir / 'belief_history.csv'}")

    # ========================================================================
    # 10. Generate visualizations
    # ========================================================================
    print("\nGenerating visualizations...")

    # TODO: Uncomment when implemented
    # true_goal_id = config.deceptive_agent_config.candidate_goals.index(
    #     config.deceptive_agent_config.true_goal
    # )
    # candidate_goals = jnp.array(config.deceptive_agent_config.candidate_goals)

    # plot_workspace_with_trajectories(
    #     workspace=config.workspace,
    #     trajectories=[result.trajectory_D, result.trajectory_I],
    #     goals=candidate_goals,
    #     save_path=str(figures_dir / "trajectories.png"),
    # )
    # print(f"  Saved trajectory plot to: {figures_dir / 'trajectories.png'}")

    # plot_belief_evolution(
    #     result.belief_history,
    #     candidate_goals=candidate_goals,
    #     save_path=str(figures_dir / "belief_evolution.png"),
    # )
    # print(f"  Saved belief plot to: {figures_dir / 'belief_evolution.png'}")

    # ========================================================================
    # Done
    # ========================================================================
    print("\n" + "=" * 50)
    print("Simulation complete!")
    print(f"Outputs saved to: {output_path}")


if __name__ == "__main__":
    main()
