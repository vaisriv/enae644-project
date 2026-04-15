"""Main entry point for adversarial motion planning simulation.

This script runs the complete simulation pipeline:
1. Load configuration from YAML
2. Initialize JAX random key
3. Run simulation
4. Save outputs (trajectories, metrics, visualizations)
"""

import argparse
from pathlib import Path
import jax
import jax.numpy as jnp

# TODO: Import when modules are implemented
# from src.simulation.controller import run_simulation
# from src.simulation.config import load_config
# from src.simulation.visualization import (
#     plot_trajectories,
#     plot_belief_evolution,
#     save_trajectories_csv,
#     save_metrics_csv,
#     save_belief_history_csv
# )

__all__ = ["main"]


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
    #     required=True,
    #     help="Path to YAML configuration file"
    # )
    # parser.add_argument(
    #     "--output-dir",
    #     type=str,
    #     default="outputs",
    #     help="Directory for output files"
    # )
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=None,
    #     help="Random seed (overrides config)"
    # )
    # args = parser.parse_args()

    # Placeholder values
    config_path = "configs/default.yaml"
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
    #     config.simulation.random_seed = seed

    # print(f"  Workspace bounds: {config.workspace.bounds}")
    # print(f"  Random seed: {config.simulation.random_seed}")
    # print(f"  Max simulation time: {config.simulation.max_time}s")

    # ========================================================================
    # 3. Initialize JAX random key
    # ========================================================================
    # TODO: Use config.simulation.random_seed
    key = jax.random.PRNGKey(42)
    print(f"\nInitialized JAX PRNG with seed: 42")

    # ========================================================================
    # 4. Run simulation
    # ========================================================================
    print("\nRunning simulation...")
    # TODO: Uncomment when run_simulation is implemented
    # result = run_simulation(config, key)

    # print(f"  Winner: {result.winner}")
    # print(f"  Completion time: {result.completion_time:.2f}s")

    # ========================================================================
    # 5. Create output directory
    # ========================================================================
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nCreating output directory: {output_path}")

    # Create subdirectories
    figures_dir = output_path / "figures"
    figures_dir.mkdir(exist_ok=True)

    text_dir = output_path / "text"
    text_dir.mkdir(exist_ok=True)

    # ========================================================================
    # 6. Save trajectories
    # ========================================================================
    print("\nSaving trajectories...")
    # TODO: Uncomment when visualization functions are implemented
    # save_trajectories_csv(
    #     result.trajectory_D,
    #     result.trajectory_I,
    #     str(text_dir / "trajectories.csv")
    # )
    # print(f"  Saved to: {text_dir / 'trajectories.csv'}")

    # ========================================================================
    # 7. Save metrics
    # ========================================================================
    print("\nSaving metrics...")
    # TODO: Uncomment when implemented
    # save_metrics_csv(
    #     result.metrics,
    #     str(text_dir / "metrics.csv")
    # )
    # print(f"  Saved to: {text_dir / 'metrics.csv'}")

    # Print metrics summary
    # for metric_name, value in result.metrics.items():
    #     print(f"  {metric_name}: {value:.4f}")

    # ========================================================================
    # 8. Save belief history
    # ========================================================================
    print("\nSaving belief history...")
    # TODO: Uncomment when implemented
    # save_belief_history_csv(
    #     result.belief_history,
    #     result.trajectory_I.times,
    #     str(text_dir / "belief_history.csv")
    # )
    # print(f"  Saved to: {text_dir / 'belief_history.csv'}")

    # ========================================================================
    # 9. Generate visualizations
    # ========================================================================
    print("\nGenerating visualizations...")

    # Plot trajectories
    # TODO: Uncomment when implemented
    # plot_trajectories(
    #     workspace=config.workspace,
    #     trajectory_D=result.trajectory_D,
    #     trajectory_I=result.trajectory_I,
    #     candidate_goals=[jnp.array(g) for g in config.deceptive_agent.candidate_goals],
    #     true_goal=jnp.array(config.deceptive_agent.true_goal),
    #     save_path=str(figures_dir / "trajectories.png")
    # )
    # print(f"  Saved trajectory plot to: {figures_dir / 'trajectories.png'}")

    # Plot belief evolution
    # TODO: Uncomment when implemented
    # true_goal_id = config.deceptive_agent.candidate_goals.index(
    #     config.deceptive_agent.true_goal
    # )
    # plot_belief_evolution(
    #     result.belief_history,
    #     result.trajectory_I.times,
    #     true_goal_id,
    #     save_path=str(figures_dir / "belief_evolution.png")
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
