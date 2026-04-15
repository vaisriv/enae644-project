"""Demo script showing the complete game simulation pipeline.

This script demonstrates:
1. Creating a workspace with obstacles
2. Setting up agent controllers
3. Running the game simulation
4. Generating visualizations and data exports
"""

from pathlib import Path
import jax.numpy as jnp

from src.shared.workspace import Workspace, CircleObstacle, PolygonObstacle
from src.shared.controller import SimpleGoalController, AgentState
from src.simulation.controller import run_game_with_controllers
from src.simulation.visualization import (
    plot_trajectories,
    plot_belief_evolution,
    plot_distance_over_time,
    create_animation,
    save_trajectories_csv,
    save_metrics_csv,
)


def create_demo_workspace() -> Workspace:
    """Create a demo workspace with bounds and obstacles."""
    # Workspace bounds: 20m x 20m
    bounds = jnp.array([[0.0, 20.0], [0.0, 20.0]])

    # Create obstacles
    obstacles = [
        # Circle obstacle in the middle
        CircleObstacle(center=jnp.array([10.0, 10.0]), radius=2.0),
        # Polygon obstacle (triangle) in top-left
        PolygonObstacle(vertices=jnp.array([[2.0, 15.0], [5.0, 18.0], [2.0, 18.0]])),
        # Polygon obstacle (rectangle) in bottom-right
        PolygonObstacle(
            vertices=jnp.array([[15.0, 2.0], [18.0, 2.0], [18.0, 5.0], [15.0, 5.0]])
        ),
    ]

    return Workspace(bounds=bounds, obstacles=obstacles)


def demo_simple_chase():
    """Demo: Simple chase scenario where Agent I tries to intercept Agent D."""
    print("=" * 70)
    print("DEMO: Simple Chase Scenario")
    print("=" * 70)

    # Create workspace
    print("\n[1/7] Creating workspace...")
    workspace = create_demo_workspace()
    print(f"  Workspace: {workspace.bounds[0, 1]}m x {workspace.bounds[1, 1]}m")
    print(f"  Obstacles: {len(workspace.obstacles)}")

    # Setup controllers
    print("\n[2/7] Setting up controllers...")

    # Agent D: Moves toward goal at top-right at moderate speed
    goal_D = jnp.array([18.0, 18.0])
    controller_D = SimpleGoalController(goal=goal_D, max_speed=2.0)
    print(f"  Agent D: Goal = {goal_D}, Max Speed = 2.0 m/s")

    # Agent I: Tries to intercept Agent D, moving faster
    # For this demo, Agent I moves toward Agent D's starting position initially
    # (In a real scenario, it would use more sophisticated planning)
    goal_I = jnp.array([10.0, 10.0])  # Middle of workspace
    controller_I = SimpleGoalController(goal=goal_I, max_speed=2.5)
    print(f"  Agent I: Goal = {goal_I}, Max Speed = 2.5 m/s")

    # Initial states
    initial_state_D = AgentState(
        position=jnp.array([2.0, 2.0]), velocity=jnp.zeros(2), time=0.0
    )
    initial_state_I = AgentState(
        position=jnp.array([18.0, 2.0]), velocity=jnp.zeros(2), time=0.0
    )

    print(f"  Agent D Start: {initial_state_D.position}")
    print(f"  Agent I Start: {initial_state_I.position}")

    # Run simulation
    print("\n[3/7] Running simulation...")
    result = run_game_with_controllers(
        workspace=workspace,
        controller_D=controller_D,
        controller_I=controller_I,
        initial_state_D=initial_state_D,
        initial_state_I=initial_state_I,
        goal_D=goal_D,
        max_time=20.0,
        dt=0.1,
        intercept_threshold=0.5,
        goal_radius=0.5,
    )

    print(f"  Winner: {result.winner}")
    print(f"  Completion Time: {result.completion_time:.2f}s")
    print(f"  Trajectory D: {len(result.trajectory_D.positions)} points")
    print(f"  Trajectory I: {len(result.trajectory_I.positions)} points")

    # Create output directory
    print("\n[4/7] Creating output directory...")
    output_dir = Path("outputs/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "text").mkdir(exist_ok=True)
    print(f"  Output directory: {output_dir}")

    # Save data
    print("\n[5/7] Saving data...")

    # CSV exports
    save_trajectories_csv(
        result.trajectory_D,
        result.trajectory_I,
        str(output_dir / "text" / "trajectories.csv"),
    )
    print(f"  Saved: {output_dir / 'text' / 'trajectories.csv'}")

    save_metrics_csv(result.metrics, str(output_dir / "text" / "metrics.csv"))
    print(f"  Saved: {output_dir / 'text' / 'metrics.csv'}")

    # Generate visualizations
    print("\n[6/7] Generating visualizations...")

    # Plot trajectories
    candidate_goals = [goal_D, jnp.array([2.0, 18.0])]  # True goal + decoy
    plot_trajectories(
        workspace=workspace,
        trajectory_D=result.trajectory_D,
        trajectory_I=result.trajectory_I,
        candidate_goals=candidate_goals,
        true_goal=goal_D,
        save_path=str(output_dir / "figures" / "trajectories.png"),
    )
    print(f"  Saved: {output_dir / 'figures' / 'trajectories.png'}")

    # Plot belief evolution
    plot_belief_evolution(
        belief_history=result.belief_history,
        times=result.trajectory_I.times,
        true_goal_id=0,
        save_path=str(output_dir / "figures" / "belief_evolution.png"),
    )
    print(f"  Saved: {output_dir / 'figures' / 'belief_evolution.png'}")

    # Plot distance over time
    plot_distance_over_time(
        trajectory_D=result.trajectory_D,
        trajectory_I=result.trajectory_I,
        intercept_threshold=0.5,
        save_path=str(output_dir / "figures" / "distance_over_time.png"),
    )
    print(f"  Saved: {output_dir / 'figures' / 'distance_over_time.png'}")

    # Create animation
    print("\n[7/7] Creating animation...")
    create_animation(
        workspace=workspace,
        trajectory_D=result.trajectory_D,
        trajectory_I=result.trajectory_I,
        belief_history=result.belief_history,
        save_path=str(output_dir / "figures" / "animation.gif"),
        fps=10,
    )
    print(f"  Saved: {output_dir / 'figures' / 'animation.gif'}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


def demo_goal_race():
    """Demo: Race scenario where both agents compete to reach goals."""
    print("\n" + "=" * 70)
    print("DEMO: Goal Race Scenario")
    print("=" * 70)

    # Create workspace
    print("\n[1/5] Creating workspace...")
    workspace = create_demo_workspace()

    # Setup controllers - both racing to opposite corners
    print("\n[2/5] Setting up controllers...")

    goal_D = jnp.array([18.0, 18.0])
    goal_I = jnp.array([2.0, 18.0])

    controller_D = SimpleGoalController(goal=goal_D, max_speed=3.0)
    controller_I = SimpleGoalController(goal=goal_I, max_speed=3.0)

    print(f"  Agent D: Racing to {goal_D} at 3.0 m/s")
    print(f"  Agent I: Racing to {goal_I} at 3.0 m/s")

    # Both start from bottom corners
    initial_state_D = AgentState(
        position=jnp.array([2.0, 2.0]), velocity=jnp.zeros(2), time=0.0
    )
    initial_state_I = AgentState(
        position=jnp.array([18.0, 2.0]), velocity=jnp.zeros(2), time=0.0
    )

    # Run simulation
    print("\n[3/5] Running simulation...")
    result = run_game_with_controllers(
        workspace=workspace,
        controller_D=controller_D,
        controller_I=controller_I,
        initial_state_D=initial_state_D,
        initial_state_I=initial_state_I,
        goal_D=goal_D,
        max_time=15.0,
        dt=0.1,
        intercept_threshold=0.5,  # Very close for interception in race
        goal_radius=0.5,
    )

    print(f"  Winner: {result.winner}")
    print(f"  Completion Time: {result.completion_time:.2f}s")

    # Create output directory
    print("\n[4/5] Saving results...")
    output_dir = Path("outputs/demo_race")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    # Save trajectory plot
    print("\n[5/5] Generating visualization...")
    candidate_goals = [goal_D, goal_I]
    plot_trajectories(
        workspace=workspace,
        trajectory_D=result.trajectory_D,
        trajectory_I=result.trajectory_I,
        candidate_goals=candidate_goals,
        true_goal=goal_D,
        save_path=str(output_dir / "figures" / "race_trajectories.png"),
    )
    print(f"  Saved: {output_dir / 'figures' / 'race_trajectories.png'}")

    print("\n" + "=" * 70)
    print("RACE DEMO COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


def main():
    """Run all demo scenarios."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ADVERSARIAL MOTION PLANNING GAME DEMO" + " " * 16 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run demo scenarios
    demo_simple_chase()
    print("\n\n")
    demo_goal_race()

    print("\n\nAll demos completed successfully!")


if __name__ == "__main__":
    main()
