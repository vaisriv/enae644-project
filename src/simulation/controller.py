"""Main simulation controller for adversarial motion planning.

This module orchestrates the interaction between Deceptive Agent (Agent D)
and Interceptor Agent (Agent I), manages the simulation loop, and generates results.
"""

from dataclasses import dataclass
from typing import List, Dict
import jax
import jax.numpy as jnp

from src.simulation.config import SimulationConfig
from src.shared.trajectory import Trajectory


@dataclass
class SimulationResult:
    """Result of a complete simulation run.

    Attributes:
        winner: "Agent_D", "Agent_I", or "timeout"
        completion_time: Time when simulation ended
        trajectory_D: Complete trajectory of deceptive agent
        trajectory_I: Complete trajectory of interceptor agent
        belief_history: List of belief distributions over time
        metrics: Dictionary of performance metrics
    """
    winner: str
    completion_time: float
    trajectory_D: Trajectory
    trajectory_I: Trajectory
    belief_history: List[jnp.ndarray]
    metrics: Dict[str, float]


def run_simulation(config: SimulationConfig, key: jnp.ndarray) -> SimulationResult:
    """Run full adversarial motion planning simulation.

    This is the main entry point for running a complete simulation episode.

    Args:
        config: Complete simulation configuration
        key: JAX PRNG key for reproducibility

    Returns:
        SimulationResult containing trajectories, winner, and metrics

    Flow:
        1. Initialize workspace and agents from config
        2. Load pre-trained models (observer network, IRL model)
        3. Agent D plans full deceptive trajectory (offline, once)
        4. Initialize Agent I's particle filter
        5. Main simulation loop:
           - Agent D executes pre-planned trajectory
           - Agent I observes Agent D's position
           - Agent I updates belief via particle filter
           - Agent I plans control via game-theoretic MPC
           - Agent I executes control
           - Check termination conditions
        6. Compute metrics and generate result
    """
    # ========================================================================
    # 1. Initialize workspace and agents
    # ========================================================================
    workspace_config = config.workspace
    agent_D_config = config.deceptive_agent
    agent_I_config = config.interceptor_agent
    sim_params = config.simulation

    # TODO: Create workspace from config
    # workspace = create_workspace_from_config(workspace_config)
    raise NotImplementedError("Workspace creation not implemented yet")

    # ========================================================================
    # 2. Load pre-trained models
    # ========================================================================
    # TODO: Load observer RNN network
    # observer_net = load_observer_network(agent_D_config.observer.checkpoint_path)

    # TODO: Load IRL reward model
    # irl_model = load_irl_model(agent_I_config.irl.checkpoint_path)

    # ========================================================================
    # 3. Agent D plans full trajectory (offline)
    # ========================================================================
    key, plan_key = jax.random.split(key)

    # TODO: Call adversarial RRT* planner
    # trajectory_D = adversarial_rrt_star(
    #     start=jnp.array(agent_D_config.initial_position),
    #     goal=jnp.array(agent_D_config.true_goal),
    #     workspace=workspace,
    #     observer_net=observer_net,
    #     alpha=agent_D_config.planner.deception_weight,
    #     config=agent_D_config.planner,
    #     key=plan_key
    # )

    # ========================================================================
    # 4. Initialize Agent I's particle filter
    # ========================================================================
    key, pf_key = jax.random.split(key)

    # TODO: Create particle filter instance
    # particle_filter = ParticleFilter(
    #     num_particles=agent_I_config.particle_filter.num_particles,
    #     candidate_goals=jnp.array(agent_I_config.candidate_goals),
    #     learned_model=irl_model,
    #     key=pf_key
    # )

    # ========================================================================
    # 5. Main simulation loop
    # ========================================================================
    t = 0.0
    dt = sim_params.timestep
    max_time = sim_params.max_time

    # Initialize Agent I state
    x_I = jnp.array(agent_I_config.initial_position)

    # Logging arrays
    times_log = [0.0]
    trajectory_log_D = []  # Will collect Agent D positions
    trajectory_log_I = [x_I]  # Agent I positions
    belief_history = []  # Belief distributions over time

    winner = "timeout"  # Default if no termination condition is met

    while t < max_time:
        # --------------------------------------------------------------------
        # Agent D: Execute pre-planned trajectory
        # --------------------------------------------------------------------
        # TODO: Interpolate position from trajectory at current time t
        # x_D = interpolate_position(trajectory_D, t)
        # trajectory_log_D.append(x_D)

        # --------------------------------------------------------------------
        # Agent I: Observe Agent D
        # --------------------------------------------------------------------
        # TODO: Update particle filter with observation
        # key, update_key = jax.random.split(key)
        # particle_filter.update(x_D, update_key)
        # belief = particle_filter.get_belief_distribution()
        # belief_history.append(belief)

        # --------------------------------------------------------------------
        # Agent I: Plan control via game-theoretic MPC
        # --------------------------------------------------------------------
        # TODO: Compute control action
        # key, mpc_key = jax.random.split(key)
        # u_I = game_theoretic_mpc(
        #     current_state=x_I,
        #     belief=belief,
        #     agent_D_position=x_D,
        #     learned_model=irl_model,
        #     horizon=agent_I_config.mpc.horizon,
        #     config=agent_I_config.mpc,
        #     key=mpc_key
        # )

        # --------------------------------------------------------------------
        # Agent I: Execute control
        # --------------------------------------------------------------------
        # TODO: Integrate motion with kinematic constraints
        # x_I = integrate_motion(x_I, u_I, dt)
        # trajectory_log_I.append(x_I)

        # --------------------------------------------------------------------
        # Check termination conditions
        # --------------------------------------------------------------------
        true_goal = jnp.array(agent_D_config.true_goal)

        # Condition 1: Agent D reaches goal
        # TODO: Uncomment when x_D is available
        # if jnp.linalg.norm(x_D - true_goal) < sim_params.goal_radius:
        #     winner = "Agent_D"
        #     break

        # Condition 2: Agent I intercepts Agent D
        # TODO: Uncomment when x_D is available
        # if jnp.linalg.norm(x_D - x_I) < sim_params.intercept_threshold:
        #     winner = "Agent_I"
        #     break

        # Advance time
        t += dt
        times_log.append(t)

    # ========================================================================
    # 6. Generate results and compute metrics
    # ========================================================================
    # TODO: Convert logged positions to Trajectory objects
    # final_trajectory_D = create_trajectory_from_positions(
    #     jnp.array(times_log[:len(trajectory_log_D)]),
    #     jnp.array(trajectory_log_D)
    # )
    # final_trajectory_I = create_trajectory_from_positions(
    #     jnp.array(times_log),
    #     jnp.array(trajectory_log_I)
    # )

    # TODO: Compute all performance metrics
    # metrics = compute_all_metrics(
    #     trajectory_D=final_trajectory_D,
    #     trajectory_I=final_trajectory_I,
    #     belief_history=belief_history,
    #     observer_net=observer_net,
    #     true_goal=true_goal
    # )

    # For now, return placeholder result
    raise NotImplementedError(
        "Main simulation loop not fully implemented. "
        "Need to implement: workspace creation, model loading, "
        "adversarial RRT*, particle filter, MPC planning, and metrics."
    )

    # return SimulationResult(
    #     winner=winner,
    #     completion_time=t,
    #     trajectory_D=final_trajectory_D,
    #     trajectory_I=final_trajectory_I,
    #     belief_history=belief_history,
    #     metrics=metrics
    # )


def _check_goal_reached(position: jnp.ndarray, goal: jnp.ndarray, threshold: float) -> bool:
    """Check if position is within threshold distance of goal.

    Args:
        position: (2,) current position
        goal: (2,) goal position
        threshold: Distance threshold

    Returns:
        True if ||position - goal|| < threshold
    """
    # TODO: Implement
    # return jnp.linalg.norm(position - goal) < threshold
    raise NotImplementedError("_check_goal_reached not implemented")


def _check_interception(pos_D: jnp.ndarray, pos_I: jnp.ndarray, threshold: float) -> bool:
    """Check if Agent I has intercepted Agent D.

    Args:
        pos_D: (2,) Agent D position
        pos_I: (2,) Agent I position
        threshold: Interception distance threshold

    Returns:
        True if ||pos_D - pos_I|| < threshold
    """
    # TODO: Implement
    # return jnp.linalg.norm(pos_D - pos_I) < threshold
    raise NotImplementedError("_check_interception not implemented")


def _create_trajectory_from_positions(
    times: jnp.ndarray,
    positions: jnp.ndarray
) -> Trajectory:
    """Create a Trajectory object from time-series positions.

    Args:
        times: (T,) array of timestamps
        positions: (T, 2) array of positions

    Returns:
        Trajectory with computed velocities
    """
    # TODO: Implement velocity computation via finite differences
    # dt = jnp.diff(times)
    # dpos = jnp.diff(positions, axis=0)
    # velocities = dpos / dt[:, None]
    # # Append last velocity
    # velocities = jnp.concatenate([velocities, velocities[-1:]], axis=0)
    # return Trajectory(times=times, positions=positions, velocities=velocities)
    raise NotImplementedError("_create_trajectory_from_positions not implemented")
