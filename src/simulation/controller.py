"""Main simulation controller for adversarial motion planning.

This module orchestrates the interaction between Deceptive Agent (Agent D)
and Interceptor Agent (Agent I), manages the simulation loop, and generates results.
"""

from dataclasses import dataclass
from typing import List, Dict
import jax
import jax.numpy as jnp

from src.simulation.config import SimulationConfig
from src.shared.trajectory import Trajectory, create_trajectory
from src.shared.workspace import Workspace
from src.shared.controller import AgentController, AgentState


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


def _check_goal_reached(
    position: jnp.ndarray, goal: jnp.ndarray, threshold: float
) -> bool:
    """Check if position is within threshold distance of goal.

    Args:
        position: (2,) current position
        goal: (2,) goal position
        threshold: Distance threshold

    Returns:
        True if ||position - goal|| < threshold
    """
    return jnp.linalg.norm(position - goal) < threshold


def _check_interception(
    pos_D: jnp.ndarray, pos_I: jnp.ndarray, threshold: float
) -> bool:
    """Check if Agent I has intercepted Agent D.

    Args:
        pos_D: (2,) Agent D position
        pos_I: (2,) Agent I position
        threshold: Interception distance threshold

    Returns:
        True if ||pos_D - pos_I|| < threshold
    """
    return jnp.linalg.norm(pos_D - pos_I) < threshold


def _create_trajectory_from_positions(
    times: jnp.ndarray, positions: jnp.ndarray
) -> Trajectory:
    """Create a Trajectory object from time-series positions.

    Args:
        times: (T,) array of timestamps
        positions: (T, 2) array of positions

    Returns:
        Trajectory with computed velocities
    """
    # Use create_trajectory which automatically computes velocities
    return create_trajectory(times, positions)


def run_game_with_controllers(
    workspace: Workspace,
    controller_D: AgentController,
    controller_I: AgentController,
    initial_state_D: AgentState,
    initial_state_I: AgentState,
    goal_D: jnp.ndarray,
    max_time: float = 100.0,
    dt: float = 0.1,
    intercept_threshold: float = 0.5,
    goal_radius: float = 0.5,
) -> SimulationResult:
    """Run game simulation with two agent controllers.

    This is a simplified game runner that works with the AgentController interface.
    Both agents use their controllers to compute actions, and the simulation
    integrates their motion and checks termination conditions.

    Args:
        workspace: Workspace with bounds and obstacles
        controller_D: Controller for deceptive agent (Agent D)
        controller_I: Controller for interceptor agent (Agent I)
        initial_state_D: Initial state of Agent D
        initial_state_I: Initial state of Agent I
        goal_D: Goal position for Agent D (2,) array
        max_time: Maximum simulation time (default: 100.0)
        dt: Timestep for simulation (default: 0.1)
        intercept_threshold: Distance for interception (default: 0.5)
        goal_radius: Distance to consider goal reached (default: 0.5)

    Returns:
        SimulationResult containing trajectories, winner, and placeholder metrics

    Example:
        >>> workspace = create_workspace(...)
        >>> controller_D = SimpleGoalController(goal=jnp.array([10.0, 10.0]))
        >>> controller_I = SimpleGoalController(goal=jnp.array([5.0, 5.0]))
        >>> result = run_game_with_controllers(workspace, controller_D, controller_I, ...)
    """
    # ========================================================================
    # 1. Initialize controllers
    # ========================================================================
    controller_D.reset(initial_state_D, workspace)
    controller_I.reset(initial_state_I, workspace)

    # ========================================================================
    # 2. Initialize simulation state
    # ========================================================================
    t = 0.0
    state_D = initial_state_D
    state_I = initial_state_I

    # Logging arrays
    times_log = [t]
    positions_log_D = [state_D.position]
    positions_log_I = [state_I.position]
    # Add initial belief (uniform distribution over 2 goals)
    belief_history = [jnp.array([0.5, 0.5])]

    winner = "timeout"

    # ========================================================================
    # 3. Main simulation loop
    # ========================================================================
    while t < max_time:
        # --------------------------------------------------------------------
        # Agent D: Compute control
        # --------------------------------------------------------------------
        command_D = controller_D.compute_control(state_D)

        # --------------------------------------------------------------------
        # Agent I: Compute control (with observation of Agent D)
        # --------------------------------------------------------------------
        observation = {"opponent_position": state_D.position}
        command_I = controller_I.compute_control(state_I, observation)

        # --------------------------------------------------------------------
        # Integrate motion
        # --------------------------------------------------------------------
        # Use velocity control (simple kinematic integration)
        if command_D.velocity is not None:
            new_pos_D = state_D.position + command_D.velocity * dt
            new_vel_D = command_D.velocity
        else:
            new_pos_D = state_D.position
            new_vel_D = state_D.velocity

        if command_I.velocity is not None:
            new_pos_I = state_I.position + command_I.velocity * dt
            new_vel_I = command_I.velocity
        else:
            new_pos_I = state_I.position
            new_vel_I = state_I.velocity

        # Update states
        t += dt
        state_D = AgentState(position=new_pos_D, velocity=new_vel_D, time=t)
        state_I = AgentState(position=new_pos_I, velocity=new_vel_I, time=t)

        # Log positions
        times_log.append(t)
        positions_log_D.append(state_D.position)
        positions_log_I.append(state_I.position)

        # Placeholder belief (uniform distribution over 2 goals)
        belief_history.append(jnp.array([0.5, 0.5]))

        # --------------------------------------------------------------------
        # Check termination conditions
        # --------------------------------------------------------------------
        # Condition 1: Agent D reaches goal
        if _check_goal_reached(state_D.position, goal_D, goal_radius):
            winner = "Agent_D"
            break

        # Condition 2: Agent I intercepts Agent D
        if _check_interception(state_D.position, state_I.position, intercept_threshold):
            winner = "Agent_I"
            break

    # ========================================================================
    # 4. Create result
    # ========================================================================
    trajectory_D = _create_trajectory_from_positions(
        jnp.array(times_log), jnp.stack(positions_log_D)
    )
    trajectory_I = _create_trajectory_from_positions(
        jnp.array(times_log), jnp.stack(positions_log_I)
    )

    # Placeholder metrics
    metrics = {
        "completion_time": t,
        "distance_traveled_D": 0.0,  # TODO: compute from trajectory
        "distance_traveled_I": 0.0,  # TODO: compute from trajectory
        "final_distance": float(jnp.linalg.norm(state_D.position - state_I.position)),
    }

    return SimulationResult(
        winner=winner,
        completion_time=t,
        trajectory_D=trajectory_D,
        trajectory_I=trajectory_I,
        belief_history=belief_history,
        metrics=metrics,
    )
