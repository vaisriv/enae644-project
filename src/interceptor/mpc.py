"""Game-theoretic MPC for interception planning."""

import jax
import jax.numpy as jnp
import optax

from src.interceptor.irl import LearnedRewardFunction, predict_trajectory
from src.shared.workspace import Workspace
from src.simulation.config import MPCConfig


def game_theoretic_mpc(
    current_state: jnp.ndarray,
    belief: jnp.ndarray,
    agent_D_position: jnp.ndarray,
    learned_model: LearnedRewardFunction,
    goals: jnp.ndarray,
    workspace: Workspace,
    horizon: int,
    config: MPCConfig,
    key,
) -> jnp.ndarray:
    """Compute optimal control using game-theoretic MPC.

    1. Predict Agent D's trajectory for each goal hypothesis.
    2. Compute the belief-weighted expected trajectory.
    3. Solve a receding-horizon optimisation to minimise distance to that
       expected trajectory while penalising large controls.

    Args:
        current_state: (2,) current position of Agent I
        belief: (num_goals,) belief distribution over goals
        agent_D_position: (2,) current observed position of Agent D
        learned_model: Trained LearnedRewardFunction
        goals: (num_goals, 2) candidate goal positions
        workspace: Environment
        horizon: MPC prediction horizon
        config: MPCConfig hyperparameters
        key: JAX PRNG key (unused but kept for API consistency)

    Returns:
        (2,) control action for the current timestep
    """
    num_goals = goals.shape[0]

    # Predict Agent D trajectory for each goal
    predicted_positions = []
    for goal_id in range(num_goals):
        pred_traj = predict_trajectory(
            learned_model,
            agent_D_position,
            goals[goal_id],
            horizon,
            workspace,
        )
        # Pad or trim to exactly `horizon` steps
        pos = pred_traj.positions
        if pos.shape[0] < horizon:
            pad = jnp.tile(pos[-1:], (horizon - pos.shape[0], 1))
            pos = jnp.concatenate([pos, pad], axis=0)
        else:
            pos = pos[:horizon]
        predicted_positions.append(pos)

    # Belief-weighted expected trajectory
    expected_traj = jnp.zeros((horizon, 2))
    for goal_id in range(num_goals):
        expected_traj = expected_traj + belief[goal_id] * predicted_positions[goal_id]

    # Optimise controls
    controls = solve_mpc_optimization(current_state, expected_traj, horizon, config)

    return controls[0]


def solve_mpc_optimization(
    initial_state: jnp.ndarray,
    target_trajectory: jnp.ndarray,
    horizon: int,
    config: MPCConfig,
) -> jnp.ndarray:
    """Solve the MPC optimisation via Adam gradient descent.

    Objective: min_u Σ_t ||x[t] - target[t]||² + λ ||u[t]||²
    where x[t+1] = x[t] + u[t] · dt

    Args:
        initial_state: (2,) starting position
        target_trajectory: (horizon, 2) target positions
        horizon: Number of steps
        config: MPCConfig with learning_rate, control_weight, max_iterations, dt

    Returns:
        (horizon, 2) optimised control sequence
    """
    dt = config.dt
    lam = config.control_weight
    controls = jnp.zeros((horizon, 2))

    @jax.jit
    def cost_fn(controls: jnp.ndarray) -> jnp.ndarray:
        positions = _rollout(initial_state, controls, dt)
        tracking_cost = jnp.sum((positions - target_trajectory) ** 2)
        control_cost = lam * jnp.sum(controls**2)
        return tracking_cost + control_cost

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(controls)

    for _ in range(config.max_iterations):
        loss, grads = jax.value_and_grad(cost_fn)(controls)
        updates, opt_state = optimizer.update(grads, opt_state)
        controls = jnp.asarray(optax.apply_updates(controls, updates))
        # Clip to reasonable velocity magnitude
        controls = jnp.clip(controls, -2.0, 2.0)

    return controls


@jax.jit
def _rollout(
    initial_state: jnp.ndarray,
    controls: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Rollout trajectory via Euler integration.

    Args:
        initial_state: (2,) starting position
        controls: (horizon, 2) velocity controls
        dt: Timestep

    Returns:
        (horizon, 2) positions
    """

    def step(state, control):
        next_state = state + control * dt
        return next_state, next_state

    _, positions = jax.lax.scan(step, initial_state, controls)
    return positions
