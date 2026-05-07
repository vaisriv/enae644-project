# Simulation Controller (`src/simulation/`)

## Purpose

Orchestrates the adversarial interaction between agents, manages simulation loop, and generates outputs.

## Modules

- `controller.py`: Main game loop
- `config.py`: Configuration loading
- `metrics.py`: Performance metrics
- `visualization.py`: Plotting

---

## Module: `controller.py` - Main Game Loop

```python
@dataclass
class SimulationResult:
    """Result of simulation run."""
    winner: str                      # "Agent_D", "Agent_I", or "timeout"
    completion_time: float
    trajectory_D: Trajectory
    trajectory_I: Trajectory
    belief_history: List[jnp.ndarray]  # Belief over time
    metrics: Dict[str, float]


def run_simulation(config: SimulationConfig, key: PRNGKey) -> SimulationResult:
    """
    Run full adversarial simulation.

    Flow:
        1. Initialize workspace and agents
        2. Agent D plans full trajectory (offline)
        3. Simulation loop:
            - Agent D executes trajectory
            - Agent I observes, updates belief, replans
            - Check termination conditions
        4. Generate outputs and metrics
    """
    # 1. Initialize
    workspace = config.workspace
    agent_D_config = config.deceptive_agent
    agent_I_config = config.interceptor_agent

    # Load trained models (checkpoints written by: uv run adversarial-train)
    observer_net = load_observer(
        agent_D_config.observer.checkpoint_path,
        config.training.observer,
    )
    irl_model = load_irl_model(
        agent_I_config.irl.checkpoint_path,
        config.training.irl,
    )

    # 2. Agent D plans trajectory
    key, plan_key = jax.random.split(key)
    traj_D = adversarial_rrt_star(
        start=jnp.array(agent_D_config.initial_position),
        goal=jnp.array(agent_D_config.true_goal),
        workspace=workspace,
        observer_net=observer_net,
        alpha=agent_D_config.planner.deception_weight,
        config=agent_D_config.planner,
        key=plan_key
    )

    # 3. Initialize Agent I
    particle_filter = ParticleFilter(
        num_particles=agent_I_config.particle_filter.num_particles,
        candidate_goals=jnp.array(agent_I_config.candidate_goals),
        learned_model=irl_model,
        key=key
    )

    # 4. Simulation loop
    t = 0.0
    dt = config.simulation.timestep
    x_I = jnp.array(agent_I_config.initial_position)

    trajectory_log_D = []
    trajectory_log_I = [x_I]
    belief_history = [particle_filter.get_belief_distribution()]

    while t < config.simulation.max_time:
        # Agent D executes trajectory
        x_D = interpolate_position(traj_D, t)
        trajectory_log_D.append(x_D)

        # Agent I observes and updates belief
        key, update_key = jax.random.split(key)
        particle_filter.update(x_D, update_key)
        belief_history.append(particle_filter.get_belief_distribution())

        # Agent I plans control
        belief = particle_filter.get_belief_distribution()
        key, mpc_key = jax.random.split(key)
        u_I = game_theoretic_mpc(
            x_I, belief, x_D, irl_model,
            agent_I_config.mpc.horizon,
            agent_I_config.mpc,
            mpc_key
        )

        # Agent I executes control
        x_I = integrate_motion(x_I, u_I, dt)
        trajectory_log_I.append(x_I)

        # Check termination
        true_goal = jnp.array(agent_D_config.true_goal)
        if jnp.linalg.norm(x_D - true_goal) < config.simulation.goal_radius:
            winner = "Agent_D"
            break

        if jnp.linalg.norm(x_D - x_I) < config.simulation.intercept_threshold:
            winner = "Agent_I"
            break

        t += dt
    else:
        winner = "timeout"

    # 5. Generate result
    final_traj_D = create_trajectory(jnp.array(trajectory_log_D))
    final_traj_I = create_trajectory(jnp.array(trajectory_log_I))

    metrics = compute_all_metrics(
        final_traj_D, final_traj_I, belief_history,
        observer_net, agent_D_config.true_goal
    )

    return SimulationResult(
        winner=winner,
        completion_time=t,
        trajectory_D=final_traj_D,
        trajectory_I=final_traj_I,
        belief_history=belief_history,
        metrics=metrics
    )
```

---

## Module: `metrics.py` - Performance Metrics

```python
def compute_all_metrics(
    traj_D: Trajectory,
    traj_I: Trajectory,
    belief_history: List[jnp.ndarray],
    observer_net: TrajectoryClassifier,
    true_goal: jnp.ndarray
) -> Dict[str, float]:
    """Compute all performance metrics."""

    metrics = {}

    # Observer accuracy
    final_obs_probs = observer_net(traj_D.positions)
    true_goal_id = 0  # TODO: Map true_goal to goal_id
    metrics['observer_accuracy_final'] = float(final_obs_probs[true_goal_id])

    # Path length ratio
    optimal_length = jnp.linalg.norm(traj_D.positions[-1] - traj_D.positions[0])
    actual_length = compute_path_length(traj_D)
    metrics['path_length_ratio'] = actual_length / optimal_length

    # Belief entropy
    final_entropy = -jnp.sum(belief_history[-1] * jnp.log(belief_history[-1] + 1e-8))
    metrics['belief_entropy_final'] = float(final_entropy)

    # Minimum interception distance
    min_dist = jnp.inf
    for i in range(min(len(traj_D.positions), len(traj_I.positions))):
        dist = jnp.linalg.norm(traj_D.positions[i] - traj_I.positions[i])
        min_dist = min(min_dist, dist)
    metrics['interception_distance_min'] = float(min_dist)

    return metrics
```

## Navigation

**Previous**: [`06-interceptor-agent.md`](./06-interceptor-agent.md)

**Next**: [`08-neural-network-training.md`](./08-neural-network-training.md)
