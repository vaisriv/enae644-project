# Interceptor Agent Specification (`src/interceptor/`)

## Purpose

Implements Agent I: an interceptor agent that infers the deceptive agent's hidden goal and plans interception trajectories using IRL, particle filtering, and game-theoretic MPC.

## Package Modules

- `irl.py`: Inverse reinforcement learning
- `particle_filter.py`: Belief distribution tracking
- `mpc.py`: Game-theoretic MPC planner
- `belief_update.py`: Bayesian belief update utilities

---

## Module: `irl.py` - Inverse Reinforcement Learning

### Learned Reward Function

```python
import equinox as eqx
import jax.numpy as jnp

class LearnedRewardFunction(eqx.Module):
    """Neural network parameterization of learned reward."""
    layers: List[eqx.nn.Linear]

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, key: PRNGKey):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(state_dim + action_dim, hidden_dim, key=keys[0]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            eqx.nn.Linear(hidden_dim, 1, key=keys[2])
        ]

    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> float:
        """
        Compute reward for state-action pair.

        Args:
            state: (2,) position
            action: (2,) control action

        Returns:
            Scalar reward
        """
        x = jnp.concatenate([state, action])
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x).squeeze()
```

### Maximum Entropy IRL

```python
def maximum_entropy_irl(
    demonstrations: List[Trajectory],
    goals: jnp.ndarray,              # (num_demos, 2)
    config: IRLConfig,
    key: PRNGKey
) -> LearnedRewardFunction:
    """
    Learn reward function using Maximum Entropy IRL.

    Algorithm:
        1. Initialize reward function θ
        2. For each epoch:
            a. Compute policy π_θ from reward (soft value iteration)
            b. Compute expected features under π_θ
            c. Compute empirical features from demonstrations
            d. Update θ: ∇θ = empirical_features - expected_features
    """
    import optax

    # Initialize reward function
    reward_fn = LearnedRewardFunction(
        state_dim=2, action_dim=2,
        hidden_dim=config.hidden_dim,
        key=key
    )

    # Optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(reward_fn, eqx.is_array))

    @eqx.filter_jit
    def compute_value_function(reward_fn, workspace, goal):
        """Soft value iteration to compute V(s)."""
        # Discretize state space
        # For each state, compute V(s) = log ∑_a exp(R(s,a) + γV(s'))
        # Return value function
        pass  # TODO: Implementation detail

    # Training loop
    for epoch in range(config.num_epochs):
        # Compute empirical feature expectations
        empirical_features = compute_empirical_features(demonstrations)

        # Compute expected features under current policy
        expected_features = jnp.zeros_like(empirical_features)
        for goal in goals:
            value_fn = compute_value_function(reward_fn, workspace, goal)
            expected_features += compute_expected_features(value_fn, workspace)
        expected_features /= len(goals)

        # Gradient update
        gradient = empirical_features - expected_features
        # (Use gradient to update reward_fn parameters)

    return reward_fn
```

### Trajectory Prediction

```python
def predict_trajectory(
    reward_fn: LearnedRewardFunction,
    start: jnp.ndarray,
    goal: jnp.ndarray,
    horizon: int,
    workspace: Workspace
) -> Trajectory:
    """
    Predict trajectory under learned reward function.

    Uses model-predictive control with learned reward as cost.
    """
    positions = [start]
    current_state = start

    for t in range(horizon):
        # Solve one-step optimization: argmax_a R(s, a)
        # subject to collision-free constraint
        best_action = None
        best_reward = -jnp.inf

        # Discretize action space (simple approach)
        for angle in jnp.linspace(0, 2*jnp.pi, 16):
            action = jnp.array([jnp.cos(angle), jnp.sin(angle)]) * 0.5
            next_state = current_state + action

            if is_collision_free(next_state, workspace):
                reward = reward_fn(current_state, action)
                if reward > best_reward:
                    best_reward = reward
                    best_action = action

        current_state = current_state + best_action
        positions.append(current_state)

    return create_trajectory(jnp.array(positions))
```

---

## Module: `particle_filter.py` - Particle Filter

```python
@dataclass
class Particle:
    """Particle representing goal hypothesis."""
    goal_id: int
    weight: float

class ParticleFilter:
    """Particle filter for goal inference."""

    def __init__(
        self,
        num_particles: int,
        candidate_goals: jnp.ndarray,    # (num_goals, 2)
        learned_model: LearnedRewardFunction,
        key: PRNGKey
    ):
        self.num_particles = num_particles
        self.goals = candidate_goals
        self.model = learned_model

        # Initialize particles uniformly over goals
        self.particles = []
        for i in range(num_particles):
            goal_id = i % len(candidate_goals)
            self.particles.append(Particle(goal_id, 1.0 / num_particles))

    def update(self, observation: jnp.ndarray, key: PRNGKey):
        """
        Update particle weights based on observation.

        Args:
            observation: (2,) latest observed position of Agent D
        """
        # Reweight particles based on likelihood
        for p in self.particles:
            likelihood = self._compute_likelihood(observation, p.goal_id)
            p.weight *= likelihood

        # Normalize weights
        total_weight = sum(p.weight for p in self.particles)
        for p in self.particles:
            p.weight /= total_weight

        # Resample if needed
        ess = self._effective_sample_size()
        if ess < self.num_particles * 0.5:
            self._resample(key)

    def _compute_likelihood(self, obs: jnp.ndarray, goal_id: int) -> float:
        """
        Compute P(obs | goal).

        Assumes observation is likely if it's consistent with
        trajectory predicted by learned model toward this goal.
        """
        # Simplified: Use reward as proxy for likelihood
        # Higher reward → more likely observation
        goal = self.goals[goal_id]
        action_toward_goal = goal - obs
        action_toward_goal = action_toward_goal / jnp.linalg.norm(action_toward_goal)

        reward = self.model(obs, action_toward_goal)
        return jnp.exp(reward)  # Convert to probability

    def _effective_sample_size(self) -> float:
        """Compute effective sample size: 1 / Σ w²."""
        return 1.0 / sum(p.weight**2 for p in self.particles)

    def _resample(self, key: PRNGKey):
        """Resample particles proportional to weights."""
        weights = jnp.array([p.weight for p in self.particles])
        indices = jax.random.choice(key, len(self.particles), shape=(len(self.particles),), p=weights)

        new_particles = []
        for idx in indices:
            new_particles.append(Particle(
                goal_id=self.particles[idx].goal_id,
                weight=1.0 / len(self.particles)
            ))
        self.particles = new_particles

    def estimate_goal(self) -> Tuple[int, float]:
        """
        Return MAP estimate and confidence.

        Returns:
            (goal_id, confidence)
        """
        # Count particles for each goal
        goal_counts = jnp.zeros(len(self.goals))
        for p in self.particles:
            goal_counts = goal_counts.at[p.goal_id].add(p.weight)

        best_goal_id = int(jnp.argmax(goal_counts))
        confidence = goal_counts[best_goal_id]
        return best_goal_id, confidence

    def get_belief_distribution(self) -> jnp.ndarray:
        """Return belief distribution over goals."""
        belief = jnp.zeros(len(self.goals))
        for p in self.particles:
            belief = belief.at[p.goal_id].add(p.weight)
        return belief
```

---

## Module: `mpc.py` - Game-Theoretic MPC

```python
def game_theoretic_mpc(
    current_state: jnp.ndarray,      # (2,) Agent I position
    belief: jnp.ndarray,             # (num_goals,) belief distribution
    agent_D_position: jnp.ndarray,   # (2,) current Agent D position
    learned_model: LearnedRewardFunction,
    horizon: int,
    config: MPCConfig,
    key: PRNGKey
) -> jnp.ndarray:                    # (2,) control action
    """
    Compute optimal control using game-theoretic MPC.

    Algorithm:
        1. Predict Agent D's trajectory for each goal hypothesis
        2. Compute expected trajectory weighted by belief
        3. Solve MPC optimization to minimize distance to expected trajectory
    """
    # Predict Agent D trajectories for each goal
    predicted_trajs = {}
    for goal_id, goal in enumerate(learned_model.goals):
        predicted_trajs[goal_id] = predict_trajectory(
            learned_model, agent_D_position, goal, horizon, workspace
        )

    # Compute expected trajectory
    expected_traj_positions = jnp.zeros((horizon, 2))
    for goal_id in range(len(belief)):
        expected_traj_positions += belief[goal_id] * predicted_trajs[goal_id].positions

    # Solve MPC optimization
    controls = solve_mpc_optimization(
        current_state, expected_traj_positions, horizon, config
    )

    return controls[0]  # Return first control action


def solve_mpc_optimization(
    initial_state: jnp.ndarray,
    target_trajectory: jnp.ndarray,  # (horizon, 2)
    horizon: int,
    config: MPCConfig
) -> jnp.ndarray:                    # (horizon, 2)
    """
    Solve MPC optimization problem.

    Objective:
        min Σ ||x[t] - target[t]||² + λ ||u[t]||²
        subject to: kinodynamic constraints
    """
    import optax

    @jax.jit
    def cost_fn(controls):
        # Rollout trajectory
        positions = rollout_trajectory(initial_state, controls, config.dt)

        # Terminal cost: distance to target
        terminal_cost = jnp.sum((positions - target_trajectory)**2)

        # Control cost
        control_cost = config.control_weight * jnp.sum(controls**2)

        return terminal_cost + control_cost

    # Initialize controls to zero
    controls_init = jnp.zeros((horizon, 2))

    # Optimize using gradient descent
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(controls_init)

    controls = controls_init
    for _ in range(config.max_iterations):
        loss, grads = jax.value_and_grad(cost_fn)(controls)
        updates, opt_state = optimizer.update(grads, opt_state)
        controls = optax.apply_updates(controls, updates)

    return controls


@jax.jit
def rollout_trajectory(
    initial_state: jnp.ndarray,
    controls: jnp.ndarray,          # (horizon, 2)
    dt: float
) -> jnp.ndarray:                   # (horizon, 2) positions
    """Rollout trajectory given control sequence."""
    positions = []
    state = initial_state

    for control in controls:
        state = state + control * dt
        positions.append(state)

    return jnp.array(positions)
```

---

## Module: `belief_update.py` - Belief Update Utilities

```python
@jax.jit
def bayesian_update(
    prior: jnp.ndarray,              # (num_goals,)
    likelihoods: jnp.ndarray         # (num_goals,)
) -> jnp.ndarray:                    # (num_goals,)
    """
    Bayesian belief update: posterior ∝ likelihood × prior.
    """
    posterior = prior * likelihoods
    return posterior / jnp.sum(posterior)


def compute_likelihood(
    observation: jnp.ndarray,
    goal_hypothesis: jnp.ndarray,
    learned_model: LearnedRewardFunction
) -> float:
    """
    Compute P(observation | goal_hypothesis).

    Uses learned model to evaluate how consistent observation is
    with expected behavior toward this goal.
    """
    # Direction toward goal
    direction = goal_hypothesis - observation
    direction = direction / jnp.linalg.norm(direction)

    # Reward for moving toward goal
    reward = learned_model(observation, direction)

    # Convert to likelihood (higher reward → higher probability)
    return jnp.exp(reward)
```

---

## Configuration

```python
@dataclass
class IRLConfig:
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 100

@dataclass
class MPCConfig:
    horizon: int = 20
    control_weight: float = 0.01
    learning_rate: float = 0.1
    max_iterations: int = 100
    dt: float = 0.1
```

## Testing

- Unit test: IRL recovers true reward on simple scenarios
- Unit test: Particle filter converges to true goal given enough observations
- Unit test: MPC solves simple interception problem
- Integration test: Full interception pipeline

## Navigation

**Previous**: [`05-deceptive-agent.md`](./05-deceptive-agent.md)

**Next**: [`07-simulation-controller.md`](./07-simulation-controller.md)
