# Deceptive Agent Specification (`src/deceptive/`)

## Purpose

Implements Agent D: a deceptive motion planner that generates trajectories concealing its true goal using Adversarial RRT\* with a learned RNN observer.

## Package Modules

- `planner.py`: Adversarial RRT\* implementation
- `observer.py`: RNN surrogate observer network
- `deception_cost.py`: Deception cost evaluation
- `tree.py`: RRT\* tree data structure

---

## Module: `planner.py` - Adversarial RRT\*

### Main Algorithm

```python
def adversarial_rrt_star(
    start: jnp.ndarray,              # (2,) starting position
    goal: jnp.ndarray,               # (2,) goal position
    workspace: Workspace,            # Environment
    observer_net: TrajectoryClassifier,  # Trained RNN
    alpha: float,                    # Deception weight [0, 1]
    config: RRTConfig,               # Hyperparameters
    key: PRNGKey,                    # JAX random key
) -> Trajectory:
    """
    Plan deceptive trajectory using Adversarial RRT*.

    Algorithm:
        Standard RRT* with modified cost function:
        cost(path) = α · path_length + (1-α) · deception_cost
    """
    tree = RRTTree()
    tree.add_node(start, parent_id=None, cost=0.0)

    for iteration in range(config.max_iterations):
        # 1. Sample configuration
        key, sample_key = jax.random.split(key)
        x_sample = sample_configuration(workspace, goal, config, sample_key)

        # 2. Find nearest node
        nearest_id = tree.find_nearest(x_sample)
        x_nearest = tree.nodes[nearest_id].position

        # 3. Steer toward sample
        x_new = steer(x_nearest, x_sample, config.step_size)

        # 4. Check collision
        if not line_segment_collision(x_nearest, x_new, workspace):
            # 5. Find near nodes for rewiring
            near_ids = tree.find_near(x_new, rewiring_radius(len(tree.nodes), config))

            # 6. Choose best parent
            parent_id, min_cost = choose_parent(
                tree, x_new, near_ids, goal, observer_net, alpha
            )

            # 7. Add node to tree
            new_id = tree.add_node(x_new, parent_id, min_cost)

            # 8. Rewire tree
            rewire(tree, new_id, near_ids, goal, observer_net, alpha)

            # 9. Check goal reached
            if in_goal_region(x_new, goal, config.goal_radius):
                path = tree.extract_path(new_id)
                return path_to_trajectory(path)

    # Return best path found
    best_id = find_closest_to_goal(tree, goal)
    path = tree.extract_path(best_id)
    return path_to_trajectory(path)
```

### Helper Functions

```python
def sample_configuration(
    workspace: Workspace,
    goal: jnp.ndarray,
    config: RRTConfig,
    key: PRNGKey
) -> jnp.ndarray:
    """
    Sample configuration with goal biasing.

    With probability goal_bias_probability, sample near goal.
    Otherwise, sample uniformly from workspace.
    """
    key, subkey = jax.random.split(key)
    if jax.random.uniform(subkey) < config.goal_bias_probability:
        # Sample near goal
        return goal + jax.random.normal(key, shape=(2,)) * 0.5
    else:
        # Sample uniformly from workspace
        return sample_free_position(workspace, key)


@jax.jit
def steer(x_from: jnp.ndarray, x_to: jnp.ndarray, step_size: float) -> jnp.ndarray:
    """
    Steer from x_from toward x_to by at most step_size.
    """
    direction = x_to - x_from
    distance = jnp.linalg.norm(direction)

    if distance <= step_size:
        return x_to
    else:
        return x_from + (direction / distance) * step_size


def choose_parent(
    tree: RRTTree,
    x_new: jnp.ndarray,
    near_ids: List[int],
    goal: jnp.ndarray,
    observer_net: TrajectoryClassifier,
    alpha: float
) -> Tuple[int, float]:
    """
    Choose parent that minimizes combined cost.

    Returns:
        (parent_id, min_cost)
    """
    min_cost = jnp.inf
    best_parent = near_ids[0]

    for near_id in near_ids:
        x_near = tree.nodes[near_id].position

        # Check if connection is collision-free
        if line_segment_collision(x_near, x_new, workspace):
            continue

        # Compute path from root to x_new via x_near
        path_to_near = tree.extract_path(near_id)
        candidate_path = jnp.vstack([path_to_near, x_new])

        # Compute combined cost
        cost = combined_cost(candidate_path, goal, observer_net, alpha)

        if cost < min_cost:
            min_cost = cost
            best_parent = near_id

    return best_parent, min_cost


def combined_cost(
    path: jnp.ndarray,           # (N, 2)
    goal: jnp.ndarray,           # (2,)
    observer_net: TrajectoryClassifier,
    alpha: float
) -> float:
    """
    Compute α · path_cost + (1-α) · deception_cost.
    """
    # Path cost (Euclidean length)
    J_path = compute_path_length_from_points(path)

    # Deception cost
    J_deception = evaluate_deception_cost(
        path, observer_net, goal_id, method="entropy"
    )

    return alpha * J_path + (1 - alpha) * J_deception


def rewiring_radius(num_nodes: int, config: RRTConfig) -> float:
    """
    Compute rewiring radius (shrinks as tree grows).

    r = min(γ · (log(n) / n)^(1/d), η)
    where d=2 (dimension), γ=config.gamma, η=config.max_radius
    """
    return min(
        config.gamma * (jnp.log(num_nodes) / num_nodes) ** 0.5,
        config.max_radius
    )
```

---

## Module: `observer.py` - RNN Observer Network

### Network Architecture

```python
import equinox as eqx
import jax
import jax.numpy as jnp

class TrajectoryClassifier(eqx.Module):
    """RNN-based trajectory classifier for goal prediction."""
    rnn: eqx.nn.GRU
    fc: eqx.nn.Linear

    def __init__(self, input_dim: int, hidden_dim: int, num_goals: int, key: PRNGKey):
        key1, key2 = jax.random.split(key)
        self.rnn = eqx.nn.GRU(input_dim, hidden_dim, key=key1)
        self.fc = eqx.nn.Linear(hidden_dim, num_goals, key=key2)

    def __call__(self, trajectory_sequence: jnp.ndarray) -> jnp.ndarray:
        """
        Classify trajectory to predict goal distribution.

        Args:
            trajectory_sequence: (seq_len, 2) partial trajectory

        Returns:
            (num_goals,) probability distribution over goals
        """
        # Process sequence through GRU
        hidden = jnp.zeros(self.rnn.hidden_size)
        for i in range(trajectory_sequence.shape[0]):
            hidden = self.rnn(trajectory_sequence[i], hidden)

        # Final classification from last hidden state
        logits = self.fc(hidden)
        return jax.nn.softmax(logits)
```

### Training

```python
def train_observer(
    dataset: TrajectoryDataset,
    config: TrainingConfig,
    key: PRNGKey
) -> TrajectoryClassifier:
    """
    Train RNN observer on trajectory classification.

    Dataset:
        trajectories: List of (T_i, 2) arrays
        goal_ids: (N,) integer labels

    Loss: Cross-entropy
    Optimizer: Adam
    """
    import optax

    # Initialize model
    model = TrajectoryClassifier(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        num_goals=config.num_goals,
        key=key
    )

    # Initialize optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(model, traj, goal_id):
        pred_probs = model(traj)
        return -jnp.log(pred_probs[goal_id] + 1e-8)  # Cross-entropy

    @eqx.filter_jit
    def train_step(model, opt_state, traj, goal_id):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, traj, goal_id)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Training loop
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for traj, goal_id in zip(dataset.trajectories, dataset.goal_ids):
            model, opt_state, loss = train_step(model, opt_state, traj, goal_id)
            total_loss += loss

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataset):.4f}")

    return model
```

### Loading from Checkpoint

```python
def load_observer(path: str, config: TrainingConfig) -> TrajectoryClassifier:
    """
    Load a trained observer from an Equinox checkpoint.

    Reconstructs the model skeleton from config, then deserialises weights.
    config.hidden_dim and config.num_goals must match the checkpoint.
    """
    skeleton = TrajectoryClassifier(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        num_goals=config.num_goals,
        key=jax.random.PRNGKey(0),  # key unused — overwritten by deserialisation
    )
    return eqx.tree_deserialise_leaves(path, skeleton)
```

---

## Module: `deception_cost.py` - Deception Cost Evaluator

```python
@jax.jit
def entropy_based_deception_cost(goal_probs: jnp.ndarray) -> float:
    """
    Compute negative entropy (higher = more revealing).

    H(p) = -Σ p(g) log p(g)
    deception_cost = -H(p)  (minimize → maximize entropy)
    """
    entropy = -jnp.sum(goal_probs * jnp.log(goal_probs + 1e-8))
    return -entropy  # Negate so minimizing cost maximizes uncertainty


@jax.jit
def accuracy_based_deception_cost(
    goal_probs: jnp.ndarray,
    true_goal_id: int
) -> float:
    """
    Return probability assigned to true goal.

    Minimizing this reduces observer's correct classification probability.
    """
    return goal_probs[true_goal_id]


def evaluate_deception_cost(
    partial_path: jnp.ndarray,       # (N, 2) positions
    observer_net: TrajectoryClassifier,
    true_goal_id: int,
    method: str = "entropy"          # "entropy" or "accuracy"
) -> float:
    """Evaluate deception cost for partial path."""
    # Get observer prediction
    goal_probs = observer_net(partial_path)

    if method == "entropy":
        return entropy_based_deception_cost(goal_probs)
    elif method == "accuracy":
        return accuracy_based_deception_cost(goal_probs, true_goal_id)
    else:
        raise ValueError(f"Unknown method: {method}")
```

---

## Module: `tree.py` - RRT\* Tree

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class RRTNode:
    """Node in RRT* tree."""
    position: jnp.ndarray  # (2,)
    parent_id: Optional[int]
    cost: float
    children: List[int]

class RRTTree:
    """RRT* search tree."""

    def __init__(self):
        self.nodes: List[RRTNode] = []

    def add_node(
        self,
        position: jnp.ndarray,
        parent_id: Optional[int],
        cost: float
    ) -> int:
        """Add node, return node ID."""
        node_id = len(self.nodes)
        node = RRTNode(position, parent_id, cost, children=[])
        self.nodes.append(node)

        if parent_id is not None:
            self.nodes[parent_id].children.append(node_id)

        return node_id

    def find_nearest(self, position: jnp.ndarray) -> int:
        """Find ID of nearest node."""
        distances = [jnp.linalg.norm(n.position - position) for n in self.nodes]
        return int(jnp.argmin(jnp.array(distances)))

    def find_near(self, position: jnp.ndarray, radius: float) -> List[int]:
        """Find all nodes within radius."""
        near_ids = []
        for i, node in enumerate(self.nodes):
            if jnp.linalg.norm(node.position - position) <= radius:
                near_ids.append(i)
        return near_ids

    def extract_path(self, node_id: int) -> jnp.ndarray:
        """Extract path from root to node_id."""
        path = []
        current_id = node_id

        while current_id is not None:
            path.append(self.nodes[current_id].position)
            current_id = self.nodes[current_id].parent_id

        return jnp.array(path[::-1])  # Reverse to get root→node order
```

---

## Configuration

```python
@dataclass
class RRTConfig:
    max_iterations: int = 5000
    step_size: float = 0.5
    goal_radius: float = 0.5
    goal_bias_probability: float = 0.1
    gamma: float = 2.0          # For rewiring radius
    max_radius: float = 3.0

@dataclass
class TrainingConfig:
    hidden_dim: int = 64
    num_goals: int = 3           # Set from SimulationConfig at runtime
    num_epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    samples_per_goal: int = 200
```

## Testing

- Unit test: RRT\* converges to optimal path when α=1 (pure path optimization)
- Unit test: Deception cost decreases observer accuracy when α=0
- Integration test: Full planning pipeline with trained observer

## Navigation

**Previous**: [`04-trajectory-representation.md`](./04-trajectory-representation.md)

**Next**: [`06-interceptor-agent.md`](./06-interceptor-agent.md)
