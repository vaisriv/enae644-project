# Package Structure and Organization

## Purpose

This document details the package organization, module responsibilities, import conventions, and public APIs for the adversarial motion planning system. The structure follows an **agent-based organization** principle, separating deceptive agent and interceptor agent code while sharing common functionality.

## Package Layout

```
src/
├── __init__.py                  # Root package initialization
├── index.py                     # Main entrypoint (existing, preserved)
│
├── deceptive/                   # Deceptive Agent (Agent D) components
│   ├── __init__.py              # Public API exports
│   ├── planner.py               # Adversarial RRT* implementation
│   ├── observer.py              # RNN surrogate observer network
│   ├── deception_cost.py        # Deception cost evaluation
│   └── tree.py                  # RRT* tree data structure
│
├── interceptor/                 # Interceptor Agent (Agent I) components
│   ├── __init__.py              # Public API exports
│   ├── irl.py                   # Inverse reinforcement learning
│   ├── particle_filter.py       # Belief distribution tracking
│   ├── mpc.py                   # Game-theoretic MPC planner
│   └── belief_update.py         # Bayesian belief update utilities
│
├── shared/                      # Shared components (both agents)
│   ├── __init__.py              # Public API exports
│   ├── workspace.py             # 2D environment and obstacles
│   ├── trajectory.py            # Trajectory representation
│   ├── collision.py             # Collision detection
│   ├── kinematics.py            # Kinodynamic constraints
│   └── geometry.py              # Geometric utilities
│
├── simulation/                  # Simulation framework
│   ├── __init__.py              # Public API exports
│   ├── controller.py            # Main game loop
│   ├── config.py                # Configuration loading and validation
│   ├── metrics.py               # Performance metric computation
│   └── visualization.py         # Plotting and output generation
│
└── data/                        # Data handling
    ├── __init__.py              # Public API exports
    ├── schemas.py               # Data structure definitions (pytrees)
    ├── loaders.py               # Training data loaders
    └── generators.py            # Synthetic data generation
```

## Module Responsibilities

### `src/deceptive/` - Deceptive Agent Package

#### `planner.py` - Adversarial RRT\* Planner

**Responsibility**: Implements Adversarial RRT\* algorithm with deception cost integration.

**Public API**:

```python
def adversarial_rrt_star(
    start: Array,                    # (2,) starting position
    goal: Array,                     # (2,) goal position
    workspace: Workspace,            # Workspace with obstacles
    observer_net: TrajectoryClassifier,  # RNN observer
    alpha: float,                    # Deception weight [0, 1]
    config: RRTConfig,              # RRT* hyperparameters
    key: PRNGKey,                   # JAX random key
) -> Trajectory:
    """Plan deceptive trajectory using Adversarial RRT*."""
    ...
```

**Dependencies**: `src/deceptive/{observer, deception_cost, tree}`, `src/shared/{workspace, collision, trajectory}`

#### `observer.py` - RNN Surrogate Observer

**Responsibility**: Neural network that classifies partial trajectories to predict goals.

**Public API**:

```python
class TrajectoryClassifier(eqx.Module):
    """RNN-based trajectory classifier for goal prediction."""

    def __call__(
        self,
        trajectory_sequence: Array,  # (seq_len, 2)
        state: Optional[Array] = None
    ) -> Array:                      # (num_goals,) probability distribution
        """Predict goal distribution from partial trajectory."""
        ...

def train_observer(
    dataset: TrajectoryDataset,
    config: TrainingConfig,
    key: PRNGKey,
) -> TrajectoryClassifier:
    """Train RNN observer on trajectory classification task."""
    ...
```

**Dependencies**: Equinox, Optax, `src/shared/trajectory`

#### `deception_cost.py` - Deception Cost Evaluator

**Responsibility**: Computes deception cost from observer predictions.

**Public API**:

```python
def entropy_based_deception_cost(
    goal_probs: Array,               # (num_goals,)
) -> float:
    """Compute negative entropy (higher = more revealing)."""
    ...

def accuracy_based_deception_cost(
    goal_probs: Array,               # (num_goals,)
    true_goal_id: int,
) -> float:
    """Compute classification probability for true goal."""
    ...

def evaluate_deception_cost(
    partial_path: Array,             # (path_len, 2)
    observer_net: TrajectoryClassifier,
    true_goal_id: int,
    method: str = "entropy",         # "entropy" or "accuracy"
) -> float:
    """Evaluate deception cost for a partial path."""
    ...
```

**Dependencies**: `src/deceptive/observer`, `src/shared/trajectory`

#### `tree.py` - RRT\* Tree Data Structure

**Responsibility**: Manages RRT\* search tree structure.

**Public API**:

```python
@dataclass
class RRTNode:
    """Single node in RRT* tree."""
    position: Array                  # (2,)
    parent_id: Optional[int]
    cost: float
    children: List[int]

class RRTTree:
    """RRT* search tree."""

    def add_node(self, position: Array, parent_id: int, cost: float) -> int:
        """Add node to tree, return node ID."""
        ...

    def find_nearest(self, position: Array) -> int:
        """Find nearest node to given position."""
        ...

    def find_near(self, position: Array, radius: float) -> List[int]:
        """Find all nodes within radius of position."""
        ...

    def rewire(self, node_id: int, new_parent_id: int, new_cost: float):
        """Update node parent and cost."""
        ...

    def extract_path(self, node_id: int) -> Array:
        """Extract path from root to node_id."""
        ...
```

**Dependencies**: `src/shared/geometry` (for distance computation)

### `src/interceptor/` - Interceptor Agent Package

#### `irl.py` - Inverse Reinforcement Learning

**Responsibility**: Learn behavioral model of deceptive agent from demonstrations.

**Public API**:

```python
class LearnedRewardFunction(eqx.Module):
    """Neural network parameterization of learned reward."""

    def __call__(self, state: Array, action: Array) -> float:
        """Compute reward for state-action pair."""
        ...

def maximum_entropy_irl(
    demonstrations: List[Trajectory],
    goals: Array,                    # (num_demos, 2) corresponding goals
    config: IRLConfig,
    key: PRNGKey,
) -> LearnedRewardFunction:
    """Train IRL model using maximum entropy IRL."""
    ...

def predict_trajectory(
    reward_fn: LearnedRewardFunction,
    start: Array,
    goal: Array,
    horizon: int,
    workspace: Workspace,
) -> Trajectory:
    """Predict trajectory under learned reward function."""
    ...
```

**Dependencies**: Equinox, Optax, `src/shared/{trajectory, workspace}`

#### `particle_filter.py` - Particle Filter

**Responsibility**: Maintain belief distribution over goals using particle filtering.

**Public API**:

```python
@dataclass
class Particle:
    """Single particle representing goal hypothesis."""
    goal_id: int
    weight: float

class ParticleFilter:
    """Particle filter for goal inference."""

    def __init__(
        self,
        num_particles: int,
        candidate_goals: Array,      # (num_goals, 2)
        learned_model: LearnedRewardFunction,
        key: PRNGKey,
    ):
        ...

    def predict(self, dt: float, key: PRNGKey):
        """Propagate particles forward in time."""
        ...

    def update(self, observation: Array, key: PRNGKey):
        """Update particle weights based on observation."""
        ...

    def resample(self, key: PRNGKey):
        """Resample particles based on weights."""
        ...

    def estimate_goal(self) -> Tuple[int, float]:
        """Return MAP estimate and confidence."""
        ...

    def get_belief_distribution(self) -> Array:
        """Return belief distribution over goals (num_goals,)."""
        ...
```

**Dependencies**: `src/interceptor/irl`, `src/shared/trajectory`

#### `mpc.py` - Game-Theoretic MPC

**Responsibility**: Receding-horizon optimization for interception planning.

**Public API**:

```python
def game_theoretic_mpc(
    current_state: Array,            # (2,) current position
    belief: Array,                   # (num_goals,) belief distribution
    predicted_traj_D: Trajectory,   # Agent D's predicted trajectory
    horizon: int,
    config: MPCConfig,
    key: PRNGKey,
) -> Array:                          # (2,) control action
    """Compute optimal control using game-theoretic MPC."""
    ...

def solve_mpc_optimization(
    initial_state: Array,
    target_trajectory: Trajectory,
    horizon: int,
    kinematic_constraints: KinematicConstraints,
) -> Array:                          # (horizon, 2) control sequence
    """Solve MPC optimization problem."""
    ...
```

**Dependencies**: `src/interceptor/irl`, `src/shared/{trajectory, kinematics}`

#### `belief_update.py` - Belief Update Utilities

**Responsibility**: Bayesian belief update functions.

**Public API**:

```python
def bayesian_update(
    prior: Array,                    # (num_goals,)
    likelihoods: Array,              # (num_goals,)
) -> Array:                          # (num_goals,) posterior
    """Perform Bayesian belief update."""
    ...

def compute_likelihood(
    observation: Array,
    goal_hypothesis: Array,
    learned_model: LearnedRewardFunction,
) -> float:
    """Compute likelihood P(obs | goal) using learned model."""
    ...
```

**Dependencies**: `src/interceptor/irl`

### `src/shared/` - Shared Components Package

#### `workspace.py` - 2D Workspace Environment

**Responsibility**: Represent 2D workspace with obstacles.

**Public API**:

```python
@dataclass
class CircleObstacle:
    center: Array                    # (2,)
    radius: float

@dataclass
class PolygonObstacle:
    vertices: Array                  # (n, 2)

@dataclass
class Workspace:
    bounds: Array                    # (2, 2) [[x_min, x_max], [y_min, y_max]]
    obstacles: List[Union[CircleObstacle, PolygonObstacle]]

def create_workspace(config: Dict) -> Workspace:
    """Create workspace from configuration dict."""
    ...

def is_in_workspace(position: Array, workspace: Workspace) -> bool:
    """Check if position is within workspace bounds."""
    ...
```

**Dependencies**: `src/shared/geometry`

#### `trajectory.py` - Trajectory Representation

**Responsibility**: Trajectory data structures and operations.

**Public API**:

```python
@dataclass
class Trajectory:
    """Continuous trajectory representation."""
    times: Array                     # (T,)
    positions: Array                 # (T, 2)
    velocities: Array                # (T, 2)

def create_trajectory(positions: Array, times: Array) -> Trajectory:
    """Create trajectory with computed velocities."""
    ...

def interpolate_trajectory(traj: Trajectory, new_times: Array) -> Trajectory:
    """Resample trajectory at new time points."""
    ...

def compute_path_length(traj: Trajectory) -> float:
    """Compute total path length."""
    ...

def get_partial_trajectory(traj: Trajectory, t_end: float) -> Trajectory:
    """Extract trajectory up to time t_end."""
    ...
```

**Dependencies**: None (only JAX and standard library)

#### `collision.py` - Collision Detection

**Responsibility**: Collision checking between agents and obstacles.

**Public API**:

```python
@jax.jit
def point_in_circle(
    point: Array,                    # (2,)
    obstacle: CircleObstacle,
) -> bool:
    """Check if point collides with circle obstacle."""
    ...

@jax.jit
def point_in_polygon(
    point: Array,                    # (2,)
    obstacle: PolygonObstacle,
) -> bool:
    """Check if point is inside polygon (ray casting)."""
    ...

@jax.jit
def line_segment_collision(
    p1: Array, p2: Array,           # (2,) each
    workspace: Workspace,
) -> bool:
    """Check if line segment collides with any obstacle."""
    ...

# Vectorized versions for batch operations
@jax.jit
def batch_collision_check(
    points: Array,                   # (N, 2)
    workspace: Workspace,
) -> Array:                          # (N,) boolean array
    """Check collision for batch of points (vmapped)."""
    ...
```

**Dependencies**: `src/shared/{workspace, geometry}`

#### `kinematics.py` - Kinodynamic Constraints

**Responsibility**: Enforce kinodynamic constraints on trajectories.

**Public API**:

```python
@dataclass
class KinematicConstraints:
    max_velocity: float
    max_acceleration: float
    dt: float                        # Time step

def enforce_velocity_limit(
    velocity: Array,                 # (2,)
    constraints: KinematicConstraints,
) -> Array:                          # (2,) clamped velocity
    """Clamp velocity to maximum magnitude."""
    ...

def enforce_acceleration_limit(
    v_old: Array,                    # (2,) previous velocity
    v_new: Array,                    # (2,) desired velocity
    constraints: KinematicConstraints,
) -> Array:                          # (2,) feasible velocity
    """Ensure acceleration limit is satisfied."""
    ...

def integrate_motion(
    state: Array,                    # (2,) position
    velocity: Array,                 # (2,)
    dt: float,
) -> Array:                          # (2,) new position
    """Integrate motion forward by dt."""
    ...
```

**Dependencies**: None

#### `geometry.py` - Geometric Utilities

**Responsibility**: Common geometric operations.

**Public API**:

```python
@jax.jit
def euclidean_distance(p1: Array, p2: Array) -> float:
    """Compute Euclidean distance between points."""
    ...

@jax.jit
def angle_between(v1: Array, v2: Array) -> float:
    """Compute angle between two vectors."""
    ...

@jax.jit
def closest_point_on_segment(
    point: Array,
    segment_start: Array,
    segment_end: Array,
) -> Array:
    """Find closest point on line segment to given point."""
    ...

@jax.jit
def point_to_segment_distance(
    point: Array,
    segment_start: Array,
    segment_end: Array,
) -> float:
    """Compute distance from point to line segment."""
    ...
```

**Dependencies**: None

### `src/simulation/` - Simulation Framework Package

#### `controller.py` - Simulation Controller

**Responsibility**: Main game loop orchestrating agent interaction.

**Public API**:

```python
@dataclass
class SimulationResult:
    """Result of simulation run."""
    winner: str                      # "Agent D", "Agent I", or "timeout"
    completion_time: float
    trajectory_D: Trajectory
    trajectory_I: Trajectory
    metrics: Dict[str, Any]

def run_simulation(config: SimulationConfig, key: PRNGKey) -> SimulationResult:
    """Run full adversarial simulation."""
    ...
```

**Dependencies**: All packages (`deceptive`, `interceptor`, `shared`)

#### `config.py` - Configuration Management

**Responsibility**: Load and validate YAML configuration files.

**Public API**:

```python
@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    workspace: Workspace
    deceptive_agent_config: DeceptiveAgentConfig
    interceptor_agent_config: InterceptorAgentConfig
    simulation_params: SimulationParams

def load_config(yaml_path: str) -> SimulationConfig:
    """Load configuration from YAML file."""
    ...

def validate_config(config: SimulationConfig) -> bool:
    """Validate configuration parameters."""
    ...
```

**Dependencies**: `src/shared/workspace`

#### `metrics.py` - Performance Metrics

**Responsibility**: Compute and track performance metrics.

**Public API**:

```python
def compute_observer_accuracy(
    observer_net: TrajectoryClassifier,
    trajectory: Trajectory,
    true_goal_id: int,
) -> float:
    """Compute observer classification accuracy."""
    ...

def compute_path_length_ratio(
    actual_traj: Trajectory,
    optimal_traj: Trajectory,
) -> float:
    """Compute ratio of actual to optimal path length."""
    ...

def compute_belief_entropy_over_time(
    belief_history: List[Array],
) -> Array:
    """Compute entropy of belief distribution over time."""
    ...
```

**Dependencies**: `src/deceptive/observer`, `src/shared/trajectory`

#### `visualization.py` - Visualization and Output

**Responsibility**: Generate plots and save outputs.

**Public API**:

```python
def plot_workspace_with_trajectories(
    workspace: Workspace,
    trajectories: List[Trajectory],
    goals: Array,
    save_path: str,
):
    """Plot workspace with agent trajectories."""
    ...

def plot_belief_evolution(
    belief_history: List[Array],
    candidate_goals: Array,
    save_path: str,
):
    """Plot belief distribution evolution over time."""
    ...

def save_trajectories(
    trajectories: Dict[str, Trajectory],
    save_path: str,
):
    """Save trajectories to text file."""
    ...
```

**Dependencies**: `src/shared/{workspace, trajectory}`, matplotlib

### `src/data/` - Data Handling Package

#### `schemas.py` - Data Structure Definitions

**Responsibility**: Define pytree structures for data.

**Public API**:

```python
# Register pytree structures
@jax.tree_util.register_pytree_node_class
class TrajectoryDataset:
    trajectories: List[Array]        # List of (T, 2) arrays
    goals: Array                     # (N, 2) corresponding goals
    goal_ids: Array                  # (N,) integer goal IDs
```

**Dependencies**: None

#### `loaders.py` - Data Loaders

**Responsibility**: Load training data from disk.

**Public API**:

```python
def load_trajectory_dataset(path: str) -> TrajectoryDataset:
    """Load trajectory dataset from HDF5 or NumPy file."""
    ...
```

**Dependencies**: `src/data/schemas`

#### `generators.py` - Synthetic Data Generation

**Responsibility**: Generate synthetic training data.

**Public API**:

```python
def generate_optimal_trajectories(
    workspace: Workspace,
    goals: Array,
    num_samples_per_goal: int,
    key: PRNGKey,
) -> TrajectoryDataset:
    """Generate optimal (non-deceptive) trajectories for training."""
    ...
```

**Dependencies**: `src/shared/{workspace, trajectory}`

## Import Conventions

### Import Hierarchy Rules

1. **No circular dependencies**: Enforce acyclic dependency graph
2. **`src/shared/` is foundation**: Can only import from itself and standard libraries
3. **Agent packages are independent**: `deceptive/` and `interceptor/` cannot import from each other
4. **`src/simulation/` is top-level**: Can import from all packages
5. **`src/data/` is independent**: Can only import from `shared/`

### Allowed Import Patterns

```python
# ✅ ALLOWED
from src.shared.workspace import Workspace
from src.deceptive.planner import adversarial_rrt_star
import src.shared.geometry as geom

# ❌ FORBIDDEN (violates hierarchy)
# In src/shared/workspace.py:
from src.deceptive.planner import ...  # shared cannot import agent code

# In src/deceptive/planner.py:
from src.interceptor.irl import ...     # agents cannot import from each other
```

### Public vs. Private APIs

**Public APIs** (exported in `__init__.py`):

- Functions/classes intended for use by other packages
- Stable interfaces (minimize breaking changes)

**Private implementation** (not exported):

- Helper functions with leading underscore
- Internal data structures
- Implementation details

Example `src/deceptive/__init__.py`:

```python
# Public API
from .planner import adversarial_rrt_star
from .observer import TrajectoryClassifier, train_observer
from .deception_cost import evaluate_deception_cost

# Private (not exported)
# - tree.py internals (RRTNode, RRTTree methods)
# - deception_cost.py internal helpers
```

## Package Initialization Files

### `src/__init__.py`

```python
"""Adversarial motion planning system."""

__version__ = "0.1.0"

# Top-level imports for convenience
from . import deceptive
from . import interceptor
from . import shared
from . import simulation
from . import data
```

### `src/deceptive/__init__.py`

```python
"""Deceptive agent components (Agent D)."""

from .planner import adversarial_rrt_star
from .observer import TrajectoryClassifier, train_observer
from .deception_cost import evaluate_deception_cost

__all__ = [
    "adversarial_rrt_star",
    "TrajectoryClassifier",
    "train_observer",
    "evaluate_deception_cost",
]
```

### `src/interceptor/__init__.py`

```python
"""Interceptor agent components (Agent I)."""

from .irl import LearnedRewardFunction, maximum_entropy_irl, predict_trajectory
from .particle_filter import ParticleFilter
from .mpc import game_theoretic_mpc
from .belief_update import bayesian_update, compute_likelihood

__all__ = [
    "LearnedRewardFunction",
    "maximum_entropy_irl",
    "predict_trajectory",
    "ParticleFilter",
    "game_theoretic_mpc",
    "bayesian_update",
    "compute_likelihood",
]
```

### `src/shared/__init__.py`

```python
"""Shared components used by both agents."""

from .workspace import Workspace, CircleObstacle, PolygonObstacle, create_workspace
from .trajectory import Trajectory, create_trajectory, interpolate_trajectory
from .collision import (
    point_in_circle,
    point_in_polygon,
    line_segment_collision,
    batch_collision_check,
)
from .kinematics import KinematicConstraints, integrate_motion
from . import geometry

__all__ = [
    "Workspace",
    "CircleObstacle",
    "PolygonObstacle",
    "create_workspace",
    "Trajectory",
    "create_trajectory",
    "interpolate_trajectory",
    "point_in_circle",
    "point_in_polygon",
    "line_segment_collision",
    "batch_collision_check",
    "KinematicConstraints",
    "integrate_motion",
    "geometry",
]
```

### `src/simulation/__init__.py`

```python
"""Simulation framework for adversarial evaluation."""

from .controller import run_simulation, SimulationResult
from .config import SimulationConfig, load_config
from .metrics import (
    compute_observer_accuracy,
    compute_path_length_ratio,
    compute_belief_entropy_over_time,
)
from .visualization import (
    plot_workspace_with_trajectories,
    plot_belief_evolution,
    save_trajectories,
)

__all__ = [
    "run_simulation",
    "SimulationResult",
    "SimulationConfig",
    "load_config",
    "compute_observer_accuracy",
    "compute_path_length_ratio",
    "compute_belief_entropy_over_time",
    "plot_workspace_with_trajectories",
    "plot_belief_evolution",
    "save_trajectories",
]
```

### `src/data/__init__.py`

```python
"""Data handling for training and experiments."""

from .schemas import TrajectoryDataset
from .loaders import load_trajectory_dataset
from .generators import generate_optimal_trajectories

__all__ = [
    "TrajectoryDataset",
    "load_trajectory_dataset",
    "generate_optimal_trajectories",
]
```

## Type Annotations

All public APIs should use type annotations for clarity and IDE support:

```python
from typing import List, Tuple, Optional, Union
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

# Example with jaxtyping
def adversarial_rrt_star(
    start: Float[Array, "2"],
    goal: Float[Array, "2"],
    workspace: Workspace,
    observer_net: TrajectoryClassifier,
    alpha: float,
    config: RRTConfig,
    key: PRNGKeyArray,
) -> Trajectory:
    ...
```

## Testing Organization

```
tests/
├── test_deceptive/
│   ├── test_planner.py
│   ├── test_observer.py
│   └── test_deception_cost.py
├── test_interceptor/
│   ├── test_irl.py
│   ├── test_particle_filter.py
│   └── test_mpc.py
├── test_shared/
│   ├── test_workspace.py
│   ├── test_trajectory.py
│   ├── test_collision.py
│   ├── test_kinematics.py
│   └── test_geometry.py
├── test_simulation/
│   ├── test_controller.py
│   ├── test_config.py
│   └── test_metrics.py
└── integration/
    ├── test_full_simulation.py
    └── test_agent_pipelines.py
```

Each test file mirrors the source structure and tests the corresponding module's public API.

## Navigation

**Previous**: [`00-overview.md`](./00-overview.md) - System architecture overview

**Next**: [`02-data-schemas.md`](./02-data-schemas.md) - Data formats and schemas
