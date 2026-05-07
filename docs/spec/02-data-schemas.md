# Data Formats and Schemas

## Purpose

This document specifies all data formats, schemas, and file structures used in the adversarial motion planning system. This includes training data formats, configuration file schemas (YAML), and output data specifications.

## Overview

The system handles three categories of data:

1. **Training Data**: Datasets for training RNN observer and IRL models (offline)
2. **Configuration Data**: YAML files defining experiment parameters
3. **Output Data**: Generated figures, trajectories, and metrics (results)

## Training Data Formats

### RNN Observer Training Dataset

**Purpose**: Train the surrogate observer network to classify partial trajectories by goal.

**Format**: HDF5 file or pickled dictionary

**Schema**:

```python
{
    'trajectories': List[Array],     # Length N, each entry is (T_i, 2)
    'goals': Array,                  # (N, 2) corresponding goal positions
    'goal_ids': Array,               # (N,) integer goal IDs [0, num_goals)
    'workspace_bounds': Array,       # (2, 2) [[x_min, x_max], [y_min, y_max]]
    'metadata': {
        'num_goals': int,
        'num_trajectories': int,
        'generation_method': str,    # e.g., "optimal_rrt_star"
        'timestamp': str,
    }
}
```

**Array Shapes**:

- `trajectories[i]`: Variable length (T_i, 2) where T_i is number of timesteps for trajectory i
- `goals`: (N, 2) where N is total number of trajectories
- `goal_ids`: (N,) where values are in [0, M-1] and M is number of unique goals

**File Naming Convention**: `observer_train_<workspace_id>_<num_traj>.h5`

**Example Generation** (pseudocode):

```python
# Generate 1000 optimal trajectories (200 per goal)
dataset = {
    'trajectories': [],
    'goals': [],
    'goal_ids': [],
}

candidate_goals = jnp.array([[9, 9], [9, 1], [1, 9], [1, 1], [5, 5]])  # 5 goals

for goal_id, goal in enumerate(candidate_goals):
    for _ in range(200):  # 200 samples per goal
        start = random_start_position()
        traj = plan_optimal_path(start, goal, workspace)  # Using RRT*
        dataset['trajectories'].append(traj.positions)  # (T, 2)
        dataset['goals'].append(goal)
        dataset['goal_ids'].append(goal_id)

# Save to HDF5
h5py.File('observer_train_workspace1_1000.h5', 'w').create_dataset(...)
```

**Data Augmentation**:

- **Noise injection**: Add Gaussian noise to positions (σ = 0.05)
- **Time warping**: Resample trajectories at different rates
- **Partial truncation**: Use varying completion percentages (10%-90%)
- **Mirroring**: Flip trajectories horizontally/vertically if workspace is symmetric

### IRL Training Dataset

**Purpose**: Provide expert demonstrations of deceptive agent behavior for inverse reinforcement learning.

**Format**: HDF5 file or pickled dictionary

**Schema**:

```python
{
    'demonstrations': List[Dict],    # Length N demonstrations
    # Each demonstration:
    # {
    #     'trajectory': Array,       # (T, 2) positions
    #     'velocities': Array,       # (T, 2) velocities
    #     'actions': Array,          # (T-1, 2) control actions
    #     'goal': Array,             # (2,) true goal
    #     'goal_id': int,
    #     'deception_weight': float, # α parameter used
    # }
    'workspace_bounds': Array,       # (2, 2)
    'obstacles': List[Dict],         # Obstacle specifications
    'metadata': {
        'num_demonstrations': int,
        'agent_type': str,           # e.g., "adversarial_rrt_star"
        'timestamp': str,
    }
}
```

**Array Shapes**:

- `trajectory`: (T, 2) where T varies per demonstration
- `velocities`: (T, 2)
- `actions`: (T-1, 2) discrete control actions

**File Naming Convention**: `irl_demonstrations_<agent_type>_<num_demos>.h5`

**Example Structure**:

```python
demonstrations = [
    {
        'trajectory': jnp.array([[1.0, 1.0], [1.5, 1.2], ..., [9.0, 9.0]]),  # (T, 2)
        'velocities': jnp.array([[0.5, 0.2], [0.6, 0.3], ...]),               # (T, 2)
        'actions': jnp.array([[0.1, 0.05], [0.15, 0.02], ...]),               # (T-1, 2)
        'goal': jnp.array([9.0, 9.0]),
        'goal_id': 0,
        'deception_weight': 0.3,
    },
    # ... more demonstrations
]
```

## Configuration Data (YAML)

### Experiment Configuration Schema

**Purpose**: Define all parameters for a simulation experiment.

**File Format**: YAML

**Schema**:

```yaml
# Workspace configuration
workspace:
    bounds: [[0.0, 10.0], [0.0, 10.0]] # [[x_min, x_max], [y_min, y_max]]
    obstacles:
        - type: circle
          center: [5.0, 5.0]
          radius: 1.0
        - type: circle
          center: [7.0, 3.0]
          radius: 0.8
        - type: polygon
          vertices: [[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]]

# Deceptive agent configuration
deceptive_agent:
    initial_position: [1.0, 1.0]
    true_goal: [9.0, 9.0]
    candidate_goals: # Must include true_goal
        - [9.0, 9.0]
        - [9.0, 1.0]
        - [1.0, 9.0]

    # Adversarial RRT* parameters
    planner:
        deception_weight: 0.3 # α ∈ [0, 1], 0=pure path opt, 1=pure deception
        max_iterations: 5000
        step_size: 0.5
        goal_radius: 0.5 # Goal region radius
        rewiring_radius: 2.0
        goal_bias_probability: 0.1 # Probability of sampling near goal

    # Kinodynamic constraints
    kinematics:
        max_velocity: 2.0
        max_acceleration: 1.0

    # Observer network
    observer:
        checkpoint_path: "data/models/observer_rnn.eqx"
        deception_cost_method: "entropy" # "entropy" or "accuracy"

# Interceptor agent configuration
interceptor_agent:
    initial_position: [9.0, 1.0]
    candidate_goals: # Same as deceptive agent
        - [9.0, 9.0]
        - [9.0, 1.0]
        - [1.0, 9.0]

    # IRL model
    irl:
        checkpoint_path: "data/models/irl_reward.eqx"

    # Particle filter parameters
    particle_filter:
        num_particles: 1000
        resample_threshold: 0.5 # Effective sample size ratio for resampling
        motion_noise_std: 0.1 # Process noise standard deviation

    # MPC parameters
    mpc:
        horizon: 20 # Planning horizon (timesteps)
        control_weight: 0.01 # λ_u in cost function
        optimization_method: "lbfgs" # "lbfgs", "adam", or "gradient_descent"
        max_iterations: 100

    # Kinodynamic constraints
    kinematics:
        max_velocity: 2.5
        max_acceleration: 1.5

# Simulation parameters
simulation:
    timestep: 0.1 # Δt for discrete time simulation
    max_time: 100.0 # Maximum simulation time
    intercept_threshold: 0.5 # Distance threshold for interception
    random_seed: 42

# Output configuration
outputs:
    save_figures: true
    save_trajectories: true
    save_metrics: true
    figure_format: "png" # "png", "pdf", or "svg"
    figure_dpi: 300
    output_dir: "outputs" # Relative to project root

# Training configuration (used by: uv run adversarial-train)
training:
    observer:
        hidden_dim: 64
        num_epochs: 100
        learning_rate: 1.0e-3
        batch_size: 32
        samples_per_goal: 200
    irl:
        hidden_dim: 64
        num_epochs: 50
        learning_rate: 1.0e-3
        num_demonstrations: 500
```

**Type Specifications**:

- `bounds`: List[List[float]], shape (2, 2)
- `obstacles`: List of obstacle dicts
    - Circle: `{type: "circle", center: [x, y], radius: r}`
    - Polygon: `{type: "polygon", vertices: [[x1, y1], [x2, y2], ...]}`
- `initial_position`, `true_goal`, `candidate_goals`: List of [x, y] floats
- All numeric parameters: float or int as appropriate

**Validation Rules**:

1. `deceptive_agent.true_goal` must be in `deceptive_agent.candidate_goals`
2. `deceptive_agent.candidate_goals` must equal `interceptor_agent.candidate_goals`
3. `deception_weight` must be in [0, 1]
4. `initial_position` must be within `workspace.bounds` and collision-free
5. All goals must be within `workspace.bounds` and collision-free
6. `timestep` must be > 0
7. `intercept_threshold` must be > 0

**File Naming Convention**: `experiment_<scenario_name>.yaml`

**Example Config Files**:

`experiment_simple_obstacle.yaml`:

```yaml
workspace:
    bounds: [[0, 10], [0, 10]]
    obstacles:
        - type: circle
          center: [5, 5]
          radius: 1.5

deceptive_agent:
    initial_position: [1, 1]
    true_goal: [9, 9]
    candidate_goals: [[9, 9], [9, 1], [1, 9]]
    planner:
        deception_weight: 0.4
        max_iterations: 3000
    # ... rest of config
```

## Output Data Formats

### Trajectory Output Format

**Purpose**: Save agent trajectories for post-analysis and visualization.

**Format**: CSV or text file

**Schema** (CSV):

```csv
agent,time,x,y,vx,vy
Agent_D,0.0,1.0,1.0,0.0,0.0
Agent_D,0.1,1.05,1.02,0.5,0.2
Agent_D,0.2,1.11,1.06,0.6,0.4
...
Agent_I,0.0,9.0,1.0,0.0,0.0
Agent_I,0.1,8.95,1.05,-0.5,0.5
...
```

**Columns**:

- `agent`: "Agent_D" or "Agent_I"
- `time`: float (seconds)
- `x`, `y`: float (position)
- `vx`, `vy`: float (velocity)

**File Naming Convention**: `trajectories_<experiment_name>_<timestamp>.csv`

**Alternative Format** (NumPy):

```python
{
    'Agent_D': {
        'times': Array,      # (T_D,)
        'positions': Array,  # (T_D, 2)
        'velocities': Array, # (T_D, 2)
    },
    'Agent_I': {
        'times': Array,      # (T_I,)
        'positions': Array,  # (T_I, 2)
        'velocities': Array, # (T_I, 2)
    },
}
```

Saved as: `trajectories_<experiment_name>.npz`

### Metrics Output Format

**Purpose**: Save quantitative performance metrics.

**Format**: CSV

**Schema**:

```csv
metric,value
winner,Agent_D
completion_time,45.7
observer_accuracy_final,0.42
observer_accuracy_mean,0.35
path_length_ratio,1.23
belief_entropy_final,0.87
belief_entropy_mean,1.15
interception_distance_min,1.2
goal_inference_accuracy,0.0
time_to_interception,inf
```

**Metrics Definitions**:

- `winner`: "Agent_D", "Agent_I", or "timeout"
- `completion_time`: Time when simulation ended (float)
- `observer_accuracy_final`: P(true_goal | final_trajectory)
- `observer_accuracy_mean`: Mean accuracy over all partial trajectories
- `path_length_ratio`: actual_length / optimal_length
- `belief_entropy_final`: H(b_T) at final time
- `belief_entropy_mean`: Mean entropy over time
- `interception_distance_min`: Minimum distance between agents
- `goal_inference_accuracy`: 1.0 if Agent I correctly identified goal, else 0.0
- `time_to_interception`: Time when intercepted (inf if not intercepted)

**File Naming Convention**: `metrics_<experiment_name>_<timestamp>.csv`

### Belief History Output Format

**Purpose**: Save Agent I's belief distribution evolution over time.

**Format**: CSV

**Schema**:

```csv
time,goal_0,goal_1,goal_2,entropy
0.0,0.333,0.333,0.333,1.099
0.1,0.340,0.330,0.330,1.097
0.2,0.355,0.325,0.320,1.092
...
5.0,0.720,0.180,0.100,0.763
```

**Columns**:

- `time`: float (seconds)
- `goal_0`, `goal_1`, ..., `goal_N-1`: Belief probability for each candidate goal
- `entropy`: Shannon entropy H(b_t)

**File Naming Convention**: `belief_history_<experiment_name>_<timestamp>.csv`

### Figure Output Specifications

**Purpose**: Generated visualizations of simulation results.

#### Figure 1: Workspace with Trajectories

**Content**:

- 2D workspace bounds
- Obstacles (circles, polygons)
- Agent D trajectory (blue line)
- Agent I trajectory (red line)
- Candidate goals (markers)
- True goal (highlighted)
- Start positions (markers)

**Format**: PNG, PDF, or SVG

**Resolution**: 300 DPI (configurable)

**Dimensions**: 8" × 8" (square aspect ratio)

**File Naming**: `workspace_trajectories_<experiment_name>.png`

#### Figure 2: Belief Distribution Evolution

**Content**:

- Stacked area chart or line plot
- X-axis: Time (seconds)
- Y-axis: Belief probability [0, 1]
- One series per candidate goal
- Annotate true goal

**Format**: PNG, PDF, or SVG

**Dimensions**: 10" × 6"

**File Naming**: `belief_evolution_<experiment_name>.png`

#### Figure 3: Metrics Over Time

**Content**:

- Multi-panel plot:
    - Panel 1: Distance between agents vs. time
    - Panel 2: Observer accuracy vs. time (if evaluated at multiple points)
    - Panel 3: Belief entropy vs. time

**Format**: PNG, PDF, or SVG

**Dimensions**: 12" × 8" (3 vertical panels)

**File Naming**: `metrics_over_time_<experiment_name>.png`

## Data Directory Structure

```
outputs/
├── figures/
│   ├── workspace_trajectories_exp1.png
│   ├── belief_evolution_exp1.png
│   ├── metrics_over_time_exp1.png
│   └── ...
└── text/
    ├── trajectories_exp1.csv
    ├── metrics_exp1.csv
    ├── belief_history_exp1.csv
    └── ...

data/
├── training/
│   ├── observer_train_workspace1_1000.h5
│   ├── irl_demonstrations_arrt_500.h5
│   └── ...
├── configs/
│   ├── experiment_simple_obstacle.yaml
│   ├── experiment_complex_scenario.yaml
│   └── ...
└── models/                       # Generated by: uv run adversarial-train
    ├── observer_rnn.eqx          # Trained RNN observer (Equinox pytree leaves)
    └── irl_reward.eqx            # Trained IRL reward function (Equinox pytree leaves)
```

## Pytree Structures (JAX)

### AgentState Pytree

```python
from typing import NamedTuple
import jax.numpy as jnp

class AgentState(NamedTuple):
    position: jnp.ndarray  # (2,)
    velocity: jnp.ndarray  # (2,)

# Pytree automatically registered for NamedTuple
```

### Trajectory Pytree

```python
class Trajectory(NamedTuple):
    times: jnp.ndarray      # (T,)
    positions: jnp.ndarray  # (T, 2)
    velocities: jnp.ndarray # (T, 2)

# Usage
traj = Trajectory(
    times=jnp.linspace(0, 10, 100),
    positions=jnp.zeros((100, 2)),
    velocities=jnp.zeros((100, 2)),
)

# JAX operations work seamlessly
traj_scaled = jax.tree_map(lambda x: 2 * x, traj)
```

### Workspace Pytree

```python
@dataclass
class Workspace:
    bounds: jnp.ndarray              # (2, 2)
    obstacle_circles: List[Tuple[jnp.ndarray, float]]  # [(center, radius), ...]
    obstacle_polygons: List[jnp.ndarray]                # [vertices, ...]

# Register as pytree
from jax.tree_util import register_pytree_node

def workspace_flatten(ws):
    children = (ws.bounds,)
    aux_data = (ws.obstacle_circles, ws.obstacle_polygons)
    return children, aux_data

def workspace_unflatten(aux_data, children):
    bounds, = children
    obstacle_circles, obstacle_polygons = aux_data
    return Workspace(bounds, obstacle_circles, obstacle_polygons)

register_pytree_node(Workspace, workspace_flatten, workspace_unflatten)
```

## Serialization and Deserialization

### Saving Equinox Models

```python
import equinox as eqx

# Save model (done by: uv run adversarial-train)
observer_net = TrajectoryClassifier(...)
eqx.tree_serialise_leaves("data/models/observer_rnn.eqx", observer_net)

# Load model (done by: uv run adversarial-planning)
loaded_observer = eqx.tree_deserialise_leaves("data/models/observer_rnn.eqx", observer_net)
```

### Saving/Loading Trajectories (NumPy)

```python
# Save
jnp.savez(
    "outputs/text/trajectories_exp1.npz",
    Agent_D_times=traj_D.times,
    Agent_D_positions=traj_D.positions,
    Agent_D_velocities=traj_D.velocities,
    Agent_I_times=traj_I.times,
    Agent_I_positions=traj_I.positions,
    Agent_I_velocities=traj_I.velocities,
)

# Load
data = jnp.load("outputs/text/trajectories_exp1.npz")
traj_D = Trajectory(
    times=data['Agent_D_times'],
    positions=data['Agent_D_positions'],
    velocities=data['Agent_D_velocities'],
)
```

### Loading YAML Configuration

```python
import yaml
from src.simulation.config import SimulationConfig

with open("data/configs/experiment_simple.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

# Validate and parse
config = SimulationConfig.from_dict(config_dict)
```

## Data Validation

### Configuration Validation

```python
def validate_workspace(workspace_config: Dict) -> bool:
    """Validate workspace configuration."""
    # Check bounds format
    assert len(workspace_config['bounds']) == 2
    assert len(workspace_config['bounds'][0]) == 2

    # Check obstacles
    for obs in workspace_config['obstacles']:
        if obs['type'] == 'circle':
            assert 'center' in obs and 'radius' in obs
            assert len(obs['center']) == 2
            assert obs['radius'] > 0
        elif obs['type'] == 'polygon':
            assert 'vertices' in obs
            assert len(obs['vertices']) >= 3  # At least triangle

    return True
```

### Training Data Validation

```python
def validate_trajectory_dataset(dataset: Dict) -> bool:
    """Validate RNN observer training dataset."""
    num_traj = len(dataset['trajectories'])

    assert len(dataset['goals']) == num_traj
    assert len(dataset['goal_ids']) == num_traj

    # Check trajectory shapes
    for traj in dataset['trajectories']:
        assert traj.ndim == 2
        assert traj.shape[1] == 2  # (T, 2)

    # Check goal IDs are valid
    assert jnp.all(dataset['goal_ids'] >= 0)
    assert jnp.all(dataset['goal_ids'] < dataset['metadata']['num_goals'])

    return True
```

## Example Usage

### Complete Workflow Example

```python
# 1. Train models (run once, writes checkpoints to data/models/)
#    uv run adversarial-train

# 2. Load configuration
config = load_config("data/configs/experiment_simple.yaml")

# 3. Run simulation (loads checkpoints internally from checkpoint_path fields in config)
result = run_simulation(config, key=jax.random.PRNGKey(42))

# 4. Save outputs
save_trajectories(result.trajectories, "outputs/text/trajectories_exp1.csv")
save_metrics(result.metrics, "outputs/text/metrics_exp1.csv")
plot_workspace_with_trajectories(config.workspace, result.trajectories,
                                   "outputs/figures/workspace_exp1.png")
```

## Navigation

**Previous**: [`01-package-structure.md`](./01-package-structure.md) - Package organization

**Next**: [`03-workspace-environment.md`](./03-workspace-environment.md) - Workspace & environment implementation
