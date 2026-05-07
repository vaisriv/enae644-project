# System Architecture Overview

## Purpose

This document provides a high-level overview of the adversarial motion planning system architecture, detailing how components interact to implement a two-agent adversarial scenario where a deceptive agent attempts to reach a hidden goal while concealing its intent, and an interceptor agent seeks to infer and intercept.

## System Context

This is an ENAE644 term project implementing algorithms from:

- Deceptive motion planning (Adversarial RRT\* with learned deception costs)
- Goal identification and interception (IRL + particle filtering + game-theoretic MPC)

The system evaluates these algorithms in direct adversarial competition rather than in isolation.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SIMULATION CONTROLLER                      │
│                    (src/simulation/controller.py)               │
│                                                                 │
│  • Game loop coordination                                       │
│  • Time stepping                                                │
│  • Termination detection                                        │
│  • Metrics collection                                           │
└────────────┬──────────────────────────────────┬─────────────────┘
             │                                  │
             ▼                                  ▼
┌────────────────────────────┐    ┌────────────────────────────┐
│    DECEPTIVE AGENT (D)     │    │   INTERCEPTOR AGENT (I)    │
│   (src/deceptive/)         │    │   (src/interceptor/)       │
│                            │    │                            │
│  ┌──────────────────────┐  │    │  ┌──────────────────────┐  │
│  │ Adversarial RRT*     │  │    │  │ IRL Module           │  │
│  │  • Sampling          │  │    │  │  • Behavioral model  │  │
│  │  • Tree extension    │  │    │  │  • Reward learning   │  │
│  │  • Rewiring          │  │    │  └──────────────────────┘  │
│  └──────────────────────┘  │    │                            │
│                            │    │  ┌──────────────────────┐  │
│  ┌──────────────────────┐  │    │  │ Particle Filter      │  │
│  │ RNN Observer         │  │    │  │  • Belief tracking   │  │
│  │  • Trajectory class. │  │    │  │  • Prediction        │  │
│  │  • Goal prediction   │  │    │  │  • Update            │  │
│  └──────────────────────┘  │    │  │  • Resampling        │  │
│                            │    │  └──────────────────────┘  │
│  ┌──────────────────────┐  │    │                            │
│  │ Deception Cost       │  │    │  ┌──────────────────────┐  │
│  │  • Entropy-based     │  │    │  │ Game-Theoretic MPC   │  │
│  │  • Accuracy-based    │  │    │  │  • Trajectory pred.  │  │
│  └──────────────────────┘  │    │  │  • Control opt.      │  │
│                            │    │  │  • Replanning        │  │
└────────────┬───────────────┘    │  └──────────────────────┘  │
             │                    └────────────┬───────────────┘
             │                                 │
             └─────────────┬───────────────────┘
                           ▼
            ┌──────────────────────────────┐
            │      SHARED COMPONENTS       │
            │       (src/shared/)          │
            │                              │
            │  • Workspace (2D environment)│
            │  • Trajectory representation │
            │  • Collision detection       │
            │  • Kinodynamic constraints   │
            │  • Geometric utilities       │
            └──────────────────────────────┘
                           ▲
                           │
            ┌──────────────┴───────────────┐
            │                              │
  ┌─────────▼────────┐         ┌──────────▼──────────┐
  │  CONFIGURATION   │         │   DATA HANDLING     │
  │ (src/simulation/ │         │    (src/data/)      │
  │     config.py)   │         │                     │
  │                  │         │  • Schemas          │
  │  • YAML parser   │         │  • Loaders          │
  │  • Validation    │         │  • Generators       │
  └──────────────────┘         └─────────────────────┘
```

## Component Hierarchy

### Top Level: Simulation Controller

- **Package**: `src/simulation/`
- **Responsibility**: Orchestrates the adversarial interaction between agents
- **Key Modules**:
    - `controller.py`: Main game loop
    - `config.py`: Configuration loading and validation
    - `metrics.py`: Performance metric computation
    - `visualization.py`: Plotting and output generation

### Agent Level: Deceptive Agent & Interceptor Agent

- **Packages**: `src/deceptive/`, `src/interceptor/`
- **Responsibility**: Implement agent-specific planning and learning algorithms
- **Interaction**: Agents are independent; interaction is mediated by simulation controller

### Foundation Level: Shared Components

- **Package**: `src/shared/`
- **Responsibility**: Common functionality used by both agents
- **Key Modules**:
    - `workspace.py`: 2D environment with obstacles
    - `trajectory.py`: Trajectory representation and interpolation
    - `collision.py`: Collision detection
    - `kinematics.py`: Kinodynamic constraint checking
    - `geometry.py`: Geometric utilities (distances, angles, etc.)

### Support Level: Data & Configuration

- **Packages**: `src/data/`, `src/simulation/config.py`
- **Responsibility**: Data loading, schema definitions, configuration management

## Data Flow

### Offline Training Phase

```
1. Generate synthetic trajectory data
   ↓
2. Train RNN Observer (Agent D)
   • Input: Partial trajectories → Output: Goal probabilities
   ↓
3. Train IRL Model (Agent I)
   • Input: Expert demonstrations → Output: Learned reward function
   ↓
4. Save model checkpoints
```

### Online Simulation Phase

```
1. Load configuration (YAML) → Initialize workspace, agent parameters
   ↓
2. Agent D: Plan full trajectory using Adversarial RRT*
   • Uses RNN observer to evaluate deception cost
   • Balances path length vs. deception (α parameter)
   ↓
3. Simulation Loop (t = 0 → T):
   ├─→ Agent D: Execute trajectory segment
   ├─→ Agent I: Observe partial trajectory
   ├─→ Agent I: Update belief distribution (particle filter)
   ├─→ Agent I: Plan control action (game-theoretic MPC)
   ├─→ Agent I: Execute control
   ├─→ Check termination:
   │    • Agent D reached goal? → D wins
   │    • Agent I intercepted D? → I wins
   │    • Timeout? → Draw
   └─→ t += Δt
   ↓
4. Generate outputs:
   • Trajectories (TXT/CSV)
   • Plots (PNG): workspace, trajectories, belief evolution
   • Metrics: classification accuracy, path length ratio, etc.
```

## Key Abstractions

### AgentState (Pytree)

```python
AgentState = {
    'position': Array,     # (2,) - [x, y]
    'velocity': Array,     # (2,) - [vx, vy]
    'acceleration': Array, # (2,) - [ax, ay] (optional)
}
```

### Trajectory (Pytree)

```python
Trajectory = {
    'times': Array,       # (T,)
    'positions': Array,   # (T, 2)
    'velocities': Array,  # (T, 2)
}
```

### Workspace Configuration

```python
WorkspaceConfig = {
    'bounds': Array,      # (2, 2) - [[x_min, x_max], [y_min, y_max]]
    'obstacles': List[Obstacle],  # Circle or Polygon obstacles
}
```

### Particle (for belief distribution)

```python
Particle = {
    'goal_id': int,           # Index into candidate goals
    'weight': float,          # Particle weight (normalized)
}
```

## Technology Stack Integration

### JAX Ecosystem

- **JAX**: Core array operations, autodiff, JIT compilation
- **Equinox**: Neural network components (RNN observer, IRL reward function)
- **Optax**: Optimizers for training (Adam, LBFGS for MPC)

### Python Scientific Stack

- **NumPy**: Array operations (interop with JAX)
- **SciPy**: Optimization, spatial data structures (KD-tree for RRT\*)
- **Matplotlib**: Visualization
- **YAML**: Configuration files

### JAX Design Patterns

- **Pure Functions**: All core algorithms are pure functions with explicit state
- **Pytrees**: Structured data (states, trajectories, configs) as pytrees
- **JIT Compilation**: Performance-critical code (collision checking, neural networks, cost functions)
- **vmap**: Parallel operations (batch collision checks, particle updates)
- **grad**: Optimization (MPC, IRL training)
- **PRNG**: Explicit key management for reproducibility

## Design Principles

### 1. Agent-Based Separation

- Clear package boundaries between `deceptive/` and `interceptor/`
- No direct imports between agent packages
- All shared functionality in `src/shared/`

### 2. Configuration-Driven

- All experiments defined in YAML files
- No hardcoded parameters in implementation
- Easy parameter sweeps and ablation studies

### 3. JAX-First

- Performance-critical code uses JAX (>90% of compute time)
- Pure functions enable easy testing and JIT compilation
- Non-JIT-friendly operations (RRT\* tree building) use Python with JIT subroutines

### 4. Reproducibility

- All random operations use JAX PRNG with explicit seeds
- Configuration files include random seed
- Deterministic simulation given same config

### 5. Testability

- Pure functions with clear inputs/outputs
- Unit tests for individual components
- Integration tests for agent pipelines
- Validation tests against known baselines

## Module Dependencies

```
src/simulation/controller.py
  ├─→ src/deceptive/planner.py
  │    ├─→ src/deceptive/observer.py (RNN)
  │    ├─→ src/deceptive/deception_cost.py
  │    ├─→ src/shared/workspace.py
  │    ├─→ src/shared/collision.py
  │    └─→ src/shared/trajectory.py
  │
  ├─→ src/interceptor/particle_filter.py
  │    ├─→ src/interceptor/irl.py (learned model)
  │    └─→ src/shared/trajectory.py
  │
  ├─→ src/interceptor/mpc.py
  │    ├─→ src/interceptor/irl.py (for prediction)
  │    ├─→ src/shared/kinematics.py
  │    └─→ src/shared/trajectory.py
  │
  ├─→ src/simulation/config.py
  ├─→ src/simulation/metrics.py
  └─→ src/simulation/visualization.py
       └─→ src/shared/workspace.py

src/shared/* (no dependencies on agent-specific code)
```

### Import Rules

1. `src/shared/` can only import from `src/shared/` and standard libraries
2. `src/deceptive/` can import from `src/shared/` and `src/deceptive/`
3. `src/interceptor/` can import from `src/shared/` and `src/interceptor/`
4. `src/simulation/` can import from all packages
5. `src/data/` can import from `src/shared/`

## Performance Considerations

### Computational Bottlenecks

1. **RRT\* Planning**: O(n log n) per iteration, ~5000 iterations
    - Mitigation: JIT-compiled collision checks, efficient nearest-neighbor search
2. **RNN Observer Inference**: O(sequence_length × hidden_size)
    - Mitigation: JIT-compiled forward pass, small hidden size
3. **MPC Optimization**: O(iterations × horizon × state_dim)
    - Mitigation: Warm-start from previous solution, short horizon (~20 steps)
4. **Particle Filter Update**: O(num_particles × observation_complexity)
    - Mitigation: vmap for parallel updates, adaptive resampling

### Memory Usage

- Trajectory storage: O(time_steps × num_agents × state_dim)
- RRT\* tree: O(num_nodes × (state_dim + connectivity))
- Particle filter: O(num_particles × state_dim)
- Neural networks: O(parameters) - stored on GPU if available

### GPU Utilization

- Neural network inference (RNN observer)
- Batch collision checking (vmap)
- MPC optimization (JAX autodiff on GPU)
- IRL training (offline, GPU-accelerated)

## Extension Points

Future enhancements that can build on this architecture:

1. **3D Workspace**: Extend `src/shared/workspace.py` to 3D (minimal changes elsewhere)
2. **Alternative Planners**: Add new planners in `src/deceptive/` or `src/interceptor/`
3. **Multi-Agent**: Extend to N agents (requires simulation controller changes)
4. **Different Observer Models**: Swap RNN for transformer, attention-based models
5. **Alternative IRL Algorithms**: Replace MaxEnt IRL with other variants
6. **Real-Time Visualization**: Add animation during simulation (not just post-hoc)

## Navigation

**Next Steps**:

- [`01-package-structure.md`](./01-package-structure.md) - Detailed package organization
- [`02-data-schemas.md`](./02-data-schemas.md) - Data format specifications
- [`05-deceptive-agent.md`](./05-deceptive-agent.md) - Agent D implementation guide
- [`06-interceptor-agent.md`](./06-interceptor-agent.md) - Agent I implementation guide

**Related**:

- Project root: `../../`
- Source code: `../../src/` (implementation goes here)
- Report: `../../reports/main.typ` (technical paper)
