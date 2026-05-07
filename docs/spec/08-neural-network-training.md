# Neural Network Training Procedures

## Purpose

Specifies the offline training pipeline for both learned models — the RNN surrogate observer (Agent D) and the IRL reward function (Agent I) — including data generation, training orchestration, checkpoint persistence, and the startup guard in the main simulation.

## Overview

Training is separated from simulation via two CLI entry points:

| Command                             | Source                  | Description                         |
| ----------------------------------- | ----------------------- | ----------------------------------- |
| `uv run adversarial-planning-train` | `src/training.py:train` | Train all models, write checkpoints |
| `uv run adversarial-planning`       | `src/index.py:main`     | Load checkpoints, run simulation    |

This separation ensures:

- **Reproducibility**: The same trained weights are used across all simulation runs
- **Determinism**: Re-running the simulation does not re-introduce stochastic training variance
- **Speed**: The main program starts immediately without waiting for training

---

## Training Entrypoint (`src/training.py`)

### Entry Function

```python
def train(config_path: str = "data/configs/experiment_simple_obstacle.yaml") -> None:
    """
    Full offline training pipeline. Invoked via: uv run adversarial-planning-train
    Trains the RNN observer and IRL reward function, then saves checkpoints.
    """
    config = load_config(config_path)
    key = jax.random.PRNGKey(config.simulation_params.random_seed)

    key, obs_key, irl_key = jax.random.split(key, 3)

    print("=== Phase 1: Training RNN Observer ===")
    _train_observer_from_config(config, obs_key)

    print("=== Phase 2: Training IRL Reward Function ===")
    _train_irl_from_config(config, irl_key)

    print("Training complete.")
    print(f"  Observer checkpoint : outputs/models/observer_rnn.eqx")
    print(f"  IRL checkpoint      : outputs/models/irl_reward.eqx")
```

### Observer Training Helper

```python
def _train_observer_from_config(config: SimulationConfig, key: PRNGKey) -> None:
    """Generate trajectory data, train the RNN observer, and save checkpoint."""
    key, data_key, train_key = jax.random.split(key, 3)

    # Build TrainingConfig from simulation config
    training_cfg = TrainingConfig(
        hidden_dim=config.training.observer.hidden_dim,
        num_goals=len(config.deceptive_agent.candidate_goals),
        num_epochs=config.training.observer.num_epochs,
        learning_rate=config.training.observer.learning_rate,
        batch_size=config.training.observer.batch_size,
        samples_per_goal=config.training.observer.samples_per_goal,
    )

    # Generate dataset
    dataset = generate_optimal_trajectories(
        workspace=config.workspace,
        goals=jnp.array(config.deceptive_agent.candidate_goals),
        num_samples_per_goal=training_cfg.samples_per_goal,
        key=data_key,
    )

    # Train model (see src/deceptive/observer.py for train_observer implementation)
    observer = train_observer(dataset=dataset, config=training_cfg, key=train_key)

    # Save checkpoint
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves("outputs/models/observer_rnn.eqx", observer)
```

### IRL Training Helper

```python
def _train_irl_from_config(config: SimulationConfig, key: PRNGKey) -> None:
    """Generate demonstrations, train the IRL reward function, and save checkpoint."""
    key, data_key, train_key = jax.random.split(key, 3)

    # Build IRLConfig from simulation config
    irl_cfg = IRLConfig(
        hidden_dim=config.training.irl.hidden_dim,
        learning_rate=config.training.irl.learning_rate,
        num_epochs=config.training.irl.num_epochs,
    )

    # Generate demonstrations
    demonstrations = generate_irl_demonstrations(
        workspace=config.workspace,
        goals=jnp.array(config.deceptive_agent.candidate_goals),
        num_demonstrations=config.training.irl.num_demonstrations,
        key=data_key,
    )

    # Train model (see src/interceptor/irl.py for maximum_entropy_irl implementation)
    irl_model = maximum_entropy_irl(
        demonstrations=demonstrations,
        goals=jnp.array(config.deceptive_agent.candidate_goals),
        config=irl_cfg,
        key=train_key,
    )

    # Save checkpoint
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves("outputs/models/irl_reward.eqx", irl_model)
```

---

## Startup Guard in Main Program (`src/index.py`)

Before running the simulation, `main()` checks that both checkpoints exist and exits with a clear error if either is missing.

```python
_OBSERVER_PATH = Path("outputs/models/observer_rnn.eqx")
_IRL_PATH      = Path("outputs/models/irl_reward.eqx")

def _check_checkpoints() -> None:
    missing = [p for p in [_OBSERVER_PATH, _IRL_PATH] if not p.exists()]
    if missing:
        paths = ", ".join(str(p) for p in missing)
        raise RuntimeError(
            f"Model checkpoints not found: {paths}\n"
            f"Run training first:  uv run adversarial-planning-train"
        )

def main() -> None:
    config = load_config(...)
    _check_checkpoints()
    run_simulation(config, key=jax.random.PRNGKey(config.simulation_params.random_seed))
```

---

## Loading Checkpoints

`load_observer` and `load_irl_model` live alongside the classes they load, in `src/deceptive/observer.py` and `src/interceptor/irl.py` respectively. See [05-deceptive-agent.md](./05-deceptive-agent.md) and [06-interceptor-agent.md](./06-interceptor-agent.md) for their implementations.

Both use `eqx.tree_deserialise_leaves` with a model skeleton reconstructed from `SimulationConfig`. The skeleton must match the architecture used at training time — architecture hyperparameters come from `config.training`, ensuring the same config file that trains the model also governs how it is loaded.

---

## RNN Observer Training Details

The training loop implementation lives in `src/deceptive/observer.py` as `train_observer`. The procedure is:

1. Generate optimal (non-deceptive) RRT\* trajectories to each candidate goal via `generate_optimal_trajectories`
2. Apply augmentation during training: Gaussian noise (σ=0.05), partial truncation (10–90%), time warping
3. Minimise cross-entropy loss with Adam over batched samples

**Hyperparameters** (defaults in `TrainingConfig`, overridden by YAML `training.observer`):

| Parameter        | Default |
| ---------------- | ------- |
| Hidden dim       | 64      |
| Learning rate    | 1e-3    |
| Batch size       | 32      |
| Epochs           | 100     |
| Samples per goal | 200     |

---

## IRL Model Training Details

The training loop implementation lives in `src/interceptor/irl.py` as `maximum_entropy_irl`. The procedure is:

1. Generate non-deceptive RRT\* demonstrations via `generate_irl_demonstrations`
2. Run maximum entropy IRL: alternate between soft value iteration (to compute expected features under current reward) and gradient updates (empirical − expected features)

**Hyperparameters** (defaults in `IRLConfig`, overridden by YAML `training.irl`):

| Parameter          | Default |
| ------------------ | ------- |
| Hidden dim         | 64      |
| Learning rate      | 1e-3    |
| Epochs             | 50      |
| Num demonstrations | 500     |

---

## Checkpoint Files

All model artifacts are written to `outputs/models/`, treated as a generated directory (not committed to version control).

| File                              | Contents                                                        |
| --------------------------------- | --------------------------------------------------------------- |
| `outputs/models/observer_rnn.eqx` | Trained `TrajectoryClassifier` weights (Equinox pytree leaves)  |
| `outputs/models/irl_reward.eqx`   | Trained `LearnedRewardFunction` weights (Equinox pytree leaves) |

The `.eqx` format stores only the array leaves of the pytree. The model structure is not stored — it is reconstructed from `SimulationConfig.training` at load time. The same YAML config must be used for both `adversarial-planning-train` and `adversarial-planning`.

---

## Navigation

**Previous**: [`07-simulation-controller.md`](./07-simulation-controller.md)

**Next**: [`09-testing-strategy.md`](./09-testing-strategy.md)
