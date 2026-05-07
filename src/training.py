"""Offline model training for adversarial motion planning.

Trains the neural network models required by the simulation:
1. RNN observer (TrajectoryClassifier) for Agent D's deception cost
2. IRL reward function (LearnedRewardFunction) for Agent I's behavioural model

Run once offline via:
    uv run adversarial-planning-train
"""

import csv
from pathlib import Path

import equinox as eqx
import jax

from src.simulation.config import (
    IRLTrainingConfig,
    ObserverTrainingConfig,
    SimulationConfig,
    load_config,
)
from src.simulation.config import create_workspace_from_config  # type: ignore[attr-defined]


def train(config_path: str = "data/configs/experiment_simple_obstacle.yaml") -> None:
    """Full offline training pipeline.

    Generates synthetic training data, trains both neural network models and saves
    checkpoints to data/models/. Called via: uv run adversarial-planning-train

    Args:
        config_path: Path to YAML experiment configuration file
    """
    print("Adversarial Motion Planning — Training")
    print("=" * 50)

    config = load_config(config_path)
    print(f"Config: {config_path}")
    print(f"Seed:   {config.simulation_params.random_seed}")

    key = jax.random.PRNGKey(config.simulation_params.random_seed)
    obs_key, irl_key = jax.random.split(key)
    _OBSERVER_CHECKPOINT = config.deceptive_agent_config.observer.checkpoint_path
    _IRL_CHECKPOINT = config.interceptor_agent_config.irl.checkpoint_path

    print("\n[1/2] Training observer RNN...")
    _observer = train_observer_from_config(config, obs_key)

    print("\n[2/2] Training IRL reward function...")
    _irl_model = train_irl_from_config(config, irl_key)

    print("\nTraining complete.")

    print(f"  Observer checkpoint : {_OBSERVER_CHECKPOINT}")
    print(f"  IRL checkpoint      : {_IRL_CHECKPOINT}")


def train_observer_from_config(
    config: SimulationConfig,
    key,
):
    """Generate trajectory data and train the RNN observer. Saves checkpoint.

    Args:
        config: Simulation configuration
        key: JAX PRNG key

    Returns:
        Trained TrajectoryClassifier
    """
    import jax.numpy as jnp

    from src.data.generators import generate_optimal_trajectories
    from src.deceptive.observer import (
        load_obs_training_state,
        save_obs_training_state,
        train_observer,
    )

    workspace = create_workspace_from_config(config.workspace)
    goals = jnp.array(config.deceptive_agent_config.candidate_goals)
    num_goals = int(goals.shape[0])

    tr = config.training.observer
    obs_config = ObserverTrainingConfig(
        hidden_dim=config.deceptive_agent_config.observer.hidden_size,
        num_epochs=tr.num_epochs,
        learning_rate=tr.learning_rate,
        batch_size=tr.batch_size,
        samples_per_goal=tr.samples_per_goal,
    )
    # Attach num_goals as a dynamic attribute for use in train_observer
    object.__setattr__(obs_config, "num_goals", num_goals)

    # Resume from checkpoint if available
    _OBSERVER_CHECKPOINT = config.deceptive_agent_config.observer.checkpoint_path
    resume_state = load_obs_training_state(_OBSERVER_CHECKPOINT, obs_config)
    if resume_state is not None:
        initial_model, initial_opt_state, start_epoch = resume_state
        print(
            f"  Resuming Observer training from epoch {start_epoch} → {start_epoch + tr.num_epochs}"
        )
    else:
        initial_model, initial_opt_state, start_epoch = None, None, 0
        print(f"  Starting Observer training from scratch (0 → {tr.num_epochs} epochs)")


    key, data_key = jax.random.split(key)
    print(f"  Generating {tr.samples_per_goal} trajectories × {num_goals} goals...")
    dataset = generate_optimal_trajectories(
        workspace, goals, tr.samples_per_goal, data_key
    )
    print(f"  Dataset size: {len(dataset.trajectories)} trajectories")

    key, train_key = jax.random.split(key)
    model, opt_state, loss_history = train_observer(
        dataset,
        obs_config,
        train_key,
        initial_model=initial_model,
        initial_opt_state=initial_opt_state,
        start_epoch=start_epoch,
    )

    Path(_OBSERVER_CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)
    save_obs_training_state(
        _OBSERVER_CHECKPOINT, model, opt_state, start_epoch + tr.num_epochs
    )
    print(f"  Saved → {_OBSERVER_CHECKPOINT}")

    _save_loss_csv(Path(_OBSERVER_CHECKPOINT).parent / "observer_training_loss.csv", loss_history)
    model = None
    return model


def train_irl_from_config(
    config: SimulationConfig,
    key,
):
    """Generate demonstrations and train the IRL reward function. Saves checkpoint.

    If a checkpoint and optimizer-state sidecar already exist, training resumes
    from the last completed epoch rather than starting from scratch.

    Args:
        config: Simulation configuration
        key: JAX PRNG key

    Returns:
        Trained LearnedRewardFunction
    """
    import jax.numpy as jnp

    from src.data.generators import generate_irl_demonstrations
    from src.interceptor.irl import (
        load_irl_training_state,
        save_irl_training_state,
        maximum_entropy_irl,
    )

    workspace = create_workspace_from_config(config.workspace)
    goals = jnp.array(config.deceptive_agent_config.candidate_goals)

    tr = config.training.irl
    irl_config = IRLTrainingConfig(
        hidden_dim=config.interceptor_agent_config.irl.feature_dim,
        num_epochs=tr.num_epochs,
        learning_rate=tr.learning_rate,
        num_demonstrations=tr.num_demonstrations,
    )

    # Resume from checkpoint if available
    _IRL_CHECKPOINT = config.interceptor_agent_config.irl.checkpoint_path
    resume_state = load_irl_training_state(_IRL_CHECKPOINT, irl_config)
    if resume_state is not None:
        initial_model, initial_opt_state, start_epoch = resume_state
        print(
            f"  Resuming IRL training from epoch {start_epoch} → {start_epoch + tr.num_epochs}"
        )
    else:
        initial_model, initial_opt_state, start_epoch = None, None, 0
        print(f"  Starting IRL training from scratch (0 → {tr.num_epochs} epochs)")

    key, data_key = jax.random.split(key)
    print(f"  Generating {tr.num_demonstrations} IRL demonstrations...")
    demonstrations = generate_irl_demonstrations(
        workspace, goals, tr.num_demonstrations, data_key
    )
    print(f"  Demonstrations: {len(demonstrations)}")

    key, train_key = jax.random.split(key)
    model, opt_state, loss_history = maximum_entropy_irl(
        demonstrations,
        goals,
        irl_config,
        train_key,
        initial_model=initial_model,
        initial_opt_state=initial_opt_state,
        start_epoch=start_epoch,
    )

    Path(_IRL_CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)
    save_irl_training_state(
        _IRL_CHECKPOINT, model, opt_state, start_epoch + tr.num_epochs
    )
    print(f"  Saved → {_IRL_CHECKPOINT}  (epoch {start_epoch + tr.num_epochs})")

    _append_loss_csv(Path(_IRL_CHECKPOINT).parent / "irl_training_loss.csv", loss_history, start_epoch)
    return model


def _save_loss_csv(path: (str | Path), loss_history) -> None:
    """Write a per-epoch loss list to a CSV file (overwrites)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for epoch, loss in enumerate(loss_history):
            writer.writerow([epoch, loss])


def _append_loss_csv(path: (str | Path), loss_history, start_epoch: int) -> None:
    """Append new epochs to an existing loss CSV, or create it if missing."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(path).exists() or start_epoch == 0
    with open(path, "a" if not write_header else "w", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "loss"])
        for i, loss in enumerate(loss_history):
            writer.writerow([start_epoch + i, loss])
