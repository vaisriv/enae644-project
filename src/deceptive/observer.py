"""RNN observer network for trajectory-to-goal classification."""

import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from src.simulation.config import ObserverConfig, ObserverTrainingConfig

_SIDECAR_SUFFIX = ".optstate.pkl"

class TrajectoryClassifier(eqx.Module):
    """GRU-based trajectory classifier for goal prediction.

    Maps a partial trajectory (sequence of 2-D positions) to a probability
    distribution over candidate goals.
    """

    rnn: eqx.nn.GRUCell
    fc: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    num_goals: int = eqx.field(static=True)

    def __init__(self, input_dim: int, hidden_dim: int, num_goals: int, key):
        key1, key2 = jax.random.split(key)
        self.rnn = eqx.nn.GRUCell(
            input_size=input_dim, hidden_size=hidden_dim, key=key1
        )
        self.fc = eqx.nn.Linear(hidden_dim, num_goals, key=key2)
        self.hidden_size = hidden_dim
        self.num_goals = num_goals

    def __call__(self, trajectory_sequence: jnp.ndarray) -> jnp.ndarray:
        """Classify trajectory to predict goal distribution.

        Args:
            trajectory_sequence: (seq_len, 2) partial trajectory

        Returns:
            (num_goals,) probability distribution over goals
        """
        hidden = jnp.zeros(self.hidden_size)
        for i in range(trajectory_sequence.shape[0]):
            hidden = self.rnn(trajectory_sequence[i], hidden)
        logits = self.fc(hidden)
        return jax.nn.softmax(logits)


def save_obs_training_state(
    checkpoint_path: str,
    model: "TrajectoryClassifier",
    opt_state,
    epoch_completed: int,
) -> None:
    """Persist model weights and optimizer state so training can be resumed.

    Saves two files:
    - ``checkpoint_path``: Equinox leaf serialization of the model weights
    - ``checkpoint_path + _SIDECAR_SUFFIX``: pickle of (opt_state, epoch_completed)
    """
    eqx.tree_serialise_leaves(checkpoint_path, model)
    sidecar = str(checkpoint_path) + _SIDECAR_SUFFIX
    with open(sidecar, "wb") as f:
        pickle.dump({"opt_state": opt_state, "epoch_completed": epoch_completed}, f)


def load_obs_training_state(
    checkpoint_path: str,
    config: ObserverTrainingConfig,
) -> Optional[Tuple["TrajectoryClassifier", object, int]]:
    """Load a previously saved training state, or return None if no model exists.

    Three cases:
    - No ``.eqx`` file → return None (fresh start).
    - ``.eqx`` + sidecar both exist → full resume with weights, optimizer state,
      and epoch counter.
    - ``.eqx`` exists but no sidecar → weights-only resume: load model, create a
      fresh optimizer state, and infer the completed epoch count from the loss CSV
      (falls back to 0 if the CSV is absent).  Adam momentum is lost but the
      trained weights are preserved.

    Args:
        checkpoint_path: Path used when saving (the ``.eqx`` file).
        config: Must match the hidden_dim used during the original run.

    Returns:
        ``(model, opt_state, epoch_completed)`` or ``None`` if no checkpoint exists.
    """
    if not Path(checkpoint_path).exists():
        return None

    observer_cfg = ObserverConfig(
        checkpoint_path=checkpoint_path,
        num_goals=config.num_goals, # type: ignore[unresolved-attribute]
        hidden_size=config.hidden_dim,
    )
    model = load_observer(checkpoint_path, observer_cfg)

    sidecar = str(checkpoint_path) + _SIDECAR_SUFFIX
    if Path(sidecar).exists():
        with open(sidecar, "rb") as f:
            data = pickle.load(f)
        return model, data["opt_state"], data["epoch_completed"]

    # Weights-only fallback: reconstruct a fresh optimizer state and estimate epoch
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    epoch_completed = _infer_epoch_from_loss_csv(checkpoint_path)
    return model, opt_state, epoch_completed


def _infer_epoch_from_loss_csv(checkpoint_path: str) -> int:
    """Return the highest epoch number in the loss CSV adjacent to checkpoint_path."""
    import csv as _csv

    loss_csv = Path(checkpoint_path).parent / "observer_training_loss.csv"
    if not loss_csv.exists():
        return 0
    try:
        with open(loss_csv, newline="") as f:
            rows = list(_csv.DictReader(f))
        if rows:
            return int(rows[-1]["epoch"]) + 1
    except Exception:
        pass
    return 0


def train_observer(
    dataset,  # TrajectoryDataset
    config: ObserverTrainingConfig,
    key,
    initial_model: Optional["TrajectoryClassifier"] = None,
    initial_opt_state=None,
    start_epoch: int = 0,
) -> Tuple[TrajectoryClassifier, Any, List[float]]:
    """Train RNN observer on trajectory classification.

    Args:
        dataset: TrajectoryDataset with trajectories and goal_ids
        config: Training hyperparameters (must include num_goals field)
        key: JAX PRNG key

    Returns:
        (trained_model, loss_history) where loss_history is per-epoch mean loss
    """
    num_goals = int(getattr(config, "num_goals", len(dataset.goals)))

    key, init_key = jax.random.split(key)
    model = TrajectoryClassifier(
        input_dim=2,
        hidden_dim=config.hidden_dim,
        num_goals=num_goals,
        key=init_key,
    )

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, traj, goal_id):
        def loss_fn(model):
            pred_probs = model(traj)
            return -jnp.log(pred_probs[goal_id] + 1e-8)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    n = len(dataset.trajectories)
    loss_history: List[float] = []

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        # Shuffle indices
        key, shuffle_key = jax.random.split(key)
        indices = jax.random.permutation(shuffle_key, n)

        for idx in indices:
            idx = int(idx)
            traj = dataset.trajectories[idx]
            goal_id = dataset.goal_ids[idx]
            model, opt_state, loss = train_step(model, opt_state, traj, goal_id)
            epoch_loss += float(loss)

        mean_loss = epoch_loss / n
        loss_history.append(mean_loss)
        if epoch % 10 == 0:
            print(
                f"  Observer epoch {epoch:4d}/{config.num_epochs}  loss={mean_loss:.4f}"
            )

    return model, opt_state, loss_history


def load_observer(path: str, config: ObserverConfig) -> TrajectoryClassifier:
    """Load a trained observer from an Equinox checkpoint.

    Args:
        path: Path to checkpoint file saved with eqx.tree_serialise_leaves
        config: ObserverConfig with hidden_size and num_goals matching the checkpoint

    Returns:
        Loaded TrajectoryClassifier
    """
    skeleton = TrajectoryClassifier(
        input_dim=2,
        hidden_dim=config.hidden_size,
        num_goals=config.num_goals,
        key=jax.random.PRNGKey(0),
    )
    return eqx.tree_deserialise_leaves(path, skeleton)
