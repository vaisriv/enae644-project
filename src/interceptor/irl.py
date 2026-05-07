"""Inverse reinforcement learning for behavioral modeling."""

import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from src.shared.trajectory import Trajectory, create_trajectory
from src.shared.workspace import Workspace
from src.simulation.config import IRLConfig, IRLTrainingConfig

_SIDECAR_SUFFIX = ".optstate.pkl"


class LearnedRewardFunction(eqx.Module):
    """Neural network parameterization of the learned reward function.

    Maps a (state, action) pair to a scalar reward.
    """

    layers: List[eqx.nn.Linear]
    hidden_dim: int = eqx.field(static=True)

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, key):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(state_dim + action_dim, hidden_dim, key=keys[0]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            eqx.nn.Linear(hidden_dim, 1, key=keys[2]),
        ]
        self.hidden_dim = hidden_dim

    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Compute reward for a state-action pair.

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


def save_irl_training_state(
    checkpoint_path: str,
    model: "LearnedRewardFunction",
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


def load_irl_training_state(
    checkpoint_path: str,
    config: IRLTrainingConfig,
) -> Optional[Tuple["LearnedRewardFunction", object, int]]:
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

    irl_cfg = IRLConfig(
        checkpoint_path=checkpoint_path,
        feature_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
    )
    model = load_irl_model(checkpoint_path, irl_cfg)

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

    loss_csv = Path(checkpoint_path).parent / "irl_training_loss.csv"
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


def maximum_entropy_irl(
    demonstrations: List[Trajectory],
    goals: jnp.ndarray,
    config: IRLTrainingConfig,
    key,
    initial_model: Optional["LearnedRewardFunction"] = None,
    initial_opt_state=None,
    start_epoch: int = 0,
) -> Tuple["LearnedRewardFunction", Any, List[float]]:
    """Learn a reward function via contrastive/NCE approximation of MaxEnt IRL.

    For each expert (state, action) pair we maximize R(state, action) while
    minimizing R(state, random_action) for several noise samples. This produces
    a reward that assigns high values to goal-directed behaviour — the continuous
    equivalent of the MaxEnt gradient step without requiring value iteration.

    To resume a previous run, pass the values returned by
    :func:`load_irl_training_state` as ``initial_model``, ``initial_opt_state``,
    and ``start_epoch``.  The returned loss history covers only the epochs run
    in this call; the caller is responsible for appending it to any prior history.

    Args:
        demonstrations: List of expert Trajectory objects
        goals: (num_goals, 2) candidate goal positions
        config: IRLTrainingConfig with hidden_dim, num_epochs, learning_rate
        key: JAX PRNG key
        initial_model: Pre-trained model to continue from (None → fresh init)
        initial_opt_state: Optimizer state to restore (None → fresh init)
        start_epoch: Epoch offset used for progress logging only

    Returns:
        (trained_model, loss_history) per-epoch mean loss for epochs run here
    """
    key, init_key = jax.random.split(key)
    if initial_model is not None:
        reward_fn = initial_model
    else:
        reward_fn = LearnedRewardFunction(
            state_dim=2, action_dim=2, hidden_dim=config.hidden_dim, key=init_key
        )

    optimizer = optax.adam(config.learning_rate)
    if initial_opt_state is not None:
        opt_state = initial_opt_state
    else:
        opt_state = optimizer.init(eqx.filter(reward_fn, eqx.is_array))

    # Build flat list of (state, action) pairs from demonstrations
    sa_pairs: List[Tuple[jnp.ndarray, jnp.ndarray]] = []
    for traj in demonstrations:
        pos = traj.positions
        for i in range(len(pos) - 1):
            state = pos[i]
            action = pos[i + 1] - pos[i]
            action_norm = jnp.linalg.norm(action)
            if float(action_norm) > 1e-6:
                action = action / action_norm * 0.5  # normalise to step_size
            sa_pairs.append((state, action))

    if not sa_pairs:
        return reward_fn, opt_state, []

    num_neg = 8  # negative samples per positive

    @eqx.filter_jit
    def train_step(reward_fn, opt_state, state, expert_action, neg_actions):
        def loss_fn(reward_fn):
            pos_reward = reward_fn(state, expert_action)
            neg_rewards = jax.vmap(lambda a: reward_fn(state, a))(neg_actions)
            # NCE loss: -R_expert + log(exp(R_expert) + Σ exp(R_neg))
            all_rewards = jnp.concatenate([pos_reward[None], neg_rewards])
            log_partition = jax.nn.logsumexp(all_rewards)
            return -pos_reward + log_partition

        loss, grads = eqx.filter_value_and_grad(loss_fn)(reward_fn)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(reward_fn, eqx.is_array)
        )
        new_reward_fn = eqx.apply_updates(reward_fn, updates)
        return new_reward_fn, new_opt_state, loss

    loss_history: List[float] = []
    n = len(sa_pairs)
    total_epochs = start_epoch + config.num_epochs

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        key, shuffle_key = jax.random.split(key)
        indices = jax.random.permutation(shuffle_key, n)

        for idx in indices:
            idx = int(idx)
            state, expert_action = sa_pairs[idx]

            key, neg_key = jax.random.split(key)
            angles = jnp.linspace(0.0, 2 * jnp.pi, num_neg, endpoint=False)
            neg_actions = jnp.stack(
                [jnp.array([jnp.cos(a), jnp.sin(a)]) * 0.5 for a in angles]
            )

            reward_fn, opt_state, loss = train_step(
                reward_fn, opt_state, state, expert_action, neg_actions
            )
            epoch_loss += float(loss)

        mean_loss = epoch_loss / n
        loss_history.append(mean_loss)
        display_epoch = start_epoch + epoch
        if display_epoch % 10 == 0:
            print(
                f"  IRL epoch {display_epoch:4d}/{total_epochs}  loss={mean_loss:.4f}"
            )

    return reward_fn, opt_state, loss_history


def predict_trajectory(
    reward_fn: "LearnedRewardFunction",
    start: jnp.ndarray,
    goal: jnp.ndarray,
    horizon: int,
    workspace: Workspace,
    step_size: float = 0.5,
) -> Trajectory:
    """Predict a trajectory under the learned reward function.

    At each step, evaluates 16 candidate actions and greedily selects the one
    with the highest reward, subject to a soft goal-attraction bias.

    Args:
        reward_fn: Trained LearnedRewardFunction
        start: (2,) starting position
        goal: (2,) goal position
        horizon: Number of steps to predict
        workspace: Environment (for bounds clamping)
        step_size: Action magnitude

    Returns:
        Trajectory from start toward goal
    """
    positions = [start]
    current = start
    angles = jnp.linspace(0.0, 2 * jnp.pi, 16, endpoint=False)
    bounds = workspace.bounds

    for _ in range(horizon):
        best_action = None
        best_reward = -float("inf")

        for angle in angles:
            action = jnp.array([jnp.cos(angle), jnp.sin(angle)]) * step_size
            next_pos = current + action

            # Stay within bounds
            if not (
                float(bounds[0, 0]) <= float(next_pos[0]) <= float(bounds[0, 1])
                and float(bounds[1, 0]) <= float(next_pos[1]) <= float(bounds[1, 1])
            ):
                continue

            r = float(reward_fn(current, action))
            # Add goal-attraction bias
            dist_improvement = float(jnp.linalg.norm(current - goal)) - float(
                jnp.linalg.norm(next_pos - goal)
            )
            r += 0.5 * dist_improvement

            if r > best_reward:
                best_reward = r
                best_action = action

        if best_action is None:
            # Fallback: move directly toward goal
            direction = goal - current
            dist = float(jnp.linalg.norm(direction))
            if dist > 1e-6:
                best_action = direction / dist * step_size
            else:
                break

        current = current + best_action
        positions.append(current)

        if float(jnp.linalg.norm(current - goal)) < step_size:
            break

    positions_arr = jnp.stack(positions)
    n = positions_arr.shape[0]
    times = jnp.arange(n, dtype=jnp.float32) * 0.1
    return create_trajectory(times, positions_arr)


def load_irl_model(path: str, config: IRLConfig) -> "LearnedRewardFunction":
    """Load a trained IRL reward function from an Equinox checkpoint.

    Args:
        path: Path to checkpoint saved with eqx.tree_serialise_leaves
        config: IRLConfig with feature_dim matching the saved model

    Returns:
        Loaded LearnedRewardFunction
    """
    skeleton = LearnedRewardFunction(
        state_dim=2,
        action_dim=2,
        hidden_dim=config.feature_dim,
        key=jax.random.PRNGKey(0),
    )
    return eqx.tree_deserialise_leaves(path, skeleton)
