"""Offline model training for adversarial motion planning.

Trains the neural network models required by the simulation:
1. RNN observer (TrajectoryClassifier) for Agent D's deception cost
2. IRL reward function (LearnedRewardFunction) for Agent I's behavioral model

Checkpoints are saved to outputs/models/ and loaded at simulation time by
uv run adversarial-planning. This script is run once offline via:
    uv run adversarial-planning-train [--config path/to/config.yaml]
"""

from typing import TYPE_CHECKING, Any

from src.simulation.config import SimulationConfig, load_config

if TYPE_CHECKING:
    from jaxtyping import PRNGKeyArray

_OBSERVER_CHECKPOINT = "outputs/models/observer_rnn.eqx"
_IRL_CHECKPOINT = "outputs/models/irl_reward.eqx"


def train(config_path: str = "data/configs/experiment_simple_obstacle.yaml") -> None:
    """Full offline training pipeline.

    Generates synthetic training data, trains both neural network models, and saves
    checkpoints to outputs/models/. Called via: uv run adversarial-planning-train

    Args:
        config_path: Path to YAML experiment configuration file
    """
    # TODO: Implement training pipeline
    # 1. Parse CLI args (config path, optional --seed override)
    # 2. config = load_config(config_path)
    # 3. key = jax.random.PRNGKey(config.simulation_params.random_seed)
    # 4. obs_key, irl_key = jax.random.split(key)
    # 5. train_observer_from_config(config, obs_key)  → saves observer_rnn.eqx
    # 6. train_irl_from_config(config, irl_key)       → saves irl_reward.eqx
    # 7. Print summary of training metrics and checkpoint paths
    raise NotImplementedError("train not implemented")


def train_observer_from_config(
    config: "SimulationConfig",
    key: "PRNGKeyArray",
) -> Any:
    """Generate trajectory data and train the RNN observer. Saves checkpoint.

    Generates optimal (non-deceptive) RRT* trajectories for all candidate goals,
    trains a TrajectoryClassifier via cross-entropy loss, and serialises the result
    to outputs/models/observer_rnn.eqx using eqx.tree_serialise_leaves.

    Args:
        config: Simulation configuration. Uses config.training.observer for
                hyperparameters and config.deceptive_agent_config.candidate_goals
                for goal set.
        key: JAX PRNG key

    Returns:
        Trained TrajectoryClassifier
    """
    # TODO: Implement observer training
    # dataset = data.generators.generate_optimal_trajectories(
    #     workspace, goals, config.training.observer.samples_per_goal, key
    # )
    # observer_net = deceptive.observer.train_observer(
    #     dataset, config.training.observer, key
    # )
    # eqx.tree_serialise_leaves(_OBSERVER_CHECKPOINT, observer_net)
    # return observer_net
    raise NotImplementedError("train_observer_from_config not implemented")


def train_irl_from_config(
    config: "SimulationConfig",
    key: "PRNGKeyArray",
) -> Any:
    """Generate demonstrations and train the IRL reward function. Saves checkpoint.

    Plans non-deceptive RRT* trajectories (alpha=0) as expert demonstrations, trains
    a LearnedRewardFunction via maximum entropy IRL, and serialises the result to
    outputs/models/irl_reward.eqx using eqx.tree_serialise_leaves.

    Args:
        config: Simulation configuration. Uses config.training.irl for
                hyperparameters and config.deceptive_agent_config.candidate_goals
                for goal set.
        key: JAX PRNG key

    Returns:
        Trained LearnedRewardFunction
    """
    # TODO: Implement IRL training
    # demonstrations = data.generators.generate_irl_demonstrations(
    #     workspace, goals, config.training.irl.num_demonstrations, key
    # )
    # reward_fn = interceptor.irl.maximum_entropy_irl(
    #     demonstrations, goals, config.training.irl, key
    # )
    # eqx.tree_serialise_leaves(_IRL_CHECKPOINT, reward_fn)
    # return reward_fn
    raise NotImplementedError("train_irl_from_config not implemented")
