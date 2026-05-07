"""Configuration loading and validation for simulation parameters.

This module provides dataclasses for all configuration sections and
functions to load/validate YAML configuration files.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ObstacleConfig:
    """Configuration for a single obstacle."""

    type: str  # "circle" or "polygon"
    params: Dict[str, Any]  # For circle: {center: [x, y], radius: r}
    # For polygon: {vertices: [[x1, y1], [x2, y2], ...]}


@dataclass
class WorkspaceConfig:
    """2D workspace configuration.

    Attributes:
        bounds: [[x_min, x_max], [y_min, y_max]]
        obstacles: List of obstacle configurations
    """

    bounds: List[List[float]]
    obstacles: List[ObstacleConfig] = field(default_factory=list)


@dataclass
class PlannerConfig:
    """RRT* planner configuration for deceptive agent."""

    max_iterations: int = 5000
    step_size: float = 0.5
    goal_radius: float = 0.5
    goal_bias_probability: float = 0.1
    gamma: float = 2.0  # Rewiring radius scaling
    max_radius: float = 3.0
    deception_weight: float = 0.3  # α ∈ [0, 1]


@dataclass
class ObserverConfig:
    """Observer network configuration for deceptive agent."""

    checkpoint_path: str
    num_goals: int
    hidden_size: int = 64


@dataclass
class DeceptiveAgentConfig:
    """Deceptive agent (Agent D) configuration.

    Attributes:
        initial_position: [x, y] starting position
        true_goal: [x, y] actual goal (hidden from interceptor)
        candidate_goals: List of [x, y] possible goals (includes true goal)
        planner: RRT* planner parameters
        observer: Observer network parameters
    """

    initial_position: List[float]
    true_goal: List[float]
    candidate_goals: List[List[float]]
    planner: PlannerConfig
    observer: ObserverConfig


@dataclass
class IRLConfig:
    """Inverse reinforcement learning configuration."""

    checkpoint_path: str
    feature_dim: int = 32
    learning_rate: float = 0.001


@dataclass
class ParticleFilterConfig:
    """Particle filter configuration for belief tracking."""

    num_particles: int = 1000
    resample_threshold: float = 0.5  # Effective sample size threshold


@dataclass
class MPCConfig:
    """Model predictive control configuration."""

    horizon: int = 20
    dt: float = 0.1
    control_weight: float = 0.01  # λ_u regularization
    learning_rate: float = 0.1
    max_iterations: int = 100


@dataclass
class InterceptorAgentConfig:
    """Interceptor agent (Agent I) configuration.

    Attributes:
        initial_position: [x, y] starting position
        candidate_goals: List of [x, y] possible goals for Agent D
        irl: IRL model parameters
        particle_filter: Particle filter parameters
        mpc: MPC planner parameters
    """

    initial_position: List[float]
    candidate_goals: List[List[float]]
    irl: IRLConfig
    particle_filter: ParticleFilterConfig
    mpc: MPCConfig


@dataclass
class SimulationParams:
    """Core simulation parameters."""

    timestep: float = 0.1  # Δt for discrete time stepping
    max_time: float = 100.0  # Timeout
    intercept_threshold: float = 0.5  # Distance for successful interception
    goal_radius: float = 0.5  # Distance for goal reaching
    random_seed: int = 42


@dataclass
class ObserverTrainingConfig:
    """Hyperparameters for training the RNN observer network."""

    hidden_dim: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    samples_per_goal: int = 200


@dataclass
class IRLTrainingConfig:
    """Hyperparameters for training the IRL reward function."""

    hidden_dim: int = 64
    num_epochs: int = 50
    learning_rate: float = 1e-3
    num_demonstrations: int = 500


@dataclass
class TrainingConfig:
    """Configuration for offline model training (uv run adversarial-planning-train)."""

    observer: ObserverTrainingConfig = field(default_factory=ObserverTrainingConfig)
    irl: IRLTrainingConfig = field(default_factory=IRLTrainingConfig)


@dataclass
class SimulationConfig:
    """Complete simulation configuration.

    This is the top-level config object loaded from YAML.
    """

    workspace: WorkspaceConfig
    deceptive_agent_config: DeceptiveAgentConfig
    interceptor_agent_config: InterceptorAgentConfig
    simulation_params: SimulationParams
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(yaml_path: str) -> SimulationConfig:
    """Load and validate simulation configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Validated SimulationConfig object

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If configuration is invalid

    Example:
        >>> config = load_config("data/configs/experiment_simple_obstacle.yaml")
        >>> print(config.deceptive_agent_config.true_goal)
    """
    import yaml
    from pathlib import Path

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    config = dict_to_config(data)
    _validate_config(config)
    return config


def _validate_config(config: SimulationConfig) -> None:
    """Validate configuration constraints.

    Args:
        config: SimulationConfig to validate

    Raises:
        ValueError: If any validation check fails
    """
    d_cfg = config.deceptive_agent_config
    i_cfg = config.interceptor_agent_config
    sim = config.simulation_params

    true_goal = d_cfg.true_goal
    candidates = d_cfg.candidate_goals

    # 1. True goal must be in candidate goals
    if true_goal not in candidates:
        raise ValueError(
            f"true_goal {true_goal} not found in candidate_goals {candidates}"
        )

    # 2. Candidate goals must match between agents
    if d_cfg.candidate_goals != i_cfg.candidate_goals:
        raise ValueError(
            "deceptive_agent_config.candidate_goals must match "
            "interceptor_agent_config.candidate_goals"
        )

    # 3. Deception weight α ∈ [0, 1]
    alpha = d_cfg.planner.deception_weight
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"planner.deception_weight must be in [0, 1], got {alpha}")

    # 4. Workspace bounds valid
    bounds = config.workspace.bounds
    if bounds[0][0] >= bounds[0][1]:
        raise ValueError("workspace x bounds: x_min must be < x_max")
    if bounds[1][0] >= bounds[1][1]:
        raise ValueError("workspace y bounds: y_min must be < y_max")

    # 5. Positive simulation parameters
    if sim.timestep <= 0:
        raise ValueError(f"simulation.timestep must be > 0, got {sim.timestep}")
    if sim.max_time <= 0:
        raise ValueError(f"simulation.max_time must be > 0, got {sim.max_time}")


def save_config(config: SimulationConfig, yaml_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: SimulationConfig to save
        yaml_path: Output path for YAML file
    """
    import yaml
    from pathlib import Path

    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config_to_dict(config), f, default_flow_style=False)


def config_to_dict(config: SimulationConfig) -> Dict[str, Any]:
    """Convert SimulationConfig to a YAML-compatible dictionary.

    Uses the same top-level key names as the YAML schema so that
    ``yaml.dump(config_to_dict(cfg))`` produces a file that ``load_config``
    can read back unchanged.

    Args:
        config: SimulationConfig object

    Returns:
        Nested dictionary with YAML-schema keys
    """
    import dataclasses

    raw = dataclasses.asdict(config)
    # Remap internal field names → YAML schema keys
    return {
        "workspace": raw["workspace"],
        "deceptive_agent": raw["deceptive_agent_config"],
        "interceptor_agent": raw["interceptor_agent_config"],
        "simulation": raw["simulation_params"],
        "training": raw["training"],
    }


def dict_to_config(data: Dict[str, Any]) -> SimulationConfig:
    """Convert dictionary to SimulationConfig.

    The YAML uses keys deceptive_agent / interceptor_agent / simulation while
    SimulationConfig uses deceptive_agent_config / interceptor_agent_config /
    simulation_params — this function handles the remapping.

    Args:
        data: Nested dictionary from YAML

    Returns:
        SimulationConfig object
    """
    ws_data = data["workspace"]
    workspace = WorkspaceConfig(
        bounds=ws_data["bounds"],
        obstacles=[
            ObstacleConfig(type=o["type"], params=o["params"])
            for o in ws_data.get("obstacles", [])
        ],
    )

    da_data = data["deceptive_agent"]
    p = da_data["planner"]
    planner = PlannerConfig(
        max_iterations=p.get("max_iterations", 5000),
        step_size=p.get("step_size", 0.5),
        goal_radius=p.get("goal_radius", 0.5),
        goal_bias_probability=p.get("goal_bias_probability", 0.1),
        gamma=p.get("gamma", 2.0),
        max_radius=p.get("max_radius", 3.0),
        deception_weight=p.get("deception_weight", 0.3),
    )
    obs = da_data["observer"]
    observer = ObserverConfig(
        checkpoint_path=obs["checkpoint_path"],
        num_goals=obs.get("num_goals", 3),
        hidden_size=obs.get("hidden_size", 64),
    )
    deceptive_agent_config = DeceptiveAgentConfig(
        initial_position=da_data["initial_position"],
        true_goal=da_data["true_goal"],
        candidate_goals=da_data["candidate_goals"],
        planner=planner,
        observer=observer,
    )

    ia_data = data["interceptor_agent"]
    irl_data = ia_data["irl"]
    irl = IRLConfig(
        checkpoint_path=irl_data["checkpoint_path"],
        feature_dim=irl_data.get("feature_dim", 32),
        learning_rate=irl_data.get("learning_rate", 0.001),
    )
    pf_data = ia_data["particle_filter"]
    particle_filter = ParticleFilterConfig(
        num_particles=pf_data.get("num_particles", 1000),
        resample_threshold=pf_data.get("resample_threshold", 0.5),
    )
    mpc_data = ia_data["mpc"]
    mpc = MPCConfig(
        horizon=mpc_data.get("horizon", 20),
        dt=mpc_data.get("dt", 0.1),
        control_weight=mpc_data.get("control_weight", 0.01),
        learning_rate=mpc_data.get("learning_rate", 0.1),
        max_iterations=mpc_data.get("max_iterations", 100),
    )
    interceptor_agent_config = InterceptorAgentConfig(
        initial_position=ia_data["initial_position"],
        candidate_goals=ia_data["candidate_goals"],
        irl=irl,
        particle_filter=particle_filter,
        mpc=mpc,
    )

    sim_data = data.get("simulation", {})
    simulation_params = SimulationParams(
        timestep=sim_data.get("timestep", 0.1),
        max_time=sim_data.get("max_time", 100.0),
        intercept_threshold=sim_data.get("intercept_threshold", 0.5),
        goal_radius=sim_data.get("goal_radius", 0.5),
        random_seed=sim_data.get("random_seed", 42),
    )

    tr_data = data.get("training", {})
    obs_tr = tr_data.get("observer", {})
    observer_training = ObserverTrainingConfig(
        hidden_dim=obs_tr.get("hidden_dim", 64),
        num_epochs=obs_tr.get("num_epochs", 100),
        learning_rate=obs_tr.get("learning_rate", 1e-3),
        batch_size=obs_tr.get("batch_size", 32),
        samples_per_goal=obs_tr.get("samples_per_goal", 200),
    )
    irl_tr = tr_data.get("irl", {})
    irl_training = IRLTrainingConfig(
        hidden_dim=irl_tr.get("hidden_dim", 64),
        num_epochs=irl_tr.get("num_epochs", 50),
        learning_rate=irl_tr.get("learning_rate", 1e-3),
        num_demonstrations=irl_tr.get("num_demonstrations", 500),
    )
    training = TrainingConfig(observer=observer_training, irl=irl_training)

    return SimulationConfig(
        workspace=workspace,
        deceptive_agent_config=deceptive_agent_config,
        interceptor_agent_config=interceptor_agent_config,
        simulation_params=simulation_params,
        training=training,
    )


def create_workspace_from_config(workspace_config: WorkspaceConfig):
    """Build a Workspace object from a WorkspaceConfig.

    Args:
        workspace_config: WorkspaceConfig with bounds and obstacles

    Returns:
        Workspace with parsed obstacles
    """
    import jax.numpy as jnp
    from src.shared.workspace import Workspace, CircleObstacle, PolygonObstacle

    bounds = jnp.array(workspace_config.bounds)
    obstacles = []
    for obs in workspace_config.obstacles:
        if obs.type == "circle":
            obstacles.append(
                CircleObstacle(
                    center=jnp.array(obs.params["center"]),
                    radius=float(obs.params["radius"]),
                )
            )
        elif obs.type == "polygon":
            obstacles.append(
                PolygonObstacle(vertices=jnp.array(obs.params["vertices"]))
            )
        else:
            raise ValueError(f"Unknown obstacle type: {obs.type}")

    return Workspace(bounds=bounds, obstacles=obstacles)
