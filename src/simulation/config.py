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
class SimulationParameters:
    """Core simulation parameters."""

    timestep: float = 0.1  # Δt for discrete time stepping
    max_time: float = 100.0  # Timeout
    intercept_threshold: float = 0.5  # Distance for successful interception
    goal_radius: float = 0.5  # Distance for goal reaching
    random_seed: int = 42


@dataclass
class SimulationConfig:
    """Complete simulation configuration.

    This is the top-level config object loaded from YAML.
    """

    workspace: WorkspaceConfig
    deceptive_agent: DeceptiveAgentConfig
    interceptor_agent: InterceptorAgentConfig
    simulation: SimulationParameters


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
        >>> config = load_config("configs/experiment_01.yaml")
        >>> print(config.deceptive_agent.true_goal)
    """
    # TODO: Implement YAML loading
    # 1. Check if file exists
    # 2. Load YAML using yaml.safe_load()
    # 3. Parse into dataclasses
    # 4. Validate constraints (see _validate_config)
    # 5. Return SimulationConfig
    raise NotImplementedError("load_config not implemented")


def _validate_config(config: SimulationConfig) -> None:
    """Validate configuration constraints.

    Args:
        config: SimulationConfig to validate

    Raises:
        ValueError: If any validation check fails

    Validation checks:
        1. True goal must be in candidate goals
        2. Candidate goals must match between agents
        3. Initial positions must be in workspace bounds
        4. Initial positions must be collision-free
        5. Deception weight α ∈ [0, 1]
        6. All numeric parameters are positive where applicable
    """
    # TODO: Implement validation checks
    # 1. Check true goal in candidates:
    #    assert config.deceptive_agent.true_goal in config.deceptive_agent.candidate_goals

    # 2. Check candidate goals match:
    #    assert config.deceptive_agent.candidate_goals == config.interceptor_agent.candidate_goals

    # 3. Check deception weight:
    #    assert 0.0 <= config.deceptive_agent.planner.deception_weight <= 1.0

    # 4. Check bounds:
    #    bounds = config.workspace.bounds
    #    assert bounds[0][0] < bounds[0][1]  # x_min < x_max
    #    assert bounds[1][0] < bounds[1][1]  # y_min < y_max

    # 5. Check positions in bounds:
    #    for pos in [config.deceptive_agent.initial_position, config.interceptor_agent.initial_position]:
    #        assert bounds[0][0] <= pos[0] <= bounds[0][1]
    #        assert bounds[1][0] <= pos[1] <= bounds[1][1]

    # 6. Check positive parameters:
    #    assert config.simulation.timestep > 0
    #    assert config.simulation.max_time > 0
    #    etc.

    raise NotImplementedError("_validate_config not implemented")


def save_config(config: SimulationConfig, yaml_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: SimulationConfig to save
        yaml_path: Output path for YAML file
    """
    # TODO: Implement config serialization
    # Convert dataclasses to dict and save with yaml.dump()
    raise NotImplementedError("save_config not implemented")


def config_to_dict(config: SimulationConfig) -> Dict[str, Any]:
    """Convert SimulationConfig to dictionary for serialization.

    Args:
        config: SimulationConfig object

    Returns:
        Nested dictionary representation
    """
    # TODO: Implement recursive dataclass to dict conversion
    # Use dataclasses.asdict() or manual conversion
    raise NotImplementedError("config_to_dict not implemented")


def dict_to_config(data: Dict[str, Any]) -> SimulationConfig:
    """Convert dictionary to SimulationConfig.

    Args:
        data: Nested dictionary from YAML

    Returns:
        SimulationConfig object
    """
    # TODO: Implement dictionary to dataclass conversion
    # Parse nested dictionaries into appropriate dataclass instances
    raise NotImplementedError("dict_to_config not implemented")
