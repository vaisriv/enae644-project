"""Simulation framework for adversarial evaluation."""

from src.simulation.controller import run_simulation, SimulationResult
from src.simulation.config import (
    SimulationConfig,
    WorkspaceConfig,
    DeceptiveAgentConfig,
    InterceptorAgentConfig,
    SimulationParams,
    TrainingConfig,
    ObserverTrainingConfig,
    IRLTrainingConfig,
    load_config,
)
from src.simulation.metrics import (
    compute_observer_accuracy,
    compute_path_length_ratio,
    compute_belief_entropy_over_time,
)
from src.simulation.visualization import (
    plot_workspace_with_trajectories,
    plot_belief_evolution,
    save_trajectories,
)

__all__ = [
    # Controller
    "run_simulation",
    "SimulationResult",
    # Configuration
    "SimulationConfig",
    "WorkspaceConfig",
    "DeceptiveAgentConfig",
    "InterceptorAgentConfig",
    "SimulationParams",
    "TrainingConfig",
    "ObserverTrainingConfig",
    "IRLTrainingConfig",
    "load_config",
    # Metrics
    "compute_observer_accuracy",
    "compute_path_length_ratio",
    "compute_belief_entropy_over_time",
    # Visualization
    "plot_workspace_with_trajectories",
    "plot_belief_evolution",
    "save_trajectories",
]
