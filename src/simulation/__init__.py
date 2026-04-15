"""Simulation module for adversarial motion planning.

This module provides the main simulation controller, configuration loading,
metrics computation, and visualization utilities.
"""

# Export main simulation components
from src.simulation.controller import run_simulation, SimulationResult
from src.simulation.config import (
    SimulationConfig,
    WorkspaceConfig,
    DeceptiveAgentConfig,
    InterceptorAgentConfig,
    SimulationParameters,
    load_config,
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
    "SimulationParameters",
    "load_config",
]
