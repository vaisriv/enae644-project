"""Pytest configuration and shared fixtures for all tests.

This module provides common fixtures used across test modules.
"""

import pytest
import jax
import jax.numpy as jnp
from src.simulation.config import (
    WorkspaceConfig,
    ObstacleConfig,
    DeceptiveAgentConfig,
    InterceptorAgentConfig,
    SimulationParameters,
    SimulationConfig,
    PlannerConfig,
    ObserverConfig,
    IRLConfig,
    ParticleFilterConfig,
    MPCConfig,
)
from src.shared.workspace import Workspace, CircleObstacle, PolygonObstacle
from src.shared.trajectory import Trajectory


# ============================================================================
# JAX Random Keys
# ============================================================================

@pytest.fixture
def jax_key():
    """Provide deterministic JAX PRNG key for testing."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def jax_key_sequence():
    """Provide a sequence of JAX PRNG keys for tests needing multiple keys."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)
    return keys


# ============================================================================
# Workspace Fixtures
# ============================================================================

@pytest.fixture
def simple_workspace():
    """Simple workspace with no obstacles (10x10 square)."""
    bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
    return Workspace(bounds=bounds, obstacles=[])


@pytest.fixture
def workspace_with_circle_obstacle():
    """Workspace with a single circle obstacle in the center."""
    bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
    circle = CircleObstacle(
        center=jnp.array([5.0, 5.0]),
        radius=1.0
    )
    return Workspace(bounds=bounds, obstacles=[circle])


@pytest.fixture
def workspace_with_polygon_obstacle():
    """Workspace with a single triangular obstacle."""
    bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
    triangle = PolygonObstacle(
        vertices=jnp.array([
            [3.0, 3.0],
            [7.0, 3.0],
            [5.0, 7.0]
        ])
    )
    return Workspace(bounds=bounds, obstacles=[triangle])


@pytest.fixture
def workspace_with_multiple_obstacles():
    """Workspace with both circle and polygon obstacles."""
    bounds = jnp.array([[0.0, 10.0], [0.0, 10.0]])
    obstacles = [
        CircleObstacle(center=jnp.array([2.0, 2.0]), radius=0.5),
        CircleObstacle(center=jnp.array([8.0, 8.0]), radius=0.7),
        PolygonObstacle(vertices=jnp.array([
            [4.0, 4.0],
            [6.0, 4.0],
            [6.0, 6.0],
            [4.0, 6.0]
        ]))
    ]
    return Workspace(bounds=bounds, obstacles=obstacles)


# ============================================================================
# Trajectory Fixtures
# ============================================================================

@pytest.fixture
def straight_line_trajectory():
    """Simple straight-line trajectory from (0,0) to (10,0)."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    positions = jnp.array([
        [0.0, 0.0],
        [2.5, 0.0],
        [5.0, 0.0],
        [7.5, 0.0],
        [10.0, 0.0]
    ])
    velocities = jnp.array([
        [2.5, 0.0],
        [2.5, 0.0],
        [2.5, 0.0],
        [2.5, 0.0],
        [2.5, 0.0]
    ])
    return Trajectory(times=times, positions=positions, velocities=velocities)


@pytest.fixture
def curved_trajectory():
    """Curved trajectory forming a quarter circle."""
    times = jnp.linspace(0.0, jnp.pi / 2, 10)
    positions = jnp.stack([
        jnp.cos(times),
        jnp.sin(times)
    ], axis=1)
    velocities = jnp.stack([
        -jnp.sin(times),
        jnp.cos(times)
    ], axis=1)
    return Trajectory(times=times, positions=positions, velocities=velocities)


@pytest.fixture
def stationary_trajectory():
    """Trajectory with no movement (agent stays at origin)."""
    times = jnp.array([0.0, 1.0, 2.0])
    positions = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    velocities = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    return Trajectory(times=times, positions=positions, velocities=velocities)


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def workspace_config_simple():
    """Simple workspace configuration (no obstacles)."""
    return WorkspaceConfig(
        bounds=[[0.0, 10.0], [0.0, 10.0]],
        obstacles=[]
    )


@pytest.fixture
def workspace_config_with_obstacles():
    """Workspace configuration with obstacles."""
    return WorkspaceConfig(
        bounds=[[0.0, 10.0], [0.0, 10.0]],
        obstacles=[
            ObstacleConfig(
                type="circle",
                params={"center": [5.0, 5.0], "radius": 1.0}
            ),
            ObstacleConfig(
                type="polygon",
                params={"vertices": [[2.0, 2.0], [3.0, 2.0], [2.5, 3.0]]}
            )
        ]
    )


@pytest.fixture
def planner_config():
    """Default RRT* planner configuration."""
    return PlannerConfig(
        max_iterations=1000,  # Reduced for faster tests
        step_size=0.5,
        goal_radius=0.5,
        goal_bias_probability=0.1,
        gamma=2.0,
        max_radius=3.0,
        deception_weight=0.3
    )


@pytest.fixture
def observer_config():
    """Default observer network configuration."""
    return ObserverConfig(
        checkpoint_path="models/observer_test.eqx",
        num_goals=3,
        hidden_size=32
    )


@pytest.fixture
def deceptive_agent_config(planner_config, observer_config):
    """Default deceptive agent configuration."""
    return DeceptiveAgentConfig(
        initial_position=[1.0, 1.0],
        true_goal=[9.0, 9.0],
        candidate_goals=[[9.0, 9.0], [9.0, 1.0], [1.0, 9.0]],
        planner=planner_config,
        observer=observer_config
    )


@pytest.fixture
def interceptor_agent_config():
    """Default interceptor agent configuration."""
    return InterceptorAgentConfig(
        initial_position=[1.0, 9.0],
        candidate_goals=[[9.0, 9.0], [9.0, 1.0], [1.0, 9.0]],
        irl=IRLConfig(
            checkpoint_path="models/irl_test.eqx",
            feature_dim=16,
            learning_rate=0.001
        ),
        particle_filter=ParticleFilterConfig(
            num_particles=100,  # Reduced for faster tests
            resample_threshold=0.5
        ),
        mpc=MPCConfig(
            horizon=10,  # Reduced for faster tests
            dt=0.1,
            control_weight=0.01,
            learning_rate=0.1,
            max_iterations=50  # Reduced for faster tests
        )
    )


@pytest.fixture
def simulation_parameters():
    """Default simulation parameters."""
    return SimulationParameters(
        timestep=0.1,
        max_time=10.0,  # Reduced for faster tests
        intercept_threshold=0.5,
        goal_radius=0.5,
        random_seed=42
    )


@pytest.fixture
def minimal_simulation_config(
    workspace_config_simple,
    deceptive_agent_config,
    interceptor_agent_config,
    simulation_parameters
):
    """Minimal valid simulation configuration for testing."""
    return SimulationConfig(
        workspace=workspace_config_simple,
        deceptive_agent=deceptive_agent_config,
        interceptor_agent=interceptor_agent_config,
        simulation=simulation_parameters
    )


# ============================================================================
# Mock Models
# ============================================================================

@pytest.fixture
def mock_observer_net():
    """Mock observer network for testing (returns uniform distribution)."""
    class MockObserver:
        def __call__(self, trajectory_positions):
            """Return uniform distribution over 3 goals."""
            num_goals = 3
            return jnp.ones(num_goals) / num_goals

    return MockObserver()


@pytest.fixture
def mock_irl_model():
    """Mock IRL reward model for testing (returns constant reward)."""
    class MockIRLModel:
        def __call__(self, state, action):
            """Return constant reward."""
            return 0.0

    return MockIRLModel()


# ============================================================================
# Test Data Paths
# ============================================================================

@pytest.fixture
def test_data_dir(tmp_path):
    """Provide temporary directory for test data."""
    return tmp_path


@pytest.fixture
def sample_config_yaml(test_data_dir):
    """Create a sample YAML config file for testing."""
    config_content = """
workspace:
  bounds: [[0.0, 10.0], [0.0, 10.0]]
  obstacles: []

deceptive_agent:
  initial_position: [1.0, 1.0]
  true_goal: [9.0, 9.0]
  candidate_goals: [[9.0, 9.0], [9.0, 1.0], [1.0, 9.0]]
  planner:
    max_iterations: 1000
    step_size: 0.5
    goal_radius: 0.5
    goal_bias_probability: 0.1
    gamma: 2.0
    max_radius: 3.0
    deception_weight: 0.3
  observer:
    checkpoint_path: "models/observer_test.eqx"
    num_goals: 3
    hidden_size: 32

interceptor_agent:
  initial_position: [1.0, 9.0]
  candidate_goals: [[9.0, 9.0], [9.0, 1.0], [1.0, 9.0]]
  irl:
    checkpoint_path: "models/irl_test.eqx"
    feature_dim: 16
    learning_rate: 0.001
  particle_filter:
    num_particles: 100
    resample_threshold: 0.5
  mpc:
    horizon: 10
    dt: 0.1
    control_weight: 0.01
    learning_rate: 0.1
    max_iterations: 50

simulation:
  timestep: 0.1
  max_time: 10.0
  intercept_threshold: 0.5
  goal_radius: 0.5
  random_seed: 42
"""
    config_path = test_data_dir / "test_config.yaml"
    config_path.write_text(config_content)
    return str(config_path)
