"""Sample configuration generators for testing."""

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


def minimal_config() -> SimulationConfig:
    """Bare minimum valid configuration for quick tests."""
    return SimulationConfig(
        workspace=WorkspaceConfig(
            bounds=[[0.0, 10.0], [0.0, 10.0]],
            obstacles=[]
        ),
        deceptive_agent=DeceptiveAgentConfig(
            initial_position=[1.0, 1.0],
            true_goal=[9.0, 9.0],
            candidate_goals=[[9.0, 9.0], [9.0, 1.0], [1.0, 9.0]],
            planner=PlannerConfig(deception_weight=0.3),
            observer=ObserverConfig(
                checkpoint_path="models/observer.eqx",
                num_goals=3
            )
        ),
        interceptor_agent=InterceptorAgentConfig(
            initial_position=[1.0, 9.0],
            candidate_goals=[[9.0, 9.0], [9.0, 1.0], [1.0, 9.0]],
            irl=IRLConfig(checkpoint_path="models/irl.eqx"),
            particle_filter=ParticleFilterConfig(),
            mpc=MPCConfig()
        ),
        simulation=SimulationParameters(max_time=10.0)
    )


def config_with_obstacles() -> SimulationConfig:
    """Configuration with circle and polygon obstacles."""
    return SimulationConfig(
        workspace=WorkspaceConfig(
            bounds=[[0.0, 10.0], [0.0, 10.0]],
            obstacles=[
                ObstacleConfig(
                    type="circle",
                    params={"center": [5.0, 5.0], "radius": 1.5}
                ),
                ObstacleConfig(
                    type="polygon",
                    params={"vertices": [[2.0, 2.0], [3.0, 2.0], [2.5, 3.5]]}
                )
            ]
        ),
        deceptive_agent=DeceptiveAgentConfig(
            initial_position=[0.5, 0.5],
            true_goal=[9.5, 9.5],
            candidate_goals=[[9.5, 9.5], [9.5, 0.5], [0.5, 9.5]],
            planner=PlannerConfig(deception_weight=0.5),
            observer=ObserverConfig(
                checkpoint_path="models/observer.eqx",
                num_goals=3
            )
        ),
        interceptor_agent=InterceptorAgentConfig(
            initial_position=[0.5, 9.5],
            candidate_goals=[[9.5, 9.5], [9.5, 0.5], [0.5, 9.5]],
            irl=IRLConfig(checkpoint_path="models/irl.eqx"),
            particle_filter=ParticleFilterConfig(num_particles=500),
            mpc=MPCConfig(horizon=15)
        ),
        simulation=SimulationParameters(max_time=20.0)
    )


def config_agent_d_wins() -> SimulationConfig:
    """Configuration where Agent D easily reaches goal."""
    return SimulationConfig(
        workspace=WorkspaceConfig(
            bounds=[[0.0, 10.0], [0.0, 10.0]],
            obstacles=[]
        ),
        deceptive_agent=DeceptiveAgentConfig(
            initial_position=[8.0, 8.0],  # Very close to goal
            true_goal=[9.0, 9.0],
            candidate_goals=[[9.0, 9.0], [1.0, 1.0], [1.0, 9.0]],
            planner=PlannerConfig(deception_weight=0.0),  # No deception
            observer=ObserverConfig(
                checkpoint_path="models/observer.eqx",
                num_goals=3
            )
        ),
        interceptor_agent=InterceptorAgentConfig(
            initial_position=[1.0, 1.0],  # Far away
            candidate_goals=[[9.0, 9.0], [1.0, 1.0], [1.0, 9.0]],
            irl=IRLConfig(checkpoint_path="models/irl.eqx"),
            particle_filter=ParticleFilterConfig(),
            mpc=MPCConfig()
        ),
        simulation=SimulationParameters(
            max_time=5.0,
            intercept_threshold=0.3
        )
    )


def config_agent_i_wins() -> SimulationConfig:
    """Configuration where Agent I easily intercepts."""
    return SimulationConfig(
        workspace=WorkspaceConfig(
            bounds=[[0.0, 10.0], [0.0, 10.0]],
            obstacles=[]
        ),
        deceptive_agent=DeceptiveAgentConfig(
            initial_position=[5.0, 5.0],
            true_goal=[9.0, 9.0],
            candidate_goals=[[9.0, 9.0], [9.0, 1.0], [1.0, 9.0]],
            planner=PlannerConfig(
                deception_weight=0.0,
                step_size=0.1  # Very slow
            ),
            observer=ObserverConfig(
                checkpoint_path="models/observer.eqx",
                num_goals=3
            )
        ),
        interceptor_agent=InterceptorAgentConfig(
            initial_position=[5.5, 5.5],  # Very close to Agent D
            candidate_goals=[[9.0, 9.0], [9.0, 1.0], [1.0, 9.0]],
            irl=IRLConfig(checkpoint_path="models/irl.eqx"),
            particle_filter=ParticleFilterConfig(),
            mpc=MPCConfig()
        ),
        simulation=SimulationParameters(
            max_time=10.0,
            intercept_threshold=1.0  # Easier to intercept
        )
    )


def config_timeout() -> SimulationConfig:
    """Configuration designed to timeout."""
    return SimulationConfig(
        workspace=WorkspaceConfig(
            bounds=[[0.0, 100.0], [0.0, 100.0]],  # Very large space
            obstacles=[]
        ),
        deceptive_agent=DeceptiveAgentConfig(
            initial_position=[1.0, 1.0],
            true_goal=[99.0, 99.0],  # Very far away
            candidate_goals=[[99.0, 99.0], [99.0, 1.0], [1.0, 99.0]],
            planner=PlannerConfig(deception_weight=0.5),
            observer=ObserverConfig(
                checkpoint_path="models/observer.eqx",
                num_goals=3
            )
        ),
        interceptor_agent=InterceptorAgentConfig(
            initial_position=[1.0, 99.0],
            candidate_goals=[[99.0, 99.0], [99.0, 1.0], [1.0, 99.0]],
            irl=IRLConfig(checkpoint_path="models/irl.eqx"),
            particle_filter=ParticleFilterConfig(),
            mpc=MPCConfig()
        ),
        simulation=SimulationParameters(
            max_time=1.0,  # Very short timeout
            intercept_threshold=0.1
        )
    )
