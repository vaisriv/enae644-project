"""Integration tests for simulation controller."""

import pytest
import jax.numpy as jnp
from src.simulation.controller import (
    run_simulation,
    SimulationResult,
    _check_goal_reached,
    _check_interception,
    _create_trajectory_from_positions,
    run_game_with_controllers,
)
from src.shared.controller import SimpleGoalController, AgentState


class TestCheckGoalReached:
    """Tests for _check_goal_reached helper function."""

    def test_goal_reached(self):
        """Test that goal is detected when position is within threshold."""
        position = jnp.array([5.0, 5.0])
        goal = jnp.array([5.1, 5.0])
        threshold = 0.5

        assert _check_goal_reached(position, goal, threshold)

    def test_goal_not_reached(self):
        """Test that goal is not detected when position is beyond threshold."""
        position = jnp.array([5.0, 5.0])
        goal = jnp.array([10.0, 10.0])
        threshold = 0.5

        assert not _check_goal_reached(position, goal, threshold)

    def test_goal_exactly_at_threshold(self):
        """Test goal detection at exact threshold distance."""
        position = jnp.array([0.0, 0.0])
        goal = jnp.array([0.49, 0.0])  # Just under threshold
        threshold = 0.5

        # Should be detected since distance is less than threshold
        result = _check_goal_reached(position, goal, threshold)
        assert result


class TestCheckInterception:
    """Tests for _check_interception helper function."""

    def test_interception_detected(self):
        """Test that interception is detected when agents are close."""
        pos_D = jnp.array([5.0, 5.0])
        pos_I = jnp.array([5.2, 5.0])
        threshold = 0.5

        assert _check_interception(pos_D, pos_I, threshold)

    def test_interception_not_detected(self):
        """Test that interception is not detected when agents are far."""
        pos_D = jnp.array([5.0, 5.0])
        pos_I = jnp.array([10.0, 10.0])
        threshold = 0.5

        assert not _check_interception(pos_D, pos_I, threshold)

    def test_interception_at_same_position(self):
        """Test interception when agents are at exactly the same position."""
        pos_D = jnp.array([5.0, 5.0])
        pos_I = jnp.array([5.0, 5.0])
        threshold = 0.5

        assert _check_interception(pos_D, pos_I, threshold)


class TestCreateTrajectoryFromPositions:
    """Tests for _create_trajectory_from_positions helper function."""

    def test_create_trajectory_from_positions(self):
        """Test creating trajectory from time-series positions."""
        times = jnp.array([0.0, 1.0, 2.0, 3.0])
        positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

        traj = _create_trajectory_from_positions(times, positions)

        assert jnp.allclose(traj.times, times)
        assert jnp.allclose(traj.positions, positions)
        # Velocities should be auto-computed
        assert traj.velocities.shape == positions.shape

    def test_trajectory_velocities_computed(self):
        """Test that velocities are correctly computed."""
        times = jnp.array([0.0, 1.0, 2.0])
        positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

        traj = _create_trajectory_from_positions(times, positions)

        # For uniform motion, velocities should be constant
        # v = (1.0 - 0.0) / (1.0 - 0.0) = 1.0 in x direction
        assert traj.velocities[0, 0] == 1.0
        assert traj.velocities[0, 1] == 0.0


class TestRunGameWithControllers:
    """Tests for run_game_with_controllers function."""

    def test_agent_d_wins(self, simple_workspace):
        """Test scenario where Agent D reaches goal before interception."""
        # Setup: Agent D starts at (0,0) with goal at (5,0)
        # Agent I starts at (0,10) far away
        goal_D = jnp.array([5.0, 0.0])
        controller_D = SimpleGoalController(goal=goal_D, max_speed=2.0)
        controller_I = SimpleGoalController(
            goal=jnp.array([5.0, 0.0]), max_speed=0.5
        )  # Slower

        initial_state_D = AgentState(
            position=jnp.array([0.0, 0.0]), velocity=jnp.zeros(2), time=0.0
        )
        initial_state_I = AgentState(
            position=jnp.array([0.0, 10.0]), velocity=jnp.zeros(2), time=0.0
        )

        result = run_game_with_controllers(
            workspace=simple_workspace,
            controller_D=controller_D,
            controller_I=controller_I,
            initial_state_D=initial_state_D,
            initial_state_I=initial_state_I,
            goal_D=goal_D,
            max_time=20.0,
            dt=0.1,
            intercept_threshold=0.5,
            goal_radius=0.5,
        )

        assert isinstance(result, SimulationResult)
        assert result.winner == "Agent_D"

    def test_agent_i_wins(self, simple_workspace):
        """Test scenario where Agent I intercepts Agent D."""
        # Setup: Both agents start close, I is faster
        goal_D = jnp.array([10.0, 0.0])
        controller_D = SimpleGoalController(goal=goal_D, max_speed=0.5)
        controller_I = SimpleGoalController(
            goal=jnp.array([5.0, 0.0]), max_speed=2.0
        )  # Faster, will intercept

        initial_state_D = AgentState(
            position=jnp.array([0.0, 0.0]), velocity=jnp.zeros(2), time=0.0
        )
        initial_state_I = AgentState(
            position=jnp.array([0.0, 1.0]), velocity=jnp.zeros(2), time=0.0
        )

        result = run_game_with_controllers(
            workspace=simple_workspace,
            controller_D=controller_D,
            controller_I=controller_I,
            initial_state_D=initial_state_D,
            initial_state_I=initial_state_I,
            goal_D=goal_D,
            max_time=20.0,
            dt=0.1,
            intercept_threshold=0.5,
            goal_radius=0.5,
        )

        assert isinstance(result, SimulationResult)
        assert result.winner == "Agent_I"

    def test_timeout(self, simple_workspace):
        """Test scenario that times out."""
        # Setup: Goal far away, short timeout
        goal_D = jnp.array([100.0, 100.0])
        controller_D = SimpleGoalController(goal=goal_D, max_speed=1.0)
        controller_I = SimpleGoalController(goal=jnp.array([0.0, 0.0]), max_speed=1.0)

        initial_state_D = AgentState(
            position=jnp.array([0.0, 0.0]), velocity=jnp.zeros(2), time=0.0
        )
        initial_state_I = AgentState(
            position=jnp.array([10.0, 10.0]), velocity=jnp.zeros(2), time=0.0
        )

        result = run_game_with_controllers(
            workspace=simple_workspace,
            controller_D=controller_D,
            controller_I=controller_I,
            initial_state_D=initial_state_D,
            initial_state_I=initial_state_I,
            goal_D=goal_D,
            max_time=1.0,  # Very short timeout
            dt=0.1,
            intercept_threshold=0.5,
            goal_radius=0.5,
        )

        assert isinstance(result, SimulationResult)
        assert result.winner == "timeout"

    def test_result_structure(self, simple_workspace):
        """Test that result has all required fields."""
        goal_D = jnp.array([5.0, 0.0])
        controller_D = SimpleGoalController(goal=goal_D, max_speed=1.0)
        controller_I = SimpleGoalController(goal=jnp.array([0.0, 0.0]), max_speed=1.0)

        initial_state_D = AgentState(
            position=jnp.array([0.0, 0.0]), velocity=jnp.zeros(2), time=0.0
        )
        initial_state_I = AgentState(
            position=jnp.array([10.0, 10.0]), velocity=jnp.zeros(2), time=0.0
        )

        result = run_game_with_controllers(
            workspace=simple_workspace,
            controller_D=controller_D,
            controller_I=controller_I,
            initial_state_D=initial_state_D,
            initial_state_I=initial_state_I,
            goal_D=goal_D,
            max_time=20.0,
            dt=0.1,
        )

        # Check all required fields exist
        assert hasattr(result, "winner")
        assert hasattr(result, "completion_time")
        assert hasattr(result, "trajectory_D")
        assert hasattr(result, "trajectory_I")
        assert hasattr(result, "belief_history")
        assert hasattr(result, "metrics")

        # Check field types
        assert isinstance(result.winner, str)
        assert isinstance(result.completion_time, (float, int))
        assert isinstance(result.metrics, dict)
        assert isinstance(result.belief_history, list)

    def test_trajectories_recorded(self, simple_workspace):
        """Test that trajectories are properly recorded."""
        goal_D = jnp.array([5.0, 0.0])
        controller_D = SimpleGoalController(goal=goal_D, max_speed=1.0)
        controller_I = SimpleGoalController(goal=jnp.array([0.0, 0.0]), max_speed=1.0)

        initial_state_D = AgentState(
            position=jnp.array([0.0, 0.0]), velocity=jnp.zeros(2), time=0.0
        )
        initial_state_I = AgentState(
            position=jnp.array([10.0, 10.0]), velocity=jnp.zeros(2), time=0.0
        )

        result = run_game_with_controllers(
            workspace=simple_workspace,
            controller_D=controller_D,
            controller_I=controller_I,
            initial_state_D=initial_state_D,
            initial_state_I=initial_state_I,
            goal_D=goal_D,
            max_time=20.0,
            dt=0.1,
        )

        # Check trajectories have data
        assert len(result.trajectory_D.positions) > 0
        assert len(result.trajectory_I.positions) > 0
        assert len(result.trajectory_D.times) > 0
        assert len(result.trajectory_I.times) > 0

        # Check initial positions match
        assert jnp.allclose(result.trajectory_D.positions[0], initial_state_D.position)
        assert jnp.allclose(result.trajectory_I.positions[0], initial_state_I.position)


class TestRunSimulation:
    """Tests for run_simulation function."""

    def test_simulation_initialization(self, minimal_simulation_config, jax_key):
        """Test simulation can be initialized from config."""
        # This test will check that run_simulation can at least start
        # even if it hits NotImplementedError in stubs
        with pytest.raises(NotImplementedError):
            result = run_simulation(minimal_simulation_config, jax_key)


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""

    def test_result_structure(self, straight_line_trajectory):
        """Test SimulationResult has required fields."""
        result = SimulationResult(
            winner="Agent_D",
            completion_time=10.0,
            trajectory_D=straight_line_trajectory,
            trajectory_I=straight_line_trajectory,
            belief_history=[jnp.array([0.5, 0.5])],
            metrics={"test_metric": 1.0},
        )

        assert result.winner == "Agent_D"
        assert result.completion_time == 10.0
        assert result.trajectory_D is not None
        assert result.trajectory_I is not None
        assert len(result.belief_history) == 1
        assert "test_metric" in result.metrics
