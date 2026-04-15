"""Tests for agent controller classes."""

import jax.numpy as jnp
from src.shared.controller import (
    AgentState,
    ControlCommand,
    SimpleGoalController,
    WaypointFollower,
    ManualController,
)


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_create_agent_state(self):
        """Test creating an agent state."""
        position = jnp.array([1.0, 2.0])
        velocity = jnp.array([0.5, 0.5])
        time = 1.5

        state = AgentState(position=position, velocity=velocity, time=time)

        assert jnp.allclose(state.position, position)
        assert jnp.allclose(state.velocity, velocity)
        assert state.time == time


class TestControlCommand:
    """Tests for ControlCommand dataclass."""

    def test_create_control_command_with_velocity(self):
        """Test creating control command with velocity."""
        velocity = jnp.array([1.0, 1.0])
        cmd = ControlCommand(velocity=velocity)

        assert jnp.allclose(cmd.velocity, velocity)
        assert cmd.acceleration is None

    def test_create_control_command_with_acceleration(self):
        """Test creating control command with acceleration."""
        acceleration = jnp.array([0.5, 0.5])
        cmd = ControlCommand(acceleration=acceleration)

        assert jnp.allclose(cmd.acceleration, acceleration)
        assert cmd.velocity is None


class TestSimpleGoalController:
    """Tests for SimpleGoalController class."""

    def test_initialization(self):
        """Test SimpleGoalController initialization."""
        goal = jnp.array([5.0, 5.0])
        controller = SimpleGoalController(goal=goal, max_speed=1.0)

        assert jnp.allclose(controller.goal, goal)
        assert controller.max_speed == 1.0

    def test_reset(self, simple_workspace):
        """Test resetting the controller."""
        goal = jnp.array([5.0, 5.0])
        controller = SimpleGoalController(goal=goal)

        initial_state = AgentState(
            position=jnp.array([0.0, 0.0]), velocity=jnp.zeros(2), time=0.0
        )
        controller.reset(initial_state, simple_workspace)

        assert controller.workspace is not None

    def test_move_toward_goal(self, simple_workspace):
        """Test controller moves toward goal."""
        goal = jnp.array([5.0, 0.0])
        controller = SimpleGoalController(goal=goal, max_speed=1.0)

        initial_state = AgentState(
            position=jnp.array([0.0, 0.0]), velocity=jnp.zeros(2), time=0.0
        )
        controller.reset(initial_state, simple_workspace)

        current_state = AgentState(
            position=jnp.array([2.0, 0.0]), velocity=jnp.zeros(2), time=1.0
        )
        cmd = controller.compute_control(current_state)

        # Should have velocity pointing toward goal
        assert cmd.velocity is not None
        # Velocity should point in positive x direction
        assert cmd.velocity[0] > 0
        # Velocity magnitude should be max_speed
        assert jnp.isclose(jnp.linalg.norm(cmd.velocity), 1.0)

    def test_stop_at_goal(self, simple_workspace):
        """Test controller stops when at goal."""
        goal = jnp.array([5.0, 5.0])
        controller = SimpleGoalController(goal=goal, goal_reached_threshold=0.5)

        initial_state = AgentState(
            position=jnp.array([0.0, 0.0]), velocity=jnp.zeros(2), time=0.0
        )
        controller.reset(initial_state, simple_workspace)

        # State very close to goal
        current_state = AgentState(
            position=jnp.array([5.0, 5.1]), velocity=jnp.zeros(2), time=1.0
        )
        cmd = controller.compute_control(current_state)

        # Should have zero velocity
        assert cmd.velocity is not None
        assert jnp.allclose(cmd.velocity, jnp.zeros(2))

    def test_get_name(self):
        """Test getting controller name."""
        goal = jnp.array([5.0, 5.0])
        controller = SimpleGoalController(goal=goal)

        assert controller.get_name() == "SimpleGoalController"


class TestWaypointFollower:
    """Tests for WaypointFollower class."""

    def test_initialization(self, straight_line_trajectory):
        """Test WaypointFollower initialization."""
        controller = WaypointFollower(trajectory=straight_line_trajectory)

        assert controller.trajectory is not None

    def test_reset(self, straight_line_trajectory, simple_workspace):
        """Test resetting the controller."""
        controller = WaypointFollower(trajectory=straight_line_trajectory)

        initial_state = AgentState(
            position=jnp.array([0.0, 0.0]), velocity=jnp.zeros(2), time=0.0
        )
        controller.reset(initial_state, simple_workspace)

        assert controller.workspace is not None

    def test_follow_trajectory(self, straight_line_trajectory, simple_workspace):
        """Test controller follows trajectory."""
        controller = WaypointFollower(trajectory=straight_line_trajectory)

        initial_state = AgentState(
            position=straight_line_trajectory.positions[0],
            velocity=jnp.zeros(2),
            time=0.0,
        )
        controller.reset(initial_state, simple_workspace)

        # Get control at t=1.0
        current_state = AgentState(
            position=jnp.array([2.0, 0.0]), velocity=jnp.zeros(2), time=1.0
        )
        cmd = controller.compute_control(current_state)

        # Should return velocity from trajectory
        assert cmd.velocity is not None
        # For straight line trajectory, velocity should be constant
        assert jnp.allclose(cmd.velocity, straight_line_trajectory.velocities[0])

    def test_clamp_time_bounds(self, straight_line_trajectory, simple_workspace):
        """Test controller clamps time to trajectory bounds."""
        controller = WaypointFollower(trajectory=straight_line_trajectory)

        initial_state = AgentState(
            position=straight_line_trajectory.positions[0],
            velocity=jnp.zeros(2),
            time=0.0,
        )
        controller.reset(initial_state, simple_workspace)

        # Test with time beyond trajectory end
        current_state = AgentState(
            position=jnp.array([10.0, 0.0]), velocity=jnp.zeros(2), time=100.0
        )
        cmd = controller.compute_control(current_state)

        # Should still return valid velocity
        assert cmd.velocity is not None

    def test_get_name(self, straight_line_trajectory):
        """Test getting controller name."""
        controller = WaypointFollower(trajectory=straight_line_trajectory)

        assert controller.get_name() == "WaypointFollower"


class TestManualController:
    """Tests for ManualController class."""

    def test_initialization(self):
        """Test ManualController initialization."""
        controller = ManualController()

        assert jnp.allclose(controller.target_velocity, jnp.zeros(2))

    def test_reset(self, simple_workspace):
        """Test resetting the controller."""
        controller = ManualController()

        initial_state = AgentState(
            position=jnp.array([0.0, 0.0]), velocity=jnp.zeros(2), time=0.0
        )
        controller.reset(initial_state, simple_workspace)

        assert controller.workspace is not None
        assert jnp.allclose(controller.target_velocity, jnp.zeros(2))

    def test_set_velocity(self):
        """Test setting target velocity."""
        controller = ManualController()

        new_velocity = jnp.array([1.0, 1.0])
        controller.set_velocity(new_velocity)

        assert jnp.allclose(controller.target_velocity, new_velocity)

    def test_compute_control(self, simple_workspace):
        """Test computing control returns target velocity."""
        controller = ManualController()

        initial_state = AgentState(
            position=jnp.array([0.0, 0.0]), velocity=jnp.zeros(2), time=0.0
        )
        controller.reset(initial_state, simple_workspace)

        # Set a target velocity
        target_vel = jnp.array([0.5, 0.5])
        controller.set_velocity(target_vel)

        # Compute control
        current_state = AgentState(
            position=jnp.array([1.0, 1.0]), velocity=jnp.zeros(2), time=1.0
        )
        cmd = controller.compute_control(current_state)

        assert cmd.velocity is not None
        assert jnp.allclose(cmd.velocity, target_vel)

    def test_get_name(self):
        """Test getting controller name."""
        controller = ManualController()

        assert controller.get_name() == "ManualController"

    def test_velocity_persists(self, simple_workspace):
        """Test that velocity persists across multiple control calls."""
        controller = ManualController()

        initial_state = AgentState(
            position=jnp.array([0.0, 0.0]), velocity=jnp.zeros(2), time=0.0
        )
        controller.reset(initial_state, simple_workspace)

        # Set velocity
        target_vel = jnp.array([1.0, 0.0])
        controller.set_velocity(target_vel)

        # Call compute_control multiple times
        state1 = AgentState(
            position=jnp.array([1.0, 0.0]), velocity=jnp.zeros(2), time=1.0
        )
        cmd1 = controller.compute_control(state1)

        state2 = AgentState(
            position=jnp.array([2.0, 0.0]), velocity=jnp.zeros(2), time=2.0
        )
        cmd2 = controller.compute_control(state2)

        # Both should have same velocity
        assert jnp.allclose(cmd1.velocity, target_vel)
        assert jnp.allclose(cmd2.velocity, target_vel)
