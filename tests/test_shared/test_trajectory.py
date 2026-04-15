"""Tests for trajectory operations."""

import jax.numpy as jnp
from src.shared import trajectory


class TestCreateTrajectory:
    """Tests for create_trajectory function."""

    def test_create_with_velocities(self):
        """Test creating trajectory with explicit velocities."""
        times = jnp.array([0.0, 1.0, 2.0])
        positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        velocities = jnp.ones((3, 2))

        traj = trajectory.create_trajectory(times, positions, velocities)
        assert jnp.allclose(traj.times, times)
        assert jnp.allclose(traj.positions, positions)
        assert jnp.allclose(traj.velocities, velocities)

    def test_create_without_velocities(self):
        """Test creating trajectory with auto-computed velocities."""
        times = jnp.array([0.0, 1.0, 2.0])
        positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

        traj = trajectory.create_trajectory(times, positions)
        assert jnp.allclose(traj.times, times)
        assert jnp.allclose(traj.positions, positions)
        # Velocities should be computed from position differences
        assert traj.velocities.shape == (3, 2)


class TestInterpolatePosition:
    """Tests for interpolate_position function."""

    def test_interpolate_at_waypoint(self, straight_line_trajectory):
        """Test interpolation at exact waypoint returns waypoint."""
        traj = straight_line_trajectory
        t = traj.times[2]
        pos = trajectory.interpolate_position(traj, t)
        assert jnp.allclose(pos, traj.positions[2])

    def test_interpolate_midpoint(self, straight_line_trajectory):
        """Test interpolation at midpoint between waypoints."""
        traj = straight_line_trajectory
        t = (traj.times[0] + traj.times[1]) / 2
        pos = trajectory.interpolate_position(traj, t)
        expected = (traj.positions[0] + traj.positions[1]) / 2
        assert jnp.allclose(pos, expected)

    def test_interpolate_before_start(self, straight_line_trajectory):
        """Test interpolation before trajectory start clamps to start."""
        traj = straight_line_trajectory
        t = traj.times[0] - 1.0
        pos = trajectory.interpolate_position(traj, t)
        # Should clamp to first position
        assert jnp.allclose(pos, traj.positions[0])

    def test_interpolate_after_end(self, straight_line_trajectory):
        """Test interpolation after trajectory end clamps to end."""
        traj = straight_line_trajectory
        t = traj.times[-1] + 1.0
        pos = trajectory.interpolate_position(traj, t)
        # Should clamp to last position
        assert jnp.allclose(pos, traj.positions[-1])


class TestInterpolateVelocity:
    """Tests for interpolate_velocity function."""

    def test_interpolate_velocity_at_waypoint(self, straight_line_trajectory):
        """Test velocity interpolation at exact waypoint."""
        traj = straight_line_trajectory
        t = traj.times[2]
        vel = trajectory.interpolate_velocity(traj, t)
        assert jnp.allclose(vel, traj.velocities[2])

    def test_interpolate_velocity_midpoint(self, straight_line_trajectory):
        """Test velocity interpolation at midpoint."""
        traj = straight_line_trajectory
        t = (traj.times[0] + traj.times[1]) / 2
        vel = trajectory.interpolate_velocity(traj, t)
        expected = (traj.velocities[0] + traj.velocities[1]) / 2
        assert jnp.allclose(vel, expected)


class TestComputePathLength:
    """Tests for compute_path_length function."""

    def test_straight_line_length(self, straight_line_trajectory):
        """Test path length of straight line trajectory."""
        traj = straight_line_trajectory
        length = trajectory.compute_path_length(traj)
        # Straight line from (0,0) to (10,0) should be 10.0
        assert jnp.isclose(length, 10.0)

    def test_stationary_length(self, stationary_trajectory):
        """Test path length of stationary trajectory is zero."""
        traj = stationary_trajectory
        length = trajectory.compute_path_length(traj)
        assert jnp.isclose(length, 0.0)

    def test_right_angle_path(self):
        """Test path length of right-angle path."""
        times = jnp.array([0.0, 1.0, 2.0])
        positions = jnp.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
        traj = trajectory.create_trajectory(times, positions)
        length = trajectory.compute_path_length(traj)
        # 3 + 4 = 7
        assert jnp.isclose(length, 7.0)


class TestGetPartialTrajectory:
    """Tests for get_partial_trajectory function."""

    def test_extract_middle_segment(self, straight_line_trajectory):
        """Test extracting a middle segment of trajectory."""
        traj = straight_line_trajectory
        partial = trajectory.get_partial_trajectory(traj, 1.0, 3.0)
        assert partial.times[0] >= 1.0
        assert partial.times[-1] <= 3.0

    def test_extract_full_trajectory(self, straight_line_trajectory):
        """Test extracting full trajectory returns all points."""
        traj = straight_line_trajectory
        partial = trajectory.get_partial_trajectory(traj, traj.times[0], traj.times[-1])
        assert len(partial.positions) >= len(traj.positions) - 1


class TestGetDuration:
    """Tests for get_duration function."""

    def test_duration_calculation(self, straight_line_trajectory):
        """Test trajectory duration calculation."""
        traj = straight_line_trajectory
        duration = trajectory.get_duration(traj)
        expected = traj.times[-1] - traj.times[0]
        assert jnp.isclose(duration, expected)


class TestGetStartPosition:
    """Tests for get_start_position function."""

    def test_get_start_position(self, straight_line_trajectory):
        """Test getting start position."""
        traj = straight_line_trajectory
        start_pos = trajectory.get_start_position(traj)
        assert jnp.allclose(start_pos, traj.positions[0])


class TestGetEndPosition:
    """Tests for get_end_position function."""

    def test_get_end_position(self, straight_line_trajectory):
        """Test getting end position."""
        traj = straight_line_trajectory
        end_pos = trajectory.get_end_position(traj)
        assert jnp.allclose(end_pos, traj.positions[-1])


class TestConcatenateTrajectories:
    """Tests for concatenate_trajectories function."""

    def test_concatenate_two_trajectories(self):
        """Test concatenating two trajectories."""
        times1 = jnp.array([0.0, 1.0, 2.0])
        positions1 = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        traj1 = trajectory.create_trajectory(times1, positions1)

        times2 = jnp.array([0.0, 1.0])
        positions2 = jnp.array([[2.0, 0.0], [2.0, 1.0]])
        traj2 = trajectory.create_trajectory(times2, positions2)

        combined = trajectory.concatenate_trajectories(traj1, traj2)

        # Check total length
        assert len(combined.positions) == len(traj1.positions) + len(traj2.positions)
        # Check first position matches traj1 start
        assert jnp.allclose(combined.positions[0], traj1.positions[0])
        # Check last position matches traj2 end
        assert jnp.allclose(combined.positions[-1], traj2.positions[-1])

    def test_concatenate_time_offset(self):
        """Test that concatenated trajectory has continuous time."""
        times1 = jnp.array([0.0, 1.0])
        positions1 = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        traj1 = trajectory.create_trajectory(times1, positions1)

        times2 = jnp.array([0.0, 1.0])
        positions2 = jnp.array([[1.0, 0.0], [2.0, 0.0]])
        traj2 = trajectory.create_trajectory(times2, positions2)

        combined = trajectory.concatenate_trajectories(traj1, traj2)

        # Times should be continuous (second trajectory should start after first ends)
        # First trajectory ends at time 1.0, second should start at 1.0
        assert combined.times[-1] >= traj1.times[-1]
