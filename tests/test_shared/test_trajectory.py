"""Tests for trajectory operations."""

import pytest
import jax.numpy as jnp
from src.shared import trajectory


class TestCreateTrajectory:
    """Tests for create_trajectory function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_create_with_velocities(self):
        """Test creating trajectory with explicit velocities."""
        times = jnp.array([0.0, 1.0, 2.0])
        positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        velocities = jnp.ones((3, 2))

        traj = trajectory.create_trajectory(times, positions, velocities)
        assert jnp.allclose(traj.times, times)
        assert jnp.allclose(traj.positions, positions)
        assert jnp.allclose(traj.velocities, velocities)


class TestInterpolatePosition:
    """Tests for interpolate_position function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_interpolate_at_waypoint(self, straight_line_trajectory):
        """Test interpolation at exact waypoint returns waypoint."""
        traj = straight_line_trajectory
        t = traj.times[2]
        pos = trajectory.interpolate_position(traj, t)
        assert jnp.allclose(pos, traj.positions[2])

    @pytest.mark.skip(reason="Not implemented yet")
    def test_interpolate_midpoint(self, straight_line_trajectory):
        """Test interpolation at midpoint between waypoints."""
        traj = straight_line_trajectory
        t = (traj.times[0] + traj.times[1]) / 2
        pos = trajectory.interpolate_position(traj, t)
        expected = (traj.positions[0] + traj.positions[1]) / 2
        assert jnp.allclose(pos, expected)


class TestComputePathLength:
    """Tests for compute_path_length function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_straight_line_length(self, straight_line_trajectory):
        """Test path length of straight line trajectory."""
        traj = straight_line_trajectory
        length = trajectory.compute_path_length(traj)
        # Straight line from (0,0) to (10,0) should be 10.0
        assert jnp.isclose(length, 10.0)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_stationary_length(self, stationary_trajectory):
        """Test path length of stationary trajectory is zero."""
        traj = stationary_trajectory
        length = trajectory.compute_path_length(traj)
        assert jnp.isclose(length, 0.0)


class TestGetPartialTrajectory:
    """Tests for get_partial_trajectory function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_extract_middle_segment(self, straight_line_trajectory):
        """Test extracting a middle segment of trajectory."""
        traj = straight_line_trajectory
        partial = trajectory.get_partial_trajectory(traj, 1.0, 3.0)
        assert partial.times[0] >= 1.0
        assert partial.times[-1] <= 3.0


class TestGetDuration:
    """Tests for get_duration function."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_duration_calculation(self, straight_line_trajectory):
        """Test trajectory duration calculation."""
        traj = straight_line_trajectory
        duration = trajectory.get_duration(traj)
        expected = traj.times[-1] - traj.times[0]
        assert jnp.isclose(duration, expected)
