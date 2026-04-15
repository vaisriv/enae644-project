"""Trajectory representation and operations for continuous 2D motion.

This module provides the Trajectory pytree dataclass and related functions for:
- Representing time-parameterized trajectories
- Interpolating positions/velocities at arbitrary times
- Computing path properties (length, duration, etc.)
- Extracting partial trajectories
"""

from dataclasses import dataclass
import jax.numpy as jnp
from typing import Optional


@dataclass
class Trajectory:
    """Continuous trajectory represented as time-series of states.

    This is a JAX pytree, compatible with jit, vmap, and grad.

    Attributes:
        times: (T,) array of timestamps
        positions: (T, 2) array of [x, y] positions
        velocities: (T, 2) array of [vx, vy] velocities
    """
    times: jnp.ndarray       # (T,)
    positions: jnp.ndarray   # (T, 2)
    velocities: jnp.ndarray  # (T, 2)


def create_trajectory(
    times: jnp.ndarray,
    positions: jnp.ndarray,
    velocities: Optional[jnp.ndarray] = None
) -> Trajectory:
    """Create a Trajectory from time-series data.

    Args:
        times: (T,) array of timestamps
        positions: (T, 2) array of positions
        velocities: (T, 2) array of velocities (optional, will be computed if None)

    Returns:
        Trajectory object

    Example:
        >>> times = jnp.array([0.0, 1.0, 2.0])
        >>> positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> traj = create_trajectory(times, positions)
    """
    # TODO: Implement velocity computation if not provided
    # Use finite differences: v[i] = (pos[i+1] - pos[i]) / (t[i+1] - t[i])
    # For last point, use v[T-1] = v[T-2]
    if velocities is None:
        raise NotImplementedError("Automatic velocity computation not implemented")

    # TODO: Add validation:
    #   - times is sorted and strictly increasing
    #   - shapes are consistent (same T)
    #   - positions/velocities are (T, 2)

    return Trajectory(times=times, positions=positions, velocities=velocities)


def interpolate_position(traj: Trajectory, t: float) -> jnp.ndarray:
    """Interpolate position at a specific time using linear interpolation.

    Args:
        traj: Trajectory object
        t: Query time

    Returns:
        (2,) array of interpolated [x, y] position

    Raises:
        ValueError: If t is outside trajectory time range
    """
    # TODO: Implement linear interpolation
    # 1. Find interval: traj.times[i] <= t <= traj.times[i+1]
    # 2. Compute interpolation weight: alpha = (t - t[i]) / (t[i+1] - t[i])
    # 3. Return: (1 - alpha) * pos[i] + alpha * pos[i+1]
    # Use jnp.searchsorted() to find interval
    raise NotImplementedError("interpolate_position not implemented")


def interpolate_velocity(traj: Trajectory, t: float) -> jnp.ndarray:
    """Interpolate velocity at a specific time using linear interpolation.

    Args:
        traj: Trajectory object
        t: Query time

    Returns:
        (2,) array of interpolated [vx, vy] velocity
    """
    # TODO: Implement (same as interpolate_position but for velocities)
    raise NotImplementedError("interpolate_velocity not implemented")


def compute_path_length(traj: Trajectory) -> float:
    """Compute total path length by summing Euclidean distances.

    Args:
        traj: Trajectory object

    Returns:
        Scalar path length

    Example:
        >>> traj = create_trajectory(
        ...     times=jnp.array([0.0, 1.0, 2.0]),
        ...     positions=jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        ...     velocities=jnp.zeros((3, 2))
        ... )
        >>> compute_path_length(traj)  # Should return 2.0
    """
    # TODO: Implement path length computation
    # Sum ||pos[i+1] - pos[i]|| for i in range(T-1)
    # Use jnp.linalg.norm() or jnp.sqrt(jnp.sum((pos[i+1] - pos[i])**2))
    raise NotImplementedError("compute_path_length not implemented")


def get_partial_trajectory(traj: Trajectory, t_start: float, t_end: float) -> Trajectory:
    """Extract a partial trajectory within a time window.

    Args:
        traj: Trajectory object
        t_start: Start time (inclusive)
        t_end: End time (inclusive)

    Returns:
        New Trajectory containing only points in [t_start, t_end]

    Note:
        If t_start or t_end don't align with existing timestamps,
        interpolated points will be added at the boundaries.
    """
    # TODO: Implement partial trajectory extraction
    # 1. Find indices where t_start <= times <= t_end
    # 2. Interpolate at t_start and t_end if needed
    # 3. Return new Trajectory with filtered/interpolated points
    raise NotImplementedError("get_partial_trajectory not implemented")


def get_duration(traj: Trajectory) -> float:
    """Get total duration of trajectory.

    Args:
        traj: Trajectory object

    Returns:
        Scalar duration (t_final - t_initial)
    """
    # TODO: Implement
    # return traj.times[-1] - traj.times[0]
    raise NotImplementedError("get_duration not implemented")


def get_start_position(traj: Trajectory) -> jnp.ndarray:
    """Get starting position of trajectory.

    Args:
        traj: Trajectory object

    Returns:
        (2,) array of starting [x, y] position
    """
    return traj.positions[0]


def get_end_position(traj: Trajectory) -> jnp.ndarray:
    """Get ending position of trajectory.

    Args:
        traj: Trajectory object

    Returns:
        (2,) array of ending [x, y] position
    """
    return traj.positions[-1]


def concatenate_trajectories(traj1: Trajectory, traj2: Trajectory) -> Trajectory:
    """Concatenate two trajectories end-to-end.

    Args:
        traj1: First trajectory
        traj2: Second trajectory

    Returns:
        New Trajectory that is traj1 followed by traj2

    Note:
        The second trajectory's times will be shifted so it starts
        where the first trajectory ends.
    """
    # TODO: Implement trajectory concatenation
    # 1. Get traj1 final time: t_final = traj1.times[-1]
    # 2. Shift traj2 times: new_times2 = traj2.times - traj2.times[0] + t_final
    # 3. Concatenate arrays: jnp.concatenate([traj1.times, new_times2])
    # 4. Return new Trajectory
    raise NotImplementedError("concatenate_trajectories not implemented")
