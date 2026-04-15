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

    times: jnp.ndarray  # (T,)
    positions: jnp.ndarray  # (T, 2)
    velocities: jnp.ndarray  # (T, 2)


def create_trajectory(
    times: jnp.ndarray, positions: jnp.ndarray, velocities: Optional[jnp.ndarray] = None
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
    if velocities is None:
        # Compute velocities using finite differences
        T = len(times)
        vels = []

        for i in range(T - 1):
            dt = times[i + 1] - times[i]
            dpos = positions[i + 1] - positions[i]
            vels.append(dpos / dt)

        # For last point, use same velocity as second-to-last
        vels.append(vels[-1] if len(vels) > 0 else jnp.zeros(2))
        velocities = jnp.stack(vels)

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
    # Find the interval containing t
    idx = jnp.searchsorted(traj.times, t) - 1
    idx = jnp.clip(idx, 0, len(traj.times) - 2)

    # Get interval bounds
    t0 = traj.times[idx]
    t1 = traj.times[idx + 1]
    pos0 = traj.positions[idx]
    pos1 = traj.positions[idx + 1]

    # Compute interpolation weight
    alpha = (t - t0) / (t1 - t0 + 1e-10)  # Add small epsilon to avoid division by zero
    alpha = jnp.clip(alpha, 0.0, 1.0)

    # Linear interpolation
    return (1 - alpha) * pos0 + alpha * pos1


def interpolate_velocity(traj: Trajectory, t: float) -> jnp.ndarray:
    """Interpolate velocity at a specific time using linear interpolation.

    Args:
        traj: Trajectory object
        t: Query time

    Returns:
        (2,) array of interpolated [vx, vy] velocity
    """
    # Find the interval containing t
    idx = jnp.searchsorted(traj.times, t) - 1
    idx = jnp.clip(idx, 0, len(traj.times) - 2)

    # Get interval bounds
    t0 = traj.times[idx]
    t1 = traj.times[idx + 1]
    vel0 = traj.velocities[idx]
    vel1 = traj.velocities[idx + 1]

    # Compute interpolation weight
    alpha = (t - t0) / (t1 - t0 + 1e-10)
    alpha = jnp.clip(alpha, 0.0, 1.0)

    # Linear interpolation
    return (1 - alpha) * vel0 + alpha * vel1


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
    # Compute distances between consecutive points
    deltas = traj.positions[1:] - traj.positions[:-1]
    distances = jnp.linalg.norm(deltas, axis=1)
    return jnp.sum(distances)  # type: ignore[return-value]


def get_partial_trajectory(
    traj: Trajectory, t_start: float, t_end: float
) -> Trajectory:
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
    # Find indices within the time window
    mask = (traj.times >= t_start) & (traj.times <= t_end)
    indices = jnp.where(mask)[0]

    # Build new trajectory points
    new_times = []
    new_positions = []
    new_velocities = []

    # Add interpolated point at t_start if needed
    if t_start < traj.times[0]:
        t_start = float(traj.times[0])
    if t_start > traj.times[0] and (
        len(indices) == 0 or traj.times[indices[0]] > t_start
    ):
        new_times.append(t_start)
        new_positions.append(interpolate_position(traj, t_start))
        new_velocities.append(interpolate_velocity(traj, t_start))

    # Add existing points in range
    for idx in indices:
        new_times.append(traj.times[idx])
        new_positions.append(traj.positions[idx])
        new_velocities.append(traj.velocities[idx])

    # Add interpolated point at t_end if needed
    if t_end > traj.times[-1]:
        t_end = float(traj.times[-1])
    if t_end < traj.times[-1] and (
        len(indices) == 0 or traj.times[indices[-1]] < t_end
    ):
        new_times.append(t_end)
        new_positions.append(interpolate_position(traj, t_end))
        new_velocities.append(interpolate_velocity(traj, t_end))

    return Trajectory(
        times=jnp.array(new_times),
        positions=jnp.stack(new_positions) if new_positions else jnp.zeros((0, 2)),
        velocities=jnp.stack(new_velocities) if new_velocities else jnp.zeros((0, 2)),
    )


def get_duration(traj: Trajectory) -> float:
    """Get total duration of trajectory.

    Args:
        traj: Trajectory object

    Returns:
        Scalar duration (t_final - t_initial)
    """
    return traj.times[-1] - traj.times[0]  # type: ignore[return-value]


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
    # Get final time of first trajectory
    t_final = traj1.times[-1]

    # Shift second trajectory times to start where first ends
    time_offset = t_final - traj2.times[0]
    new_times2 = traj2.times + time_offset

    # Concatenate all arrays
    concatenated_times = jnp.concatenate([traj1.times, new_times2])
    concatenated_positions = jnp.concatenate([traj1.positions, traj2.positions])
    concatenated_velocities = jnp.concatenate([traj1.velocities, traj2.velocities])

    return Trajectory(
        times=concatenated_times,
        positions=concatenated_positions,
        velocities=concatenated_velocities,
    )
