"""Sample trajectory generators for testing."""

import jax.numpy as jnp
from src.shared.trajectory import Trajectory


def straight_line_trajectory() -> Trajectory:
    """Simple A→B trajectory along x-axis."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    positions = jnp.array(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0]]
    )
    velocities = jnp.ones((6, 2)) * jnp.array([2.0, 0.0])
    return Trajectory(times=times, positions=positions, velocities=velocities)


def curved_trajectory() -> Trajectory:
    """Multi-segment curved path forming an arc."""
    t = jnp.linspace(0, jnp.pi, 20)
    positions = jnp.stack([5.0 + 5.0 * jnp.cos(t), 5.0 + 5.0 * jnp.sin(t)], axis=1)
    velocities = jnp.stack([-5.0 * jnp.sin(t), 5.0 * jnp.cos(t)], axis=1)
    times = jnp.linspace(0.0, 10.0, 20)
    return Trajectory(times=times, positions=positions, velocities=velocities)


def deceptive_trajectory() -> Trajectory:
    """Path that initially moves away from goal then curves back.

    This mimics deceptive behavior where the agent appears to go
    toward one goal then switches direction.
    """
    # Phase 1: Move toward decoy goal (0-5 seconds)
    t1 = jnp.linspace(0.0, 5.0, 10)
    pos1 = jnp.stack([t1, jnp.zeros_like(t1)], axis=1)

    # Phase 2: Curve toward true goal (5-10 seconds)
    t2 = jnp.linspace(0.0, jnp.pi / 2, 10)
    pos2 = jnp.stack([5.0 + 5.0 * jnp.cos(t2), 5.0 * jnp.sin(t2)], axis=1)

    times = jnp.concatenate([t1, t1[-1] + jnp.linspace(0.1, 5.1, 10)])
    positions = jnp.concatenate([pos1, pos2], axis=0)

    # Compute velocities via finite differences
    dt = jnp.diff(times)
    dpos = jnp.diff(positions, axis=0)
    vels = dpos / dt[:, None]
    velocities = jnp.concatenate([vels, vels[-1:]], axis=0)

    return Trajectory(times=times, positions=positions, velocities=velocities)


def zigzag_trajectory() -> Trajectory:
    """Zigzag trajectory for testing collision detection."""
    positions = jnp.array(
        [[0.0, 0.0], [2.0, 2.0], [4.0, 0.0], [6.0, 2.0], [8.0, 0.0], [10.0, 2.0]]
    )
    times = jnp.linspace(0.0, 5.0, 6)

    # Compute velocities
    dt = jnp.diff(times)
    dpos = jnp.diff(positions, axis=0)
    vels = dpos / dt[:, None]
    velocities = jnp.concatenate([vels, vels[-1:]], axis=0)

    return Trajectory(times=times, positions=positions, velocities=velocities)


def circle_trajectory() -> Trajectory:
    """Complete circular trajectory."""
    t = jnp.linspace(0, 2 * jnp.pi, 50)
    radius = 5.0
    center = jnp.array([5.0, 5.0])

    positions = jnp.stack(
        [center[0] + radius * jnp.cos(t), center[1] + radius * jnp.sin(t)], axis=1
    )

    velocities = jnp.stack([-radius * jnp.sin(t), radius * jnp.cos(t)], axis=1)

    times = jnp.linspace(0.0, 10.0, 50)

    return Trajectory(times=times, positions=positions, velocities=velocities)


def stationary_trajectory() -> Trajectory:
    """Trajectory where agent doesn't move."""
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    positions = jnp.zeros((4, 2))
    velocities = jnp.zeros((4, 2))
    return Trajectory(times=times, positions=positions, velocities=velocities)
