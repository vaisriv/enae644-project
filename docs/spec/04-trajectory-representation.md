# Trajectory Representation (`src/shared/trajectory.py`)

## Purpose

Defines trajectory data structures and operations for representing agent motion in continuous time. Provides interpolation, resampling, and cost computation utilities.

## Dependencies

- JAX/NumPy for array operations
- SciPy for interpolation (fallback to JAX-compatible methods)

## Data Structures

### Trajectory

```python
from typing import NamedTuple
import jax.numpy as jnp

class Trajectory(NamedTuple):
    """Continuous trajectory representation."""
    times: jnp.ndarray       # (T,) timestamps
    positions: jnp.ndarray   # (T, 2) [x, y] positions
    velocities: jnp.ndarray  # (T, 2) [vx, vy] velocities

    @property
    def duration(self) -> float:
        return self.times[-1] - self.times[0]

    @property
    def length(self) -> int:
        return len(self.times)
```

## Public API

### Creation

```python
def create_trajectory(
    positions: jnp.ndarray,
    times: Optional[jnp.ndarray] = None,
    dt: float = 0.1
) -> Trajectory:
    """
    Create trajectory from positions, computing velocities.

    Args:
        positions: (T, 2) position array
        times: (T,) timestamps (default: linearly spaced starting at 0)
        dt: time step if times not provided

    Returns:
        Trajectory object

    Implementation:
        velocities[i] = (positions[i+1] - positions[i]) / (times[i+1] - times[i])
        velocities[-1] = velocities[-2]  # Duplicate last velocity
    """
    if times is None:
        times = jnp.arange(len(positions)) * dt

    # Compute velocities using finite differences
    velocities = jnp.diff(positions, axis=0) / jnp.diff(times)[:, None]
    # Append last velocity to match positions shape
    velocities = jnp.vstack([velocities, velocities[-1:]])

    return Trajectory(times=times, positions=positions, velocities=velocities)
```

### Interpolation

```python
@jax.jit
def interpolate_trajectory(
    traj: Trajectory,
    new_times: jnp.ndarray
) -> Trajectory:
    """
    Resample trajectory at new time points using linear interpolation.

    Args:
        traj: Original trajectory
        new_times: (T_new,) desired timestamps

    Returns:
        New trajectory sampled at new_times

    Algorithm:
        Uses jnp.interp for piecewise linear interpolation.
    """
    new_positions_x = jnp.interp(new_times, traj.times, traj.positions[:, 0])
    new_positions_y = jnp.interp(new_times, traj.times, traj.positions[:, 1])
    new_positions = jnp.stack([new_positions_x, new_positions_y], axis=1)

    new_velocities_x = jnp.interp(new_times, traj.times, traj.velocities[:, 0])
    new_velocities_y = jnp.interp(new_times, traj.times, traj.velocities[:, 1])
    new_velocities = jnp.stack([new_velocities_x, new_velocities_y], axis=1)

    return Trajectory(times=new_times, positions=new_positions, velocities=new_velocities)
```

### Cost Computation

```python
@jax.jit
def compute_path_length(traj: Trajectory) -> float:
    """
    Compute total Euclidean path length.

    Returns:
        Sum of segment lengths: Σ ||pos[i+1] - pos[i]||
    """
    segments = jnp.diff(traj.positions, axis=0)
    segment_lengths = jnp.linalg.norm(segments, axis=1)
    return jnp.sum(segment_lengths)


@jax.jit
def compute_control_effort(traj: Trajectory) -> float:
    """
    Compute control effort (integrated squared velocity).

    Returns:
        ∫ ||v(t)||² dt ≈ Σ ||v[i]||² · Δt[i]
    """
    dt = jnp.diff(traj.times)
    velocity_norms_squared = jnp.sum(traj.velocities[:-1]**2, axis=1)
    return jnp.sum(velocity_norms_squared * dt)
```

### Partial Trajectories

```python
def get_partial_trajectory(traj: Trajectory, t_end: float) -> Trajectory:
    """
    Extract partial trajectory up to time t_end.

    Args:
        traj: Full trajectory
        t_end: End time (must be <= traj.times[-1])

    Returns:
        Trajectory containing only data up to t_end
    """
    mask = traj.times <= t_end
    return Trajectory(
        times=traj.times[mask],
        positions=traj.positions[mask],
        velocities=traj.velocities[mask]
    )
```

## JAX Considerations

- All trajectory operations are JIT-compilable
- Trajectory is a NamedTuple → automatically a pytree
- Can use jax.tree_map for element-wise operations

## Testing

```python
def test_create_trajectory():
    positions = jnp.array([[0, 0], [1, 1], [2, 0]])
    traj = create_trajectory(positions, dt=0.1)
    assert traj.times.shape == (3,)
    assert traj.velocities.shape == (3, 2)

def test_path_length():
    positions = jnp.array([[0, 0], [3, 0], [3, 4]])  # Right triangle
    traj = create_trajectory(positions)
    length = compute_path_length(traj)
    assert jnp.isclose(length, 7.0)  # 3 + 4 = 7
```

## Navigation

**Previous**: [`03-workspace-environment.md`](./03-workspace-environment.md)

**Next**: [`05-deceptive-agent.md`](./05-deceptive-agent.md)
