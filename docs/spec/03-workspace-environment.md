# Workspace & Environment (`src/shared/workspace.py`)

## Purpose

Implements the 2D continuous workspace representation, obstacle modeling, and spatial queries. This is the foundation for all planning and collision checking operations.

## Dependencies

- JAX/NumPy for array operations
- `src/shared/geometry` for geometric utilities

## Data Structures

### Obstacle Types

```python
from typing import NamedTuple
import jax.numpy as jnp

class CircleObstacle(NamedTuple):
    """Circular obstacle."""
    center: jnp.ndarray  # (2,) [x, y]
    radius: float

class PolygonObstacle(NamedTuple):
    """Polygonal obstacle (convex or concave)."""
    vertices: jnp.ndarray  # (n, 2) vertices in counter-clockwise order

# For signed distance field representation (extension point)
class SDFObstacle(NamedTuple):
    """Obstacle defined by signed distance function."""
    sdf_fn: Callable[[jnp.ndarray], float]  # Maps position → distance
```

### Workspace

```python
from dataclasses import dataclass
from typing import List, Union

@dataclass
class Workspace:
    """2D workspace with obstacles."""
    bounds: jnp.ndarray  # (2, 2) [[x_min, x_max], [y_min, y_max]]
    obstacles: List[Union[CircleObstacle, PolygonObstacle]]

    @property
    def x_bounds(self) -> Tuple[float, float]:
        return (self.bounds[0, 0], self.bounds[0, 1])

    @property
    def y_bounds(self) -> Tuple[float, float]:
        return (self.bounds[1, 0], self.bounds[1, 1])

    @property
    def area(self) -> float:
        return (self.bounds[0, 1] - self.bounds[0, 0]) * \
               (self.bounds[1, 1] - self.bounds[1, 0])
```

## Public API

### Workspace Creation

```python
def create_workspace(config: Dict) -> Workspace:
    """
    Create workspace from configuration dictionary.

    Args:
        config: Dictionary with 'bounds' and 'obstacles' keys

    Returns:
        Workspace object

    Example:
        config = {
            'bounds': [[0, 10], [0, 10]],
            'obstacles': [
                {'type': 'circle', 'center': [5, 5], 'radius': 1.0},
                {'type': 'polygon', 'vertices': [[2, 2], [3, 2], [3, 3]]},
            ]
        }
        workspace = create_workspace(config)
    """
    bounds = jnp.array(config['bounds'])
    obstacles = []

    for obs_config in config['obstacles']:
        if obs_config['type'] == 'circle':
            obstacles.append(CircleObstacle(
                center=jnp.array(obs_config['center']),
                radius=obs_config['radius']
            ))
        elif obs_config['type'] == 'polygon':
            obstacles.append(PolygonObstacle(
                vertices=jnp.array(obs_config['vertices'])
            ))

    return Workspace(bounds=bounds, obstacles=obstacles)
```

### Spatial Queries

```python
@jax.jit
def is_in_bounds(position: jnp.ndarray, workspace: Workspace) -> bool:
    """
    Check if position is within workspace bounds.

    Args:
        position: (2,) [x, y]
        workspace: Workspace object

    Returns:
        True if position is within bounds
    """
    return jnp.all(position >= workspace.bounds[:, 0]) and \
           jnp.all(position <= workspace.bounds[:, 1])


@jax.jit
def is_collision_free(position: jnp.ndarray, workspace: Workspace) -> bool:
    """
    Check if position is collision-free (not in any obstacle).

    Args:
        position: (2,) [x, y]
        workspace: Workspace object

    Returns:
        True if position does not collide with any obstacle
    """
    if not is_in_bounds(position, workspace):
        return False

    for obstacle in workspace.obstacles:
        if isinstance(obstacle, CircleObstacle):
            if point_in_circle(position, obstacle):
                return False
        elif isinstance(obstacle, PolygonObstacle):
            if point_in_polygon(position, obstacle):
                return False

    return True


def sample_free_position(workspace: Workspace, key: PRNGKey) -> jnp.ndarray:
    """
    Sample a random collision-free position in workspace.

    Args:
        workspace: Workspace object
        key: JAX random key

    Returns:
        (2,) position sampled uniformly from free space

    Implementation:
        Uses rejection sampling. Samples from bounds until collision-free.
    """
    max_attempts = 1000

    for i in range(max_attempts):
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, shape=(2,),
                               minval=workspace.bounds[:, 0],
                               maxval=workspace.bounds[:, 1])
        if is_collision_free(x, workspace):
            return x

    raise RuntimeError(f"Failed to sample free position after {max_attempts} attempts")
```

### Distance Computations

```python
@jax.jit
def distance_to_nearest_obstacle(
    position: jnp.ndarray,
    workspace: Workspace
) -> float:
    """
    Compute distance from position to nearest obstacle.

    Args:
        position: (2,) [x, y]
        workspace: Workspace object

    Returns:
        Minimum distance to any obstacle (positive = outside, negative = inside)

    Algorithm:
        - For each circle: distance = ||pos - center|| - radius
        - For each polygon: compute distance to each edge, take minimum
    """
    min_dist = jnp.inf

    for obstacle in workspace.obstacles:
        if isinstance(obstacle, CircleObstacle):
            dist = jnp.linalg.norm(position - obstacle.center) - obstacle.radius
        elif isinstance(obstacle, PolygonObstacle):
            dist = distance_to_polygon(position, obstacle)  # See implementation below

        min_dist = jnp.minimum(min_dist, dist)

    return min_dist


@jax.jit
def distance_to_polygon(
    point: jnp.ndarray,
    polygon: PolygonObstacle
) -> float:
    """
    Compute signed distance from point to polygon.

    Args:
        point: (2,)
        polygon: PolygonObstacle

    Returns:
        Signed distance (negative inside, positive outside)

    Algorithm:
        1. Compute distance to each edge
        2. Take minimum
        3. If point is inside polygon, negate distance
    """
    vertices = polygon.vertices
    n = vertices.shape[0]

    # Compute distance to each edge
    min_edge_dist = jnp.inf
    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        edge_dist = point_to_segment_distance(point, v1, v2)
        min_edge_dist = jnp.minimum(min_edge_dist, edge_dist)

    # Check if inside using ray casting (odd # of intersections → inside)
    inside = point_in_polygon(point, polygon)

    return -min_edge_dist if inside else min_edge_dist
```

## Collision Checking Implementation

Located in `src/shared/collision.py` but documented here for context.

### Point-in-Circle Test

```python
@jax.jit
def point_in_circle(
    point: jnp.ndarray,
    obstacle: CircleObstacle
) -> bool:
    """Check if point is inside circle."""
    return jnp.linalg.norm(point - obstacle.center) <= obstacle.radius
```

### Point-in-Polygon Test (Ray Casting Algorithm)

```python
@jax.jit
def point_in_polygon(
    point: jnp.ndarray,
    obstacle: PolygonObstacle
) -> bool:
    """
    Check if point is inside polygon using ray casting.

    Algorithm:
        Cast ray from point to +x direction.
        Count intersections with polygon edges.
        Odd count → inside, even count → outside.
    """
    vertices = obstacle.vertices
    n = vertices.shape[0]
    inside = False

    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]

        # Check if horizontal ray from point intersects edge (v1, v2)
        if ((v1[1] > point[1]) != (v2[1] > point[1])) and \
           (point[0] < (v2[0] - v1[0]) * (point[1] - v1[1]) / (v2[1] - v1[1]) + v1[0]):
            inside = not inside

    return inside
```

## Batch Operations (vmap)

```python
# Vectorized collision checking for batch of points
batch_is_collision_free = jax.vmap(
    is_collision_free,
    in_axes=(0, None)  # vmap over points, broadcast workspace
)

# Usage:
points = jnp.array([[1, 1], [2, 2], [5, 5]])  # (N, 2)
collision_free_mask = batch_is_collision_free(points, workspace)  # (N,) bool
```

## JAX Considerations

### JIT Compilation

- **JIT-compiled**: `is_in_bounds`, `is_collision_free`, `distance_to_nearest_obstacle`, point tests
- **Not JIT-compiled**: `create_workspace` (Python list comprehension), `sample_free_position` (rejection loop)

### Pytree Registration

Workspace must be registered as a pytree for use with JAX transformations:

```python
from jax.tree_util import register_pytree_node

def workspace_flatten(ws):
    # Separate dynamic (arrays) from static (Python objects) data
    children = (ws.bounds,)  # Arrays go here
    aux_data = (ws.obstacles,)  # Static data goes here
    return children, aux_data

def workspace_unflatten(aux_data, children):
    bounds, = children
    obstacles, = aux_data
    return Workspace(bounds, obstacles)

register_pytree_node(Workspace, workspace_flatten, workspace_unflatten)
```

**Limitation**: Obstacles are stored as Python list, not JAX arrays, so workspace cannot be fully JIT-compiled. This is acceptable because obstacle configuration is static.

## Edge Cases and Error Handling

### Empty Workspace

```python
# Workspace with no obstacles
empty_workspace = Workspace(
    bounds=jnp.array([[0, 10], [0, 10]]),
    obstacles=[]
)
```

### Degenerate Obstacles

- **Zero-radius circle**: Treated as point (no collision)
- **Polygon with < 3 vertices**: Raise ValueError during creation
- **Self-intersecting polygon**: Undefined behavior (avoid or validate)

### Numerical Stability

- Use epsilon tolerance for floating-point comparisons:
    ```python
    EPS = 1e-8
    is_inside = (distance < -EPS)  # Account for numerical error
    ```

## Performance Considerations

### Computational Complexity

- `is_collision_free`: O(num_obstacles) - linear scan
- `distance_to_nearest_obstacle`: O(num_obstacles × vertices_per_obstacle)
- `sample_free_position`: O(attempts × num_obstacles)

### Optimization Strategies

1. **Spatial indexing** (extension point): Use KD-tree or grid for O(log n) obstacle queries
2. **Batch processing**: Use vmap for checking multiple points simultaneously
3. **Precomputation**: Cache obstacle bounding boxes for early rejection

## Testing Strategy

### Unit Tests

```python
def test_point_in_circle():
    obs = CircleObstacle(center=jnp.array([0, 0]), radius=1.0)
    assert point_in_circle(jnp.array([0.5, 0.5]), obs) == True
    assert point_in_circle(jnp.array([2.0, 2.0]), obs) == False

def test_point_in_polygon():
    # Square polygon
    obs = PolygonObstacle(vertices=jnp.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
    assert point_in_polygon(jnp.array([0.5, 0.5]), obs) == True
    assert point_in_polygon(jnp.array([2.0, 2.0]), obs) == False

def test_workspace_bounds():
    ws = Workspace(bounds=jnp.array([[0, 10], [0, 10]]), obstacles=[])
    assert is_in_bounds(jnp.array([5, 5]), ws) == True
    assert is_in_bounds(jnp.array([11, 5]), ws) == False
```

### Integration Tests

- Validate collision checking against known scenarios
- Test sampling produces valid free positions
- Verify batch operations match sequential implementation

## Extension Points

### Future Enhancements

1. **3D workspace**: Extend to 3D by changing shape (2,) → (3,) and adding sphere/polyhedron obstacles
2. **Signed distance fields**: More efficient for complex obstacles
3. **Dynamic obstacles**: Add time-dependent obstacle positions
4. **Spatial indexing**: Accelerate queries with KD-tree or octree

## Navigation

**Previous**: [`02-data-schemas.md`](./02-data-schemas.md) - Data formats

**Next**: [`04-trajectory-representation.md`](./04-trajectory-representation.md) - Trajectory handling
