# JAX Implementation Patterns

## Purpose

JAX-specific patterns and best practices for this project.

## JIT Compilation

### What to JIT

```python
@jax.jit
def point_in_circle(point, obstacle):
    """JIT: Pure function, small arrays."""
    ...

@jax.jit
def deception_cost(path, observer_net, goal_id):
    """JIT: Performance-critical inner loop."""
    ...
```

### What NOT to JIT

```python
def adversarial_rrt_star(...):
    """NO JIT: Contains Python loops, dynamic tree structure."""
    # Use JIT for subroutines:
    if collision_free_jit(x_new, workspace):  # ← This is JIT'd
        ...
```

## vmap for Parallelization

```python
# Batch collision checking
batch_collision = jax.vmap(
    point_in_circle,
    in_axes=(0, None)  # vmap over points, broadcast obstacle
)

points = jnp.array([[1,1], [2,2], [3,3]])
results = batch_collision(points, obstacle)  # Parallel execution
```

## Pytree Patterns

```python
# NamedTuple → automatic pytree
class Trajectory(NamedTuple):
    times: jnp.ndarray
    positions: jnp.ndarray
    velocities: jnp.ndarray

# Tree map operations
traj_scaled = jax.tree_map(lambda x: 2*x, traj)
```

## PRNG Key Management

```python
def sample_positions(workspace, n, key):
    """Proper key splitting."""
    positions = []
    for i in range(n):
        key, subkey = jax.random.split(key)  # Split before each use
        positions.append(sample_free_position(workspace, subkey))
    return jnp.array(positions)
```

## Gradient Computation

```python
# Value and gradient together
@eqx.filter_jit
def loss_fn(model, x, y):
    pred = model(x)
    return jnp.mean((pred - y)**2)

loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)

# Update with Equinox
model = eqx.apply_updates(model, updates)
```

## Common Pitfalls

1. **Mutating arrays**: Use `.at[].set()` instead of direct assignment
2. **Python control flow**: Use `jax.lax.cond` for JIT-compatible conditionals
3. **Global state**: Pass all state explicitly
4. **Type stability**: Ensure array shapes/dtypes don't change

## Navigation

**Previous**: [`09-testing-strategy.md`](./09-testing-strategy.md)

**Back to**: [`00-overview.md`](./00-overview.md)
