# Testing Strategy

## Purpose

Comprehensive testing approach for all components.

## Test Categories

### Unit Tests

**Workspace (`test_shared/test_workspace.py`)**:
- Point-in-circle detection
- Point-in-polygon detection (ray casting)
- Collision-free sampling
- Distance computations

**Trajectory (`test_shared/test_trajectory.py`)**:
- Path length computation
- Interpolation accuracy
- Partial trajectory extraction

**Deceptive Agent (`test_deceptive/`)**:
- RRT* tree operations
- Deception cost evaluation
- Observer network forward pass

**Interceptor Agent (`test_interceptor/`)**:
- Particle filter update/resample
- MPC optimization convergence
- IRL gradient computation

### Integration Tests

**Full Pipelines**:
- Deceptive planning end-to-end
- Interception planning end-to-end
- Full simulation run

### Validation Tests

**Correctness**:
- RRT* finds optimal path when α=1
- Observer achieves >80% accuracy on test set
- Particle filter converges to true goal

**Performance Benchmarks**:
- RRT* planning time < 10s for 5000 iterations
- MPC solve time < 0.1s per step
- Particle filter update < 0.01s for 1000 particles

## Test Execution

```bash
# Run all tests
pytest tests/

# Run specific category
pytest tests/test_shared/

# Run with coverage
pytest --cov=src tests/
```

## Navigation

**Previous**: [`08-neural-network-training.md`](./08-neural-network-training.md)

**Next**: [`10-jax-patterns.md`](./10-jax-patterns.md)
