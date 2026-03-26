# Neural Network Training Procedures

## Purpose

Detailed training procedures for RNN observer and IRL models.

## RNN Observer Training

### Dataset Generation

```python
def generate_observer_training_data(
    workspace: Workspace,
    candidate_goals: List[jnp.ndarray],
    num_samples_per_goal: int,
    key: PRNGKey
) -> TrajectoryDataset:
    """Generate synthetic trajectories for observer training."""

    trajectories = []
    goal_ids = []

    for goal_id, goal in enumerate(candidate_goals):
        for _ in range(num_samples_per_goal):
            # Sample random start position
            key, start_key = jax.random.split(key)
            start = sample_free_position(workspace, start_key)

            # Plan optimal path using RRT*
            key, plan_key = jax.random.split(key)
            traj = basic_rrt_star(start, goal, workspace, plan_key)

            trajectories.append(traj.positions)
            goal_ids.append(goal_id)

    return TrajectoryDataset(
        trajectories=trajectories,
        goals=jnp.array([candidate_goals[gid] for gid in goal_ids]),
        goal_ids=jnp.array(goal_ids)
    )
```

### Training Loop

See `05-deceptive-agent.md` for implementation details.

**Hyperparameters**:

- Hidden dim: 64
- Learning rate: 1e-3
- Batch size: 32
- Epochs: 100
- Optimizer: Adam

## IRL Training

See `06-interceptor-agent.md` for maximum entropy IRL implementation.

## Navigation

**Previous**: [`07-simulation-controller.md`](./07-simulation-controller.md)

**Next**: [`09-testing-strategy.md`](./09-testing-strategy.md)
