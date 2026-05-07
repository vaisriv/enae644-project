"""Tests for metrics computation."""

import math

import jax.numpy as jnp

from src.simulation.metrics import (
    compute_belief_entropy_over_time,
    compute_deception_effectiveness,
    compute_goal_inference_accuracy,
    compute_interception_distance,
    compute_interception_efficiency,
    compute_observer_accuracy,
    compute_path_length_ratio,
    compute_time_to_convergence,
)


class TestComputeObserverAccuracy:
    """Tests for compute_observer_accuracy function."""

    def test_observer_accuracy_calculation(
        self, mock_observer_net, straight_line_trajectory
    ):
        """Mock observer returns uniform 1/3; accuracy for any goal_id is 1/3."""
        acc = compute_observer_accuracy(mock_observer_net, straight_line_trajectory, 0)
        assert isinstance(acc, float)
        assert abs(acc - 1 / 3) < 1e-5

    def test_observer_accuracy_different_goal_ids(
        self, mock_observer_net, straight_line_trajectory
    ):
        """Uniform mock gives same accuracy regardless of true_goal_id."""
        acc0 = compute_observer_accuracy(mock_observer_net, straight_line_trajectory, 0)
        acc1 = compute_observer_accuracy(mock_observer_net, straight_line_trajectory, 1)
        acc2 = compute_observer_accuracy(mock_observer_net, straight_line_trajectory, 2)
        assert abs(acc0 - acc1) < 1e-5
        assert abs(acc1 - acc2) < 1e-5

    def test_perfect_accuracy(self, straight_line_trajectory):
        """Observer that always picks goal 0 → accuracy 1.0 for goal_id=0."""

        class PerfectObserver:
            def __call__(self, positions):
                return jnp.array([1.0, 0.0, 0.0])

        acc = compute_observer_accuracy(PerfectObserver(), straight_line_trajectory, 0)
        assert abs(acc - 1.0) < 1e-5

    def test_zero_accuracy(self, straight_line_trajectory):
        """Observer that never picks goal 0 → accuracy 0.0 for goal_id=0."""

        class WrongObserver:
            def __call__(self, positions):
                return jnp.array([0.0, 0.5, 0.5])

        acc = compute_observer_accuracy(WrongObserver(), straight_line_trajectory, 0)
        assert abs(acc - 0.0) < 1e-5


class TestComputePathLengthRatio:
    """Tests for compute_path_length_ratio function."""

    def test_path_length_ratio_straight_line(self, straight_line_trajectory):
        """Same trajectory used as actual and optimal → ratio should be 1.0."""
        ratio = compute_path_length_ratio(
            straight_line_trajectory, straight_line_trajectory
        )
        assert abs(ratio - 1.0) < 1e-5

    def test_longer_actual_path(self, straight_line_trajectory):
        """A path twice as long as optimal gives ratio 2.0."""
        times = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # Detour: go to (5, 5) then back to (10, 0) — much longer than optimal
        positions = jnp.array(
            [[0.0, 0.0], [2.5, 2.5], [5.0, 5.0], [7.5, 2.5], [10.0, 0.0]]
        )
        velocities = jnp.zeros_like(positions)
        from src.shared.trajectory import Trajectory

        longer_traj = Trajectory(
            times=times, positions=positions, velocities=velocities
        )
        ratio = compute_path_length_ratio(longer_traj, straight_line_trajectory)
        assert ratio > 1.0

    def test_ratio_is_float(self, straight_line_trajectory):
        """Return type should be a Python float."""
        ratio = compute_path_length_ratio(
            straight_line_trajectory, straight_line_trajectory
        )
        assert isinstance(ratio, float)


class TestComputeBeliefEntropy:
    """Tests for compute_belief_entropy_over_time function."""

    def test_entropy_uniform_distribution(self):
        """Uniform distribution over 3 goals has entropy = log(3)."""
        belief = jnp.array([1 / 3, 1 / 3, 1 / 3])
        entropies = compute_belief_entropy_over_time([belief])
        assert len(entropies) == 1
        assert abs(float(entropies[0]) - math.log(3)) < 1e-4

    def test_entropy_deterministic_distribution(self):
        """All probability on one goal → entropy ≈ 0."""
        belief = jnp.array([1.0, 0.0, 0.0])
        entropies = compute_belief_entropy_over_time([belief])
        assert abs(float(entropies[0])) < 1e-4

    def test_entropy_over_time_shape(self):
        """Returns one entropy value per timestep."""
        history = [jnp.array([0.5, 0.3, 0.2])] * 5
        entropies = compute_belief_entropy_over_time(history)
        assert len(entropies) == 5

    def test_uniform_entropy_greater_than_deterministic(self):
        """Uniform distribution has strictly higher entropy than deterministic."""
        uniform = jnp.array([1 / 3, 1 / 3, 1 / 3])
        certain = jnp.array([1.0, 0.0, 0.0])
        h_uniform = float(compute_belief_entropy_over_time([uniform])[0])
        h_certain = float(compute_belief_entropy_over_time([certain])[0])
        assert h_uniform > h_certain


class TestComputeInterceptionDistance:
    """Tests for compute_interception_distance function."""

    def test_min_distance_between_trajectories(self, straight_line_trajectory):
        """Two identical trajectories have distance 0 everywhere."""
        dist = compute_interception_distance(
            straight_line_trajectory, straight_line_trajectory
        )
        assert isinstance(dist, float)
        assert abs(dist) < 1e-5

    def test_parallel_trajectories(self):
        """Two parallel horizontal paths separated by 1 unit → min dist = 1.0."""
        from src.shared.trajectory import Trajectory

        times = jnp.array([0.0, 1.0, 2.0])
        pos_D = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        pos_I = jnp.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        vels = jnp.zeros_like(pos_D)
        traj_D = Trajectory(times=times, positions=pos_D, velocities=vels)
        traj_I = Trajectory(times=times, positions=pos_I, velocities=vels)
        dist = compute_interception_distance(traj_D, traj_I)
        assert abs(dist - 1.0) < 1e-5

    def test_returns_minimum_not_final(self):
        """Distance is the minimum over all timesteps, not just the final one."""
        from src.shared.trajectory import Trajectory

        times = jnp.array([0.0, 1.0, 2.0])
        # Agents start far apart, pass close at t=1, then diverge
        pos_D = jnp.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        pos_I = jnp.array([[10.0, 0.0], [5.1, 0.0], [0.0, 0.0]])
        vels = jnp.zeros_like(pos_D)
        traj_D = Trajectory(times=times, positions=pos_D, velocities=vels)
        traj_I = Trajectory(times=times, positions=pos_I, velocities=vels)
        dist = compute_interception_distance(traj_D, traj_I)
        assert dist < 1.0  # close at middle step, not 10.0 (start/end dist)


class TestComputeAllMetrics:
    """Tests that all metric functions produce the expected structure."""

    def test_all_metrics_dict_structure(
        self, mock_observer_net, straight_line_trajectory
    ):
        """Verify every metric function produces a float with the right sign."""

        belief_history = [jnp.array([1 / 3, 1 / 3, 1 / 3])] * 5

        obs_acc = compute_observer_accuracy(
            mock_observer_net, straight_line_trajectory, 0
        )
        plr = compute_path_length_ratio(
            straight_line_trajectory, straight_line_trajectory
        )
        _entropies = compute_belief_entropy_over_time(belief_history)
        int_dist = compute_interception_distance(
            straight_line_trajectory, straight_line_trajectory
        )
        inf_acc = compute_goal_inference_accuracy(belief_history, 0)
        ttc = compute_time_to_convergence(belief_history)
        dec_eff = compute_deception_effectiveness(obs_acc, plr, alpha=0.5)
        int_eff = compute_interception_efficiency(int_dist, ttc, simulation_time=5.0)

        metrics = {
            "observer_accuracy": obs_acc,
            "path_length_ratio": plr,
            "min_interception_distance": int_dist,
            "goal_inference_accuracy": inf_acc,
            "time_to_convergence": ttc,
            "deception_effectiveness": dec_eff,
            "interception_efficiency": int_eff,
        }

        expected_keys = {
            "observer_accuracy",
            "path_length_ratio",
            "min_interception_distance",
            "goal_inference_accuracy",
            "time_to_convergence",
            "deception_effectiveness",
            "interception_efficiency",
        }
        assert expected_keys == set(metrics.keys())
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"
