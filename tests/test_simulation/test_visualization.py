"""Tests for visualization functions."""

import jax.numpy as jnp
import csv
from src.simulation.visualization import (
    plot_workspace_with_trajectories,
    plot_belief_evolution,
    plot_distance_over_time,
    save_trajectories,
    save_metrics_csv,
    save_belief_history_csv,
)


class TestPlotWorkspaceWithTrajectories:
    """Tests for plot_workspace_with_trajectories function."""

    def test_plot_creates_figure(
        self, simple_workspace, straight_line_trajectory, test_data_dir
    ):
        """Test that plot_workspace_with_trajectories creates a matplotlib figure."""
        save_path = test_data_dir / "test_trajectories.png"

        goals = jnp.array([[5.0, 5.0], [10.0, 10.0]])

        plot_workspace_with_trajectories(
            workspace=simple_workspace,
            trajectories=[straight_line_trajectory, straight_line_trajectory],
            goals=goals,
            save_path=str(save_path),
        )

        assert save_path.exists()

    def test_plot_without_save(self, simple_workspace, straight_line_trajectory):
        """Test plotting without saving (display mode)."""
        import matplotlib

        matplotlib.use("Agg")

        goals = jnp.array([[5.0, 5.0]])

        try:
            plot_workspace_with_trajectories(
                workspace=simple_workspace,
                trajectories=[straight_line_trajectory],
                goals=goals,
                save_path=None,
            )
        except Exception as e:
            if "show" not in str(e).lower():
                raise


class TestPlotBeliefEvolution:
    """Tests for plot_belief_evolution function."""

    def test_plot_belief_evolution(self, test_data_dir):
        """Test plotting belief evolution over simulation steps."""
        save_path = test_data_dir / "test_belief.png"

        belief_history = [
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.8, 0.1, 0.1]),
            jnp.array([0.6, 0.2, 0.2]),
            jnp.array([0.4, 0.3, 0.3]),
        ]
        candidate_goals = jnp.array([[9.0, 9.0], [9.0, 1.0], [1.0, 9.0]])

        plot_belief_evolution(
            belief_history=belief_history,
            candidate_goals=candidate_goals,
            save_path=str(save_path),
        )

        assert save_path.exists()


class TestPlotDistanceOverTime:
    """Tests for plot_distance_over_time function."""

    def test_plot_distance_over_time(self, straight_line_trajectory, test_data_dir):
        """Test plotting distance between agents over time."""
        save_path = test_data_dir / "test_distance.png"

        times = jnp.array([0.0, 1.0, 2.0])
        positions_D = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        positions_I = jnp.array([[0.0, 5.0], [1.0, 4.0], [2.0, 3.0]])

        from src.shared.trajectory import create_trajectory

        traj_D = create_trajectory(times, positions_D)
        traj_I = create_trajectory(times, positions_I)

        plot_distance_over_time(
            trajectory_D=traj_D,
            trajectory_I=traj_I,
            intercept_threshold=0.5,
            save_path=str(save_path),
        )

        assert save_path.exists()


class TestSaveTrajectories:
    """Tests for save_trajectories function."""

    def test_save_csv_format(self, straight_line_trajectory, test_data_dir):
        """Test CSV output has correct format (agent, time, x, y, vx, vy)."""
        csv_path = test_data_dir / "test_trajectories.csv"

        save_trajectories(
            trajectories={
                "Agent_D": straight_line_trajectory,
                "Agent_I": straight_line_trajectory,
            },
            save_path=str(csv_path),
        )

        assert csv_path.exists()

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            assert header == ["agent", "time", "x", "y", "vx", "vy"]

            rows = list(reader)
            assert len(rows) > 0
            assert len(rows[0]) == 6

    def test_csv_agent_labels(self, test_data_dir):
        """Test that CSV rows are labelled with correct agent names."""
        from src.shared.trajectory import create_trajectory

        times = jnp.array([0.0, 1.0])
        positions = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        traj = create_trajectory(times, positions)

        csv_path = test_data_dir / "test_agent_labels.csv"
        save_trajectories(
            trajectories={"Agent_D": traj, "Agent_I": traj},
            save_path=str(csv_path),
        )

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)

        agent_names = {row[0] for row in rows}
        assert "Agent_D" in agent_names
        assert "Agent_I" in agent_names

    def test_csv_data_accuracy(self, test_data_dir):
        """Test that CSV data matches trajectory values."""
        from src.shared.trajectory import create_trajectory

        times = jnp.array([0.0, 1.0, 2.0])
        positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        velocities = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        traj = create_trajectory(times, positions, velocities)
        csv_path = test_data_dir / "test_traj_data.csv"

        save_trajectories(
            trajectories={"Agent_D": traj},
            save_path=str(csv_path),
        )

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            first_row = next(reader)

            assert first_row[0] == "Agent_D"    # agent name
            assert float(first_row[1]) == 0.0   # time
            assert float(first_row[2]) == 0.0   # x
            assert float(first_row[3]) == 0.0   # y


class TestSaveMetricsCSV:
    """Tests for save_metrics_csv function."""

    def test_save_metrics_format(self, test_data_dir):
        """Test metrics CSV output format."""
        csv_path = test_data_dir / "test_metrics.csv"

        metrics = {
            "completion_time": 10.5,
            "distance_traveled_D": 15.2,
            "distance_traveled_I": 12.8,
        }

        save_metrics_csv(metrics=metrics, output_path=str(csv_path))

        assert csv_path.exists()

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            assert header == ["metric", "value"]

            rows = list(reader)
            assert len(rows) == 3

            metric_names = [row[0] for row in rows]
            assert "completion_time" in metric_names
            assert "distance_traveled_D" in metric_names
            assert "distance_traveled_I" in metric_names

    def test_metrics_values_correct(self, test_data_dir):
        """Test that metric values are correctly saved."""
        csv_path = test_data_dir / "test_metrics_values.csv"

        metrics = {"test_metric": 42.5}

        save_metrics_csv(metrics=metrics, output_path=str(csv_path))

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)

            assert row[0] == "test_metric"
            assert float(row[1]) == 42.5


class TestSaveBeliefHistoryCSV:
    """Tests for save_belief_history_csv function."""

    def test_save_belief_history_format(self, test_data_dir):
        """Test belief history CSV output format."""
        csv_path = test_data_dir / "test_belief.csv"

        belief_history = [
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.8, 0.1, 0.1]),
            jnp.array([0.6, 0.2, 0.2]),
        ]
        times = jnp.array([0.0, 1.0, 2.0])

        save_belief_history_csv(
            belief_history=belief_history, times=times, output_path=str(csv_path)
        )

        assert csv_path.exists()

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            assert header == ["time", "goal_0_prob", "goal_1_prob", "goal_2_prob"]

            rows = list(reader)
            assert len(rows) == 3

    def test_belief_values_correct(self, test_data_dir):
        """Test that belief values are correctly saved."""
        csv_path = test_data_dir / "test_belief_values.csv"

        belief_history = [jnp.array([0.7, 0.3])]
        times = jnp.array([0.0])

        save_belief_history_csv(
            belief_history=belief_history, times=times, output_path=str(csv_path)
        )

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)

            assert float(row[0]) == 0.0
            assert abs(float(row[1]) - 0.7) < 1e-5
            assert abs(float(row[2]) - 0.3) < 1e-5
