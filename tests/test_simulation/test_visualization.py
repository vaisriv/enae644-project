"""Tests for visualization functions."""

import jax.numpy as jnp
import csv
from src.simulation.visualization import (
    plot_trajectories,
    plot_belief_evolution,
    plot_distance_over_time,
    save_trajectories_csv,
    save_metrics_csv,
    save_belief_history_csv,
)


class TestPlotTrajectories:
    """Tests for plot_trajectories function."""

    def test_plot_creates_figure(
        self, simple_workspace, straight_line_trajectory, test_data_dir
    ):
        """Test that plot_trajectories creates a matplotlib figure."""
        save_path = test_data_dir / "test_trajectories.png"

        candidate_goals = [jnp.array([5.0, 5.0]), jnp.array([10.0, 10.0])]
        true_goal = jnp.array([5.0, 5.0])

        # Should not raise an error
        plot_trajectories(
            workspace=simple_workspace,
            trajectory_D=straight_line_trajectory,
            trajectory_I=straight_line_trajectory,
            candidate_goals=candidate_goals,
            true_goal=true_goal,
            save_path=str(save_path),
        )

        # Check that file was created
        assert save_path.exists()

    def test_plot_without_save(self, simple_workspace, straight_line_trajectory):
        """Test plotting without saving (display mode)."""
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend for testing

        candidate_goals = [jnp.array([5.0, 5.0])]
        true_goal = jnp.array([5.0, 5.0])

        # Should not raise an error even without save_path
        try:
            plot_trajectories(
                workspace=simple_workspace,
                trajectory_D=straight_line_trajectory,
                trajectory_I=straight_line_trajectory,
                candidate_goals=candidate_goals,
                true_goal=true_goal,
                save_path=None,
            )
        except Exception as e:
            # It's okay if plt.show() fails in headless environment
            if "show" not in str(e).lower():
                raise


class TestPlotBeliefEvolution:
    """Tests for plot_belief_evolution function."""

    def test_plot_belief_evolution(self, test_data_dir):
        """Test plotting belief evolution over time."""
        save_path = test_data_dir / "test_belief.png"

        # Create sample belief history
        belief_history = [
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.8, 0.1, 0.1]),
            jnp.array([0.6, 0.2, 0.2]),
            jnp.array([0.4, 0.3, 0.3]),
        ]
        times = jnp.array([0.0, 1.0, 2.0, 3.0])
        true_goal_id = 0

        # Should not raise an error
        plot_belief_evolution(
            belief_history=belief_history,
            times=times,
            true_goal_id=true_goal_id,
            save_path=str(save_path),
        )

        # Check that file was created
        assert save_path.exists()


class TestPlotDistanceOverTime:
    """Tests for plot_distance_over_time function."""

    def test_plot_distance_over_time(self, straight_line_trajectory, test_data_dir):
        """Test plotting distance between agents over time."""
        save_path = test_data_dir / "test_distance.png"

        # Create two trajectories
        times = jnp.array([0.0, 1.0, 2.0])
        positions_D = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        positions_I = jnp.array([[0.0, 5.0], [1.0, 4.0], [2.0, 3.0]])

        from src.shared.trajectory import create_trajectory

        traj_D = create_trajectory(times, positions_D)
        traj_I = create_trajectory(times, positions_I)

        # Should not raise an error
        plot_distance_over_time(
            trajectory_D=traj_D,
            trajectory_I=traj_I,
            intercept_threshold=0.5,
            save_path=str(save_path),
        )

        # Check that file was created
        assert save_path.exists()


class TestSaveTrajectoriCSV:
    """Tests for save_trajectories_csv function."""

    def test_save_csv_format(self, straight_line_trajectory, test_data_dir):
        """Test CSV output has correct format."""
        csv_path = test_data_dir / "test_trajectories.csv"

        # Save trajectories
        save_trajectories_csv(
            trajectory_D=straight_line_trajectory,
            trajectory_I=straight_line_trajectory,
            output_path=str(csv_path),
        )

        # Check file exists
        assert csv_path.exists()

        # Read and verify CSV format
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Check header
            assert header == [
                "time",
                "x_D",
                "y_D",
                "vx_D",
                "vy_D",
                "x_I",
                "y_I",
                "vx_I",
                "vy_I",
            ]

            # Check that data rows exist
            rows = list(reader)
            assert len(rows) > 0

            # Check that first row has correct number of columns
            assert len(rows[0]) == 9

    def test_csv_data_accuracy(self, test_data_dir):
        """Test that CSV data matches trajectory data."""
        from src.shared.trajectory import create_trajectory

        times = jnp.array([0.0, 1.0, 2.0])
        positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        velocities = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        traj = create_trajectory(times, positions, velocities)
        csv_path = test_data_dir / "test_traj_data.csv"

        save_trajectories_csv(
            trajectory_D=traj, trajectory_I=traj, output_path=str(csv_path)
        )

        # Read CSV and check values
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            first_row = next(reader)

            # Check first timestamp
            assert float(first_row[0]) == 0.0
            # Check first position
            assert float(first_row[1]) == 0.0  # x_D
            assert float(first_row[2]) == 0.0  # y_D


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

        # Save metrics
        save_metrics_csv(metrics=metrics, output_path=str(csv_path))

        # Check file exists
        assert csv_path.exists()

        # Read and verify CSV format
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Check header
            assert header == ["metric", "value"]

            # Check that data rows exist
            rows = list(reader)
            assert len(rows) == 3

            # Check metric names
            metric_names = [row[0] for row in rows]
            assert "completion_time" in metric_names
            assert "distance_traveled_D" in metric_names
            assert "distance_traveled_I" in metric_names

    def test_metrics_values_correct(self, test_data_dir):
        """Test that metric values are correctly saved."""
        csv_path = test_data_dir / "test_metrics_values.csv"

        metrics = {"test_metric": 42.5}

        save_metrics_csv(metrics=metrics, output_path=str(csv_path))

        # Read and check value
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
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

        # Save belief history
        save_belief_history_csv(
            belief_history=belief_history, times=times, output_path=str(csv_path)
        )

        # Check file exists
        assert csv_path.exists()

        # Read and verify CSV format
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Check header
            assert header == ["time", "goal_0_prob", "goal_1_prob", "goal_2_prob"]

            # Check that data rows exist
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

        # Read and check values
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            row = next(reader)

            assert float(row[0]) == 0.0  # time
            # Use approximate comparison for floating point values
            assert abs(float(row[1]) - 0.7) < 1e-5  # goal_0_prob
            assert abs(float(row[2]) - 0.3) < 1e-5  # goal_1_prob
