"""Data loaders for pre-saved trajectory datasets."""

from src.data.schemas import TrajectoryDataset


def load_trajectory_dataset(path: str) -> TrajectoryDataset:
    """Load a trajectory dataset from a saved NumPy archive.

    Args:
        path: Path to a .npz file previously saved by save_trajectory_dataset

    Returns:
        TrajectoryDataset
    """
    raise NotImplementedError(
        "load_trajectory_dataset not implemented — "
        "datasets are generated on-the-fly by generate_optimal_trajectories."
    )
