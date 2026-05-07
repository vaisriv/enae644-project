"""Shared components used by both agents."""

from src.shared.workspace import (
    Workspace,
    CircleObstacle,
    PolygonObstacle,
    create_workspace,
    is_in_workspace,
    is_in_bounds,
    is_collision_free,
    sample_collision_free_point,
)
from src.shared.trajectory import (
    Trajectory,
    create_trajectory,
    interpolate_position,
    interpolate_velocity,
    compute_path_length,
    get_partial_trajectory,
    get_duration,
    get_start_position,
    get_end_position,
    concatenate_trajectories,
)
from src.shared.collision import (
    point_in_circle,
    point_in_polygon,
    segment_circle_collision,
    segment_polygon_collision,
    batch_collision_check,
    path_collision_free,
)
from src.shared.controller import (
    AgentState,
    ControlCommand,
    AgentController,
    SimpleGoalController,
    WaypointFollower,
    ManualController,
)
from src.shared import geometry

# TODO: kinematics.py not yet created
# from src.shared.kinematics import KinematicConstraints, enforce_velocity_limit, integrate_motion

__all__ = [
    # Workspace
    "Workspace",
    "CircleObstacle",
    "PolygonObstacle",
    "create_workspace",
    "is_in_workspace",
    "is_in_bounds",
    "is_collision_free",
    "sample_collision_free_point",
    # Trajectory
    "Trajectory",
    "create_trajectory",
    "interpolate_position",
    "interpolate_velocity",
    "compute_path_length",
    "get_partial_trajectory",
    "get_duration",
    "get_start_position",
    "get_end_position",
    "concatenate_trajectories",
    # Collision
    "point_in_circle",
    "point_in_polygon",
    "segment_circle_collision",
    "segment_polygon_collision",
    "batch_collision_check",
    "path_collision_free",
    # Agent controller interface
    "AgentState",
    "ControlCommand",
    "AgentController",
    "SimpleGoalController",
    "WaypointFollower",
    "ManualController",
    # Geometry utilities (access as shared.geometry.euclidean_distance etc.)
    "geometry",
]
