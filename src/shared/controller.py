"""Agent controller interface and implementations for game simulation.

This module provides:
- AgentController ABC: Interface for all agent controllers
- SimpleGoalController: Moves directly toward a goal at max speed
- WaypointFollower: Follows a pre-planned trajectory
- ManualController: For interactive/keyboard control
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import jax.numpy as jnp

from src.shared.workspace import Workspace
from src.shared.trajectory import Trajectory, interpolate_velocity


@dataclass
class AgentState:
    """Current state of an agent in the game.

    Attributes:
        position: (2,) array [x, y]
        velocity: (2,) array [vx, vy]
        time: Current simulation time
    """

    position: jnp.ndarray  # (2,)
    velocity: jnp.ndarray  # (2,)
    time: float


@dataclass
class ControlCommand:
    """Control command for agent motion.

    Attributes:
        velocity: Target velocity (2,) array [vx, vy] (optional)
        acceleration: Target acceleration (2,) array [ax, ay] (optional)
    """

    velocity: Optional[jnp.ndarray] = None  # (2,)
    acceleration: Optional[jnp.ndarray] = None  # (2,)


class AgentController(ABC):
    """Abstract base class for agent controllers.

    All agent controllers must implement:
    - reset(): Initialize controller state
    - compute_control(): Generate control command from current state
    - get_name(): Return controller name for logging
    """

    @abstractmethod
    def reset(self, initial_state: AgentState, workspace: Workspace):
        """Reset controller with initial state and workspace.

        Args:
            initial_state: Starting state of the agent
            workspace: Workspace the agent operates in
        """
        pass

    @abstractmethod
    def compute_control(
        self, current_state: AgentState, observation: Optional[Dict[str, Any]] = None
    ) -> ControlCommand:
        """Compute control command for current state.

        Args:
            current_state: Current agent state
            observation: Optional observation data (e.g., opponent position)

        Returns:
            ControlCommand with velocity or acceleration
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return controller name for logging."""
        pass


class SimpleGoalController(AgentController):
    """Simple controller that moves directly toward a goal at max speed.

    Attributes:
        goal: (2,) array target position
        max_speed: Maximum speed (scalar)
        goal_reached_threshold: Distance threshold for goal reached
    """

    def __init__(
        self,
        goal: jnp.ndarray,
        max_speed: float = 1.0,
        goal_reached_threshold: float = 0.1,
    ):
        """Initialize SimpleGoalController.

        Args:
            goal: (2,) array target position
            max_speed: Maximum speed (default: 1.0)
            goal_reached_threshold: Distance to consider goal reached (default: 0.1)
        """
        self.goal = goal
        self.max_speed = max_speed
        self.goal_reached_threshold = goal_reached_threshold
        self.workspace = None

    def reset(self, initial_state: AgentState, workspace: Workspace):
        """Reset controller."""
        self.workspace = workspace

    def compute_control(
        self, current_state: AgentState, observation: Optional[Dict[str, Any]] = None
    ) -> ControlCommand:
        """Move toward goal at max speed.

        Args:
            current_state: Current agent state
            observation: Ignored

        Returns:
            ControlCommand with velocity toward goal
        """
        # Compute direction to goal
        direction = self.goal - current_state.position
        distance = jnp.linalg.norm(direction)

        # If close to goal, stop
        if distance < self.goal_reached_threshold:
            return ControlCommand(velocity=jnp.zeros(2))

        # Move at max speed toward goal
        velocity = (direction / distance) * self.max_speed

        return ControlCommand(velocity=velocity)

    def get_name(self) -> str:
        """Return controller name."""
        return "SimpleGoalController"


class WaypointFollower(AgentController):
    """Controller that follows a pre-planned trajectory.

    Attributes:
        trajectory: Pre-planned Trajectory to follow
    """

    def __init__(self, trajectory: Trajectory):
        """Initialize WaypointFollower.

        Args:
            trajectory: Pre-planned trajectory to follow
        """
        self.trajectory = trajectory
        self.workspace = None

    def reset(self, initial_state: AgentState, workspace: Workspace):
        """Reset controller."""
        self.workspace = workspace

    def compute_control(
        self, current_state: AgentState, observation: Optional[Dict[str, Any]] = None
    ) -> ControlCommand:
        """Follow pre-planned trajectory.

        Args:
            current_state: Current agent state
            observation: Ignored

        Returns:
            ControlCommand with velocity from trajectory
        """
        # Get target velocity from trajectory at current time
        t = current_state.time

        # Clamp time to trajectory bounds
        t = float(jnp.clip(t, self.trajectory.times[0], self.trajectory.times[-1]))

        # Interpolate velocity
        target_velocity = interpolate_velocity(self.trajectory, t)

        return ControlCommand(velocity=target_velocity)

    def get_name(self) -> str:
        """Return controller name."""
        return "WaypointFollower"


class ManualController(AgentController):
    """Controller for manual/interactive control (e.g., keyboard input).

    This controller maintains a target velocity that can be set externally.

    Attributes:
        target_velocity: (2,) array current target velocity
    """

    def __init__(self):
        """Initialize ManualController with zero velocity."""
        self.target_velocity = jnp.zeros(2)
        self.workspace = None

    def reset(self, initial_state: AgentState, workspace: Workspace):
        """Reset controller."""
        self.workspace = workspace
        self.target_velocity = jnp.zeros(2)

    def set_velocity(self, velocity: jnp.ndarray):
        """Set target velocity (call from external input handler).

        Args:
            velocity: (2,) array target velocity
        """
        self.target_velocity = velocity

    def compute_control(
        self, current_state: AgentState, observation: Optional[Dict[str, Any]] = None
    ) -> ControlCommand:
        """Return current target velocity.

        Args:
            current_state: Current agent state
            observation: Ignored

        Returns:
            ControlCommand with current target velocity
        """
        return ControlCommand(velocity=self.target_velocity)

    def get_name(self) -> str:
        """Return controller name."""
        return "ManualController"
