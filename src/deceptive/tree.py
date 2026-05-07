"""RRT* tree data structure."""

from dataclasses import dataclass, field
from typing import List, Optional

import jax.numpy as jnp


@dataclass
class RRTNode:
    """Node in RRT* tree."""

    position: jnp.ndarray  # (2,)
    parent_id: Optional[int]
    cost: float
    children: List[int] = field(default_factory=list)


class RRTTree:
    """RRT* search tree with Python list backing."""

    def __init__(self):
        self.nodes: List[RRTNode] = []

    def add_node(
        self,
        position: jnp.ndarray,
        parent_id: Optional[int],
        cost: float,
    ) -> int:
        """Add a node and return its ID."""
        node_id = len(self.nodes)
        node = RRTNode(position=position, parent_id=parent_id, cost=cost)
        self.nodes.append(node)
        if parent_id is not None:
            self.nodes[parent_id].children.append(node_id)
        return node_id

    def find_nearest(self, position: jnp.ndarray) -> int:
        """Return ID of the nearest node."""
        best_id = 0
        best_dist = float("inf")
        for i, node in enumerate(self.nodes):
            d = float(jnp.linalg.norm(node.position - position))
            if d < best_dist:
                best_dist = d
                best_id = i
        return best_id

    def find_near(self, position: jnp.ndarray, radius: float) -> List[int]:
        """Return IDs of all nodes within radius."""
        near_ids = []
        for i, node in enumerate(self.nodes):
            if float(jnp.linalg.norm(node.position - position)) <= radius:
                near_ids.append(i)
        return near_ids

    def extract_path(self, node_id: int) -> jnp.ndarray:
        """Return (N, 2) positions from root to node_id."""
        path = []
        current_id: Optional[int] = node_id
        while current_id is not None:
            path.append(self.nodes[current_id].position)
            current_id = self.nodes[current_id].parent_id
        positions = jnp.stack(path[::-1])  # root → node order
        return positions

    def update_node(self, node_id: int, parent_id: int, cost: float) -> None:
        """Rewire node to a new parent with updated cost."""
        old_parent = self.nodes[node_id].parent_id
        if old_parent is not None and node_id in self.nodes[old_parent].children:
            self.nodes[old_parent].children.remove(node_id)
        self.nodes[node_id].parent_id = parent_id
        self.nodes[node_id].cost = cost
        self.nodes[parent_id].children.append(node_id)
