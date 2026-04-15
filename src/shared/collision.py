"""Collision detection utilities for 2D geometric primitives.

This module provides JIT-compilable collision detection functions for:
- Point vs circle
- Point vs polygon
- Line segment vs circle
- Line segment vs polygon
- Batch collision checking (vectorized)
"""

import jax
import jax.numpy as jnp
from typing import List

from src.shared.geometry import point_to_segment_distance, cross_product_2d


def point_in_circle(point: jnp.ndarray, center: jnp.ndarray, radius: float) -> bool:
    """Check if a point is inside a circle.

    Args:
        point: (2,) array [x, y]
        center: (2,) array [cx, cy]
        radius: Circle radius

    Returns:
        True if ||point - center|| <= radius, False otherwise

    Note:
        This function is JIT-compilable.
    """
    distance = jnp.linalg.norm(point - center)
    return distance <= radius


def point_in_polygon(point: jnp.ndarray, vertices: jnp.ndarray) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm.

    Args:
        point: (2,) array [x, y]
        vertices: (N, 2) array of polygon vertices (counter-clockwise order)

    Returns:
        True if point is inside polygon, False otherwise

    Note:
        Uses the ray casting algorithm:
        Cast a ray from the point to infinity and count edge crossings.
        Odd number of crossings = inside, even = outside.

        This function is JIT-compilable.
    """
    # Ray casting: cast horizontal ray from point to the right
    # Count edge crossings; odd = inside, even = outside

    n = vertices.shape[0]
    px, py = point[0], point[1]

    def check_edge(i, count):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]

        # Check if ray crosses edge (v1, v2)
        # Edge must straddle the horizontal line at py
        crosses_y = (v1[1] > py) != (v2[1] > py)

        # Compute x-coordinate of intersection with horizontal line
        # x = v1.x + (py - v1.y) * (v2.x - v1.x) / (v2.y - v1.y)
        x_intersect = v1[0] + (py - v1[1]) * (v2[0] - v1[0]) / (v2[1] - v1[1] + 1e-10)

        # Ray goes to the right (+x), so intersection must be at x > px
        crosses_x = x_intersect > px

        # Count crossing if both conditions are met
        return count + jnp.where(crosses_y & crosses_x, 1, 0)

    crossing_count = jax.lax.fori_loop(0, n, check_edge, 0)
    return (crossing_count % 2) == 1


def segment_circle_collision(
    seg_start: jnp.ndarray, seg_end: jnp.ndarray, center: jnp.ndarray, radius: float
) -> bool:
    """Check if a line segment intersects a circle.

    Args:
        seg_start: (2,) array - segment start point
        seg_end: (2,) array - segment end point
        center: (2,) array - circle center
        radius: Circle radius

    Returns:
        True if segment intersects circle, False otherwise

    Note:
        Uses point-to-segment distance formula.
        This function is JIT-compilable.
    """
    distance = point_to_segment_distance(center, seg_start, seg_end)
    return distance <= radius


def segment_polygon_collision(
    seg_start: jnp.ndarray, seg_end: jnp.ndarray, vertices: jnp.ndarray
) -> bool:
    """Check if a line segment intersects a polygon.

    Args:
        seg_start: (2,) array - segment start point
        seg_end: (2,) array - segment end point
        vertices: (N, 2) array of polygon vertices

    Returns:
        True if segment intersects polygon boundary or interior, False otherwise

    Note:
        Checks:
        1. If either endpoint is inside polygon
        2. If segment intersects any polygon edge
        This function is JIT-compilable.
    """
    # Check if either endpoint is inside polygon
    if point_in_polygon(seg_start, vertices) or point_in_polygon(seg_end, vertices):
        return True

    # Check if segment intersects any polygon edge
    n = vertices.shape[0]

    def check_edge_intersection(i, has_collision):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        intersects = segment_segment_intersection(seg_start, seg_end, v1, v2)
        return has_collision | intersects

    return jax.lax.fori_loop(0, n, check_edge_intersection, False)


def segment_segment_intersection(
    seg1_start: jnp.ndarray,
    seg1_end: jnp.ndarray,
    seg2_start: jnp.ndarray,
    seg2_end: jnp.ndarray,
) -> bool:
    """Check if two line segments intersect.

    Args:
        seg1_start: (2,) array - first segment start
        seg1_end: (2,) array - first segment end
        seg2_start: (2,) array - second segment start
        seg2_end: (2,) array - second segment end

    Returns:
        True if segments intersect, False otherwise

    Note:
        Uses cross product method for intersection testing.
        This function is JIT-compilable.
    """
    # Use cross product method
    # Segments (p1,p2) and (p3,p4) intersect if they straddle each other

    p1, p2 = seg1_start, seg1_end
    p3, p4 = seg2_start, seg2_end

    # Check if p3 and p4 are on opposite sides of line through p1-p2
    d1 = cross_product_2d(p2 - p1, p3 - p1)
    d2 = cross_product_2d(p2 - p1, p4 - p1)

    # Check if p1 and p2 are on opposite sides of line through p3-p4
    d3 = cross_product_2d(p4 - p3, p1 - p3)
    d4 = cross_product_2d(p4 - p3, p2 - p3)

    # Segments intersect if signs are opposite (or zero for touching)
    return (d1 * d2 <= 0) & (d3 * d4 <= 0)


@jax.jit
def batch_collision_check(
    points: jnp.ndarray, circle_centers: jnp.ndarray, circle_radii: jnp.ndarray
) -> jnp.ndarray:
    """Vectorized collision check for multiple points and circles.

    Args:
        points: (N, 2) array of points
        circle_centers: (M, 2) array of circle centers
        circle_radii: (M,) array of circle radii

    Returns:
        (N,) boolean array where True = point collides with at least one circle

    Note:
        Uses vmap for vectorization. This is JIT-compiled.
    """
    # For each point, check collision with all circles
    # Use vmap to vectorize over points and circles

    def check_point_all_circles(point):
        # Check this point against all circles
        def check_one_circle(center, radius):
            return point_in_circle(point, center, radius)

        collisions = jax.vmap(check_one_circle)(circle_centers, circle_radii)
        return jnp.any(collisions)

    # Vectorize over all points
    return jax.vmap(check_point_all_circles)(points)


def path_collision_free(
    path: jnp.ndarray, circle_obstacles: List, polygon_obstacles: List
) -> bool:
    """Check if an entire path is collision-free.

    Args:
        path: (N, 2) array of waypoints
        circle_obstacles: List of (center, radius) tuples
        polygon_obstacles: List of vertices arrays

    Returns:
        True if all segments in path are collision-free, False otherwise

    Note:
        Checks each segment (path[i], path[i+1]) against all obstacles.
    """
    # Check each segment in path against all obstacles
    n_waypoints = path.shape[0]

    for i in range(n_waypoints - 1):
        seg_start = path[i]
        seg_end = path[i + 1]

        # Check against circle obstacles
        for center, radius in circle_obstacles:
            if segment_circle_collision(seg_start, seg_end, center, radius):
                return False

        # Check against polygon obstacles
        for vertices in polygon_obstacles:
            if segment_polygon_collision(seg_start, seg_end, vertices):
                return False

    return True
