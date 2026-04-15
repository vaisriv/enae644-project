"""Geometric utility functions for 2D vector operations.

This module provides JIT-compilable geometric primitives for:
- Distance computations
- Angle calculations
- Vector operations
- Point-to-segment projections
"""

import jax
import jax.numpy as jnp


@jax.jit
def euclidean_distance(p1: jnp.ndarray, p2: jnp.ndarray) -> float:
    """Compute Euclidean distance between two points.

    Args:
        p1: (2,) array [x1, y1]
        p2: (2,) array [x2, y2]

    Returns:
        Scalar distance ||p1 - p2||

    Example:
        >>> euclidean_distance(jnp.array([0.0, 0.0]), jnp.array([3.0, 4.0]))
        5.0
    """
    return jnp.linalg.norm(p1 - p2)


@jax.jit
def squared_distance(p1: jnp.ndarray, p2: jnp.ndarray) -> float:
    """Compute squared Euclidean distance (faster, no sqrt).

    Args:
        p1: (2,) array [x1, y1]
        p2: (2,) array [x2, y2]

    Returns:
        Scalar squared distance ||p1 - p2||^2
    """
    return jnp.sum((p1 - p2) ** 2)


@jax.jit
def angle_between(v1: jnp.ndarray, v2: jnp.ndarray) -> float:
    """Compute angle between two vectors in radians.

    Args:
        v1: (2,) array - first vector
        v2: (2,) array - second vector

    Returns:
        Angle in radians [0, π]

    Note:
        Uses arccos of normalized dot product.
    """
    cos_angle = jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2))
    return jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))


@jax.jit
def angle_to_point(from_point: jnp.ndarray, to_point: jnp.ndarray) -> float:
    """Compute angle from one point to another in radians.

    Args:
        from_point: (2,) array [x1, y1]
        to_point: (2,) array [x2, y2]

    Returns:
        Angle in radians [-π, π] (standard atan2 convention)

    Example:
        >>> angle_to_point(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
        0.7853981633974483  # π/4
    """
    delta = to_point - from_point
    return jnp.arctan2(delta[1], delta[0])


@jax.jit
def normalize_vector(v: jnp.ndarray) -> jnp.ndarray:
    """Normalize a vector to unit length.

    Args:
        v: (2,) array

    Returns:
        (2,) array with ||result|| = 1

    Note:
        Returns zero vector if input is zero vector.
    """
    norm = jnp.linalg.norm(v)
    return jnp.where(norm > 1e-8, v / norm, v)


@jax.jit
def point_to_segment_distance(
    point: jnp.ndarray, seg_start: jnp.ndarray, seg_end: jnp.ndarray
) -> float:
    """Compute minimum distance from a point to a line segment.

    Args:
        point: (2,) array
        seg_start: (2,) array - segment start
        seg_end: (2,) array - segment end

    Returns:
        Scalar minimum distance

    Note:
        Projects point onto line, then clamps to segment bounds.
    """
    v = seg_end - seg_start
    w = point - seg_start

    # Compute projection parameter
    v_dot_v = jnp.dot(v, v)
    t = jnp.where(v_dot_v > 1e-8, jnp.dot(w, v) / v_dot_v, 0.0)

    # Clamp to segment bounds
    t = jnp.clip(t, 0.0, 1.0)

    # Compute closest point and distance
    closest = seg_start + t * v
    return jnp.linalg.norm(point - closest)


@jax.jit
def closest_point_on_segment(
    point: jnp.ndarray, seg_start: jnp.ndarray, seg_end: jnp.ndarray
) -> jnp.ndarray:
    """Find the closest point on a line segment to a given point.

    Args:
        point: (2,) array
        seg_start: (2,) array - segment start
        seg_end: (2,) array - segment end

    Returns:
        (2,) array - closest point on segment
    """
    v = seg_end - seg_start
    w = point - seg_start

    # Compute projection parameter
    v_dot_v = jnp.dot(v, v)
    t = jnp.where(v_dot_v > 1e-8, jnp.dot(w, v) / v_dot_v, 0.0)

    # Clamp to segment bounds
    t = jnp.clip(t, 0.0, 1.0)

    # Return closest point on segment
    return seg_start + t * v


@jax.jit
def cross_product_2d(v1: jnp.ndarray, v2: jnp.ndarray) -> float:
    """Compute 2D cross product (returns scalar z-component).

    Args:
        v1: (2,) array [x1, y1]
        v2: (2,) array [x2, y2]

    Returns:
        Scalar z-component: x1*y2 - y1*x2

    Note:
        Positive = v2 is counter-clockwise from v1
        Negative = v2 is clockwise from v1
        Zero = collinear
    """
    return v1[0] * v2[1] - v1[1] * v2[0]


@jax.jit
def rotate_vector(v: jnp.ndarray, angle: float) -> jnp.ndarray:
    """Rotate a 2D vector by an angle.

    Args:
        v: (2,) array [x, y]
        angle: Rotation angle in radians (counter-clockwise)

    Returns:
        (2,) array - rotated vector

    Example:
        >>> rotate_vector(jnp.array([1.0, 0.0]), jnp.pi / 2)
        array([0., 1.])  # Rotated 90 degrees
    """
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([cos_a * v[0] - sin_a * v[1], sin_a * v[0] + cos_a * v[1]])


@jax.jit
def point_in_triangle(
    point: jnp.ndarray, v0: jnp.ndarray, v1: jnp.ndarray, v2: jnp.ndarray
) -> bool:
    """Check if a point is inside a triangle using barycentric coordinates.

    Args:
        point: (2,) array
        v0, v1, v2: (2,) arrays - triangle vertices

    Returns:
        True if point is inside triangle, False otherwise
    """
    # Use cross products to check if point is on same side of all edges
    # Edge v0->v1
    edge1 = v1 - v0
    to_point1 = point - v0
    cross1 = cross_product_2d(edge1, to_point1)

    # Edge v1->v2
    edge2 = v2 - v1
    to_point2 = point - v1
    cross2 = cross_product_2d(edge2, to_point2)

    # Edge v2->v0
    edge3 = v0 - v2
    to_point3 = point - v2
    cross3 = cross_product_2d(edge3, to_point3)

    # Point is inside if all cross products have same sign
    return (cross1 >= 0) & (cross2 >= 0) & (cross3 >= 0) | (cross1 <= 0) & (
        cross2 <= 0
    ) & (cross3 <= 0)
