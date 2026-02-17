import numpy as np

def skew(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

def quat_mul(q1, q2):
    """ Quaternion multiplication q = q1 ⊗ q2 """
    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], q2[1:]
    return np.hstack([
        w1*w2 - v1 @ v2,
        w1*v2 + w2*v1 + np.cross(v1, v2)
    ])

def quat_conj(q):
    return np.hstack([q[0], -q[1:]])

def quat_to_rot(q):
    """ R_B^I(q) """
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ])

def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def normalize_quat(q, q_prev=None):
    """
    Normalize quaternion and ensure it stays in consistent hemisphere (w >= 0).
    If q_prev is provided, ensures quaternion takes shortest path from q_prev.
    """
    q_norm = q / np.linalg.norm(q)
    
    # If previous quaternion provided, ensure shortest path (dot product > 0)
    if q_prev is not None:
        if np.dot(q_norm, q_prev) < 0:
            q_norm = -q_norm
    else:
        # Keep quaternion in upper hemisphere (w >= 0) to avoid discontinuities
        if q_norm[0] < 0:
            q_norm = -q_norm
    
    return q_norm


def yaw_from_quat(q):
    w, x, y, z = q
    return np.arctan2(
        2 * (w*z + x*y),
        1 - 2 * (y*y + z*z)
    )

def quat_from_yaw(yaw):
    """Convert yaw angle to quaternion (rotation around z-axis only)"""
    half_yaw = yaw / 2  # Negate for z-up convention
    return np.array([
        np.cos(half_yaw),
        0,
        0,
        np.sin(half_yaw)
    ])

def calculate_inertia(a, b, c, m, com_offset_x):
    # Moments of Inertia about the Base (Flat Face)
    Ixx_base = (1/5) * m * (b**2 + c**2)
    Iyy_base = (1/5) * m * (a**2 + c**2)
    Izz_base = (1/5) * m * (a**2 + b**2)

    # Parallel Axis Theorem to move MoI to the Center of Mass
    # I_com = I_base - mass * distance^2
    Ixx = Ixx_base # No shift for the axis of symmetry
    Iyy = Iyy_base - m * (com_offset_x**2)
    Izz = Izz_base - m * (com_offset_x**2)

    return np.diag([Ixx, Iyy, Izz])