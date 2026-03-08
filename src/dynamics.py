import numpy as np
from utils import quat_mul, quat_conj, quat_to_rot, skew

def rigid_body_dynamics(state, Fb, tau, params):
    '''
    :param state: current state of the robot
    :param Fb: external force applied to the robot in the body frame
    :param tau: external torque applied to the robot in the body frame
    :param params: (m, I, b_lin, b_ang, rho, volume, r_cb)
    '''
    r, v, q, omega = state
    m, I, b_lin, b_ang, rho, volume, r_cb = params
    g = 9.81

    R_B_I = quat_to_rot(q)  # rotation body -> inertial

    v_b = R_B_I.T @ v

    # Linear drag
    f_drag_linear_b = -b_lin @ v_b
    
    speed = np.linalg.norm(v_b)
    if speed > 1e-6:
        v_b_normalized = v_b / speed
        f_drag_quad_b = -0.2 * speed**2 * (b_lin @ v_b_normalized)
    else:
        f_drag_quad_b = np.zeros(3)
    
    f_drag_b = f_drag_linear_b + f_drag_quad_b

    # -----------------------------
    # Buoyancy and gravity forces
    # -----------------------------

    # Buoyancy force (upward in inertial frame)
    F_buoyancy_I = np.array([0.0, 0.0, m/2 * g])

    # Gravity force (downward in inertial frame)
    F_gravity_I = np.array([0.0, 0.0, -m/2 * g])

    # Convert both to body frame
    F_buoyancy_b = R_B_I.T @ F_buoyancy_I
    F_gravity_b = R_B_I.T @ F_gravity_I

    # Total body-frame force
    f_total_b = Fb + f_drag_b + F_buoyancy_b + F_gravity_b

    r_dot = v
    v_dot = (R_B_I @ f_total_b) / m

    q_dot = 0.5 * quat_mul(q, np.hstack([0.0, omega]))

    # -----------------------------
    # Angular drag
    # -----------------------------
    tau_drag_angular = -b_ang @ omega
    
    omega_mag = np.linalg.norm(omega)
    if omega_mag > 1e-6:
        omega_normalized = omega / omega_mag
        tau_drag_quad_angular = -0.15 * omega_mag**2 * (b_ang @ omega_normalized)
    else:
        tau_drag_quad_angular = np.zeros(3)

    # -----------------------------
    # Buoyancy torque (COB offset)
    # -----------------------------
    tau_buoyancy = 0.1 * np.cross(r_cb, F_buoyancy_b)

    tau_total = tau + tau_drag_angular + tau_drag_quad_angular + tau_buoyancy

    omega_dot = np.linalg.solve(I, tau_total - np.cross(omega, I @ omega))

    return r_dot, v_dot, q_dot, omega_dot