import numpy as np
from utils import quat_mul, quat_conj, quat_to_rot, skew

def rigid_body_dynamics(state, Fb, tau, params):
    '''
    Docstring for rigid_body_dynamics
    
    :param state: current state of the robot
    :param Fb: external force applied to the robot in the body frame
    :param tau: external torque applied to the robot in the body frame
    :param params: physical parameters (mass, inertia, linear damping, angular damping - state-dependent)
    '''
    r, v, q, omega = state
    m, I, b_lin, b_ang = params

    m = m*2

    R_B_I = quat_to_rot(q) # Rotation from the body frame to the inertial frame

    v_b = R_B_I.T @ v  # Velocity in body frame

    # Linear drag force: linear component (b_lin is state-dependent based on power/recovery stroke)
    f_drag_linear_b = -b_lin @ v_b
    
    # Add quadratic drag term for higher speeds (hydrodynamic drag)
    speed = np.linalg.norm(v_b)
    if speed > 1e-6:
        # Quadratic drag: direction follows velocity, magnitude scales with speed²
        v_b_normalized = v_b / speed
        f_drag_quad_b = -0.2 * speed**2 * (b_lin @ v_b_normalized)
    else:
        f_drag_quad_b = np.zeros(3)
    
    f_drag_b = f_drag_linear_b + f_drag_quad_b
    f_total_b = Fb + f_drag_b

    r_dot = v
    v_dot = (R_B_I @ f_total_b) / m

    q_dot = 0.5 * quat_mul(
        q,
        np.hstack([0.0, omega])
    )

    # Rotational dynamics with angular drag damping
    # Angular drag torque: proportional to angular velocity
    tau_drag_angular = -b_ang @ omega
    
    # Add quadratic angular drag for higher rotation rates
    omega_mag = np.linalg.norm(omega)
    if omega_mag > 1e-6:
        omega_normalized = omega / omega_mag
        tau_drag_quad_angular = -0.15 * omega_mag**2 * (b_ang @ omega_normalized)
    else:
        tau_drag_quad_angular = np.zeros(3)
    
    tau_total = tau + tau_drag_angular + tau_drag_quad_angular
    omega_dot = np.linalg.inv(I) @ (tau_total - np.cross(omega, I @ omega))

    return r_dot, v_dot, q_dot, omega_dot
