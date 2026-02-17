import numpy as np
from utils import quat_mul, quat_conj, quat_to_rot, normalize_quat

class SE3Controller:
    def __init__(self, m, I, b, Kr, Kv, Kq, Kw, Ki_omega=None, Kd_v=None, Kd_omega=None):
        self.m = m
        self.I = I
        self.b = b
        
        # Proportional gains
        self.Kr = Kr
        self.Kv = Kv
        self.Kq = Kq
        self.Kw = Kw
        self.cross_track_gain = 1.0
        
        # Integral gains (optional)
        self.Ki_omega = Ki_omega if Ki_omega is not None else np.zeros((3, 3))
        
        # Derivative gains (optional)
        self.Kd_v = Kd_v if Kd_v is not None else np.zeros((3, 3))
        self.Kd_omega = Kd_omega if Kd_omega is not None else np.zeros((3, 3))
        
        self.integral_e_omega = np.zeros(3)
        self.prev_e_omega = np.zeros(3)

    def compute_forces(self, state, ref):
        r, v, q, omega = state
        r_d, rdot_d, rddot_d, q_d, omega_d = ref

        R = quat_to_rot(q)

        # Errors in inertial frame
        e_r = r - r_d
        e_v = v - rdot_d

        # Drag in inertial frame
        D_I = R @ self.b @ R.T
        drag_I = D_I @ v

        # Desired inertial force
        F_I = (
            self.m * rddot_d
            + drag_I
            - self.Kr @ e_r
            - self.Kv @ e_v
        )

        # Convert to body frame
        F_b = R.T @ F_I

        return F_b
    
    def compute_torques(self, state, ref):
        r, v, q, omega = state
        r_d, rdot_d, rddot_d, q_d, omega_d = ref

        q = normalize_quat(q)
        # Normalize q_d relative to q to keep them in the same hemisphere
        q_d = normalize_quat(q_d, q_prev=q)

        q_err = quat_mul(quat_conj(q), q_d)

        if q_err[0] < 0:
            q_err = -q_err

        e_R = q_err[1:] # Attitude error vector

        e_omega = omega - omega_d # Angular velocity error

        tau_control = -self.I @ (self.Kq @ e_R + self.Kw @ e_omega)
        tau_gyro = np.cross(omega, self.I @ omega)

        tau = tau_control + tau_gyro

        return tau

    def compute(self, state, ref):
        """
        state: r, v, q, omega
        ref: r_d, rdot_d, rddot_d, q_d, omega_d
        dt: time step for integral and derivative calculations
        """

        Fb = self.compute_forces(state, ref)
        tau = self.compute_torques(state, ref)

        return Fb, tau
