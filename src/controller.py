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
        
        # State tracking for derivative action
        self.integral_e_omega = np.zeros(3)
        self.prev_e_omega = np.zeros(3)
        self.prev_e_v = np.zeros(3)  # Track previous velocity error for derivative
        self.prev_e_R = np.zeros(3)   # Track previous attitude error for derivative

    def compute_forces(self, state, ref, dt=0.01):
        r, v, q, omega = state
        r_d, rdot_d, rddot_d, q_d, omega_d = ref

        R = quat_to_rot(q)

        # Errors in inertial frame
        e_r = r - r_d
        e_v = v - rdot_d

        # Derivative of velocity error (numerical differentiation)
        if dt > 0:
            de_v_dt = (e_v - self.prev_e_v) / dt
        else:
            de_v_dt = np.zeros(3)
        self.prev_e_v = e_v.copy()

        # Drag in inertial frame
        D_I = R @ self.b @ R.T
        drag_I = D_I @ v

        # Desired inertial force
        F_I = (
            self.m * rddot_d
            + drag_I
            - self.Kr @ e_r
            - self.Kv @ e_v
            - self.Kd_v @ de_v_dt
        )

        # Convert to body frame
        F_b = R.T @ F_I

        return F_b
        
    def compute_forces_hybrid(self, state, ref, T_cycle, T_burst):
        r, v, q, omega = state
        r_d, rdot_d, rddot_d, q_d, omega_d = ref

        R = quat_to_rot(q)

        e_r = r - r_d
        e_v = v - rdot_d

        duty = T_burst / T_cycle
        scale = 1.0 / duty

        # Momentum-based burst force (in inertial frame)
        F_I_burst = scale * (
            - self.Kr @ e_r
            - self.Kv @ e_v
        )

        # Convert to body frame
        F_b_burst = R.T @ F_I_burst

        return F_b_burst

    def compute_torques(self, state, ref, dt=0.01):
        r, v, q, omega = state
        r_d, rdot_d, rddot_d, q_d, omega_d = ref

        q = normalize_quat(q)
        # Normalize q_d relative to q to keep them in the same hemisphere
        q_d = normalize_quat(q_d, q_prev=q)

        q_err = quat_mul(quat_conj(q), q_d)

        if q_err[0] < 0:
            q_err = -q_err

        e_R = q_err[1:] # Attitude error vector

        # Derivative of attitude error
        if dt > 0:
            de_R_dt = (e_R - self.prev_e_R) / dt
        else:
            de_R_dt = np.zeros(3)
        self.prev_e_R = e_R.copy()

        e_omega = omega - omega_d # Angular velocity error

        tau_control = -self.I @ (self.Kq @ e_R + self.Kw @ e_omega + self.Kd_omega @ de_R_dt)
        tau_gyro = np.cross(omega, self.I @ omega)

        tau = tau_control + tau_gyro

        return tau

    def compute_torques_hybrid(self, state, ref, T_cycle, T_burst, dt=0.01):
        r, v, q, omega = state
        r_d, rdot_d, rddot_d, q_d, omega_d = ref

        q = normalize_quat(q)
        q_d = normalize_quat(q_d, q_prev=q)

        q_err = quat_mul(quat_conj(q_d), q)
        
        if q_err[0] < 0:
            q_err = -q_err

        e_R = q_err[1:]
        
        # Derivative of attitude error
        if dt > 0:
            de_R_dt = (e_R - self.prev_e_R) / dt
        else:
            de_R_dt = np.zeros(3)
        self.prev_e_R = e_R.copy()
        
        e_omega = omega - omega_d

        duty = T_burst / T_cycle
        scale = 1.0 / duty

        tau_burst = scale * (
            - self.Kq @ e_R
            - self.Kw @ e_omega
            - self.Kd_omega @ de_R_dt
        )

        return tau_burst

    def compute(self, state, ref, T_cycle=None, T_burst=None, dt=0.01):
        """
        state: r, v, q, omega
        ref: r_d, rdot_d, rddot_d, q_d, omega_d
        T_cycle, T_burst: burst control parameters
        dt: time step for derivative calculations
        """

        if T_cycle is not None and T_burst is not None:
            Fb = self.compute_forces_hybrid(state, ref, T_cycle, T_burst)
            tau = self.compute_torques_hybrid(state, ref, T_cycle, T_burst, dt=dt)
        else:
            Fb = self.compute_forces(state, ref, dt=dt)
            tau = self.compute_torques(state, ref, dt=dt)

        return Fb, tau
