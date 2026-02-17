import numpy as np
from scipy.optimize import lsq_linear



class RingThrusterAllocator:
    def __init__(self, N, r, l, f_max, phi0=0.0, broken_motors=None):
        """
        N: number of thrusters
        r: radial distance from z-axis (m)
        l: x-offset from COM (forward/aft along body x-axis) (m)
        f_max: max force per thruster (N)
        phi0: initial phase angle (rad)
        broken_motors: list of motor indices that are broken/disabled (e.g., [1, 3])
        
        Note: x-offset does NOT create moments when forces are purely in x-direction
        (force is parallel to offset vector, no perpendicular component).
        The offset is noted for geometry but doesn't affect force/torque allocation.
        """
        self.N = N
        self.r = r
        self.l = l  # Store for reference, but doesn't affect moment arms for x-direction forces
        self.f_max = f_max
        self.broken_motors = set(broken_motors) if broken_motors else set()
        # Weight matrix: [Fx, Fy, Fz, tau_x, tau_y, tau_z]
        # Fy and Fz should be minimized (can't be controlled with x-direction thrusters)
        # tau_x should be minimized (can't be controlled with ring thrusters)
        self.W = np.diag([1.0, 10.0, 10.0, 10.0, 5.0, 5.0])

        self.theta = phi0 + 2 * np.pi * np.arange(N) / N

        self.A = np.vstack([
            np.ones(self.N),                 # Fx = sum of all forces
            np.zeros(self.N),                # Fy = 0 (can't produce with x-direction thrusters)
            np.zeros(self.N),                # Fz = 0 (can't produce with x-direction thrusters)
            np.zeros(self.N),                # tau_x = 0 (moment arm perpendicular to force)
            self.r * np.sin(self.theta),     # tau_y = r*sin(theta)*F
            -self.r * np.cos(self.theta)     # tau_z = -r*cos(theta)*F
        ])

    def allocate(self, Fd_B, tau_des):
        """
        Fd_B: [Fd_x, Fd_y, Fd_z] desired force in body frame (only Fd_x is relevant)
        tau_des: [tau_x, tau_y, tau_z]
        
        Uses least-squares with regularization to encourage distributed thrust across all motors.
        Fy, Fz, and tau_x are penalized to minimize unwanted motions.
        Broken motors are forced to output zero force.
        """
        # Build the desired wrench vector (6 components)
        # Thrusters can only control Fx, tau_y, tau_z
        # We want to minimize Fy, Fz, and tau_x
        b = np.array([
            Fd_B[0],       # Fx desired
            0.0,           # Fy desired (should be zero)
            0.0,           # Fz desired (should be zero)
            0.0,           # tau_x desired (should be zero)
            tau_des[1],    # tau_y desired
            tau_des[2]     # tau_z desired
        ])

        self.Aw = self.W @ self.A
        bw = self.W @ b
        
        # Get indices of working (non-broken) motors
        working_indices = [i for i in range(self.N) if i not in self.broken_motors]
        
        # If all motors are broken, return zero forces
        if not working_indices:
            return np.zeros(self.N)
        
        # Extract only the columns for working motors
        A_working = self.Aw[:, working_indices]
        
        # Add regularization term
        lambda_reg = 0.1
        A_reg = np.vstack([A_working, np.sqrt(lambda_reg) * np.eye(len(working_indices))])
        b_reg = np.hstack([bw, np.zeros(len(working_indices))])
        
        # Optimize only for working motors
        res = lsq_linear(A_reg, b_reg, bounds=(0.0, self.f_max))
        
        # Reconstruct full force vector with zeros for broken motors
        f_full = np.zeros(self.N)
        for idx, working_idx in enumerate(working_indices):
            f_full[working_idx] = res.x[idx]
        
        return f_full

    
def motors_to_wrench(f, theta, r, l):
    """
    Convert individual motor forces to body-frame wrench.
    
    f: motor forces array
    theta: thruster angles
    r: radial offset from z-axis
    l: x-offset from COM (included for completeness but doesn't affect wrench)
    """
    Fx = np.sum(f)
    tau_y = np.sum(r * np.sin(theta) * f)
    tau_z = np.sum(-r * np.cos(theta) * f)

    F = np.array([Fx, 0.0, 0.0])
    tau = np.array([0.0, tau_y, tau_z])

    return F, tau