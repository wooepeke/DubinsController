import numpy as np
from scipy.optimize import lsq_linear


class RingThrusterAllocator:

    def __init__(self, N, r, l, f_max, phi0=0.0, broken_motors=None):

        self.N = N
        self.r = r
        self.l = l
        self.f_max = f_max
        self.broken_motors = set(broken_motors) if broken_motors else set()

        self.theta = phi0 + 2*np.pi*np.arange(N)/N

        # Wrench mapping
        self.A = np.vstack([
            np.ones(self.N),
            np.zeros(self.N),
            np.zeros(self.N),
            np.zeros(self.N),
            r*np.sin(self.theta),
            -r*np.cos(self.theta)
        ])

        self.W = np.diag([1.0, 0.0, 0.0, 0.0, 12.0, 5.0])


    # ==========================================================
    # Public interface
    # ==========================================================

    def allocate(self, Fd_B, tau_des, method="min_energy"):

        b = np.array([
            Fd_B[0],
            0.0,
            0.0,
            0.0,
            tau_des[1],
            tau_des[2]
        ])

        if method == "min_energy":
            return self._allocate_min_energy(b)

        elif method == "distributed":
            return self._allocate_distributed(b)

        else:
            raise ValueError("Unknown allocation method")


    # ==========================================================
    # Optimization methods
    # ==========================================================

    def _get_working_matrix(self):

        working = [i for i in range(self.N) if i not in self.broken_motors]

        if not working:
            return None, None

        Aw = self.W @ self.A
        return Aw[:, working], working


    def _reconstruct_full(self, x, working):

        f = np.zeros(self.N)
        for i, idx in enumerate(working):
            f[idx] = x[i]

        return f


    def _allocate_min_energy(self, b):

        A_working, working = self._get_working_matrix()
        bw = self.W @ b

        if A_working is None:
            return np.zeros(self.N)

        lambda_reg = 0.1

        A_reg = np.vstack([
            A_working,
            np.sqrt(lambda_reg) * np.eye(len(working))
        ])

        b_reg = np.hstack([
            bw,
            np.zeros(len(working))
        ])

        res = lsq_linear(A_reg, b_reg, bounds=(0.0, self.f_max))

        return self._reconstruct_full(res.x, working)


    def _allocate_distributed(self, b):

        A_working, working = self._get_working_matrix()
        bw = self.W @ b

        if A_working is None:
            return np.zeros(self.N)

        Nw = len(working)

        # Penalize difference between motors
        D = np.eye(Nw) - np.ones((Nw, Nw))/Nw

        lambda_reg = 0.1

        A_reg = np.vstack([
            A_working,
            np.sqrt(lambda_reg)*D
        ])

        b_reg = np.hstack([
            bw,
            np.zeros(Nw)
        ])

        res = lsq_linear(A_reg, b_reg, bounds=(0.0, self.f_max))

        return self._reconstruct_full(res.x, working)


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