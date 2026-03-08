import numpy as np

class DubinsReference:
    def __init__(self, path, r0, psi0, m=None, Fx_max=None, N_thrusters=None):
        self.path = path
        self.r0 = r0
        self.psi0 = psi0
        
        # For velocity limiting based on control authority
        self.m = m
        self.Fx_max = Fx_max
        self.N_thrusters = N_thrusters
        
        # If thruster parameters provided, compute total available force
        if Fx_max is not None and N_thrusters is not None:
            self.Fx_total = Fx_max * N_thrusters
        else:
            self.Fx_total = None

        self.seg_times = []
        for seg in path.segments:
            self.seg_times.append(seg.length / path.v)

    def get(self, t):
        """
        Returns:
        r_d, rdot_d, rddot_d, q_d, omega_d
        """

        v = self.path.v
        r = self.r0.copy()
        psi = self.psi0
        t_remain = t

        for i, seg in enumerate(self.path.segments):
            Ti = self.seg_times[i]

            if t_remain <= Ti:
                return self._eval_segment(seg, r, psi, t_remain)
            else:
                # advance full segment
                r, _, _, _, _ = self._eval_segment(seg, r, psi, Ti, final=True)
                # Update psi based on segment
                if seg.type == 'S':
                    psi = psi
                elif seg.type == 'L':
                    psi = psi + (1.0 / seg.radius) * self.path.v * Ti
                elif seg.type == 'R':
                    psi = psi + (-1.0 / seg.radius) * self.path.v * Ti
                t_remain -= Ti

        # End of path → hold last pose
        return self._eval_segment(seg, r, psi, 0.0, final=True)
    
    def _eval_segment(self, seg, r0, psi0, t, final=False):
        v_nominal = self.path.v
        gamma = seg.gamma

        if seg.type == 'S':
            psi = psi0
            kappa = 0.0

        elif seg.type == 'L':
            kappa = 1.0 / seg.radius
            psi = psi0 + kappa * v_nominal * t

        elif seg.type == 'R':
            kappa = -1.0 / seg.radius
            psi = psi0 + kappa * v_nominal * t

        # ===== Velocity limiting based on available control authority =====
        # v_max = sqrt(Fx_total / (m * |kappa|))
        # This ensures we can generate the required centripetal acceleration
        if self.Fx_total is not None and self.m is not None and abs(kappa) > 1e-6:
            v_max = np.sqrt(self.Fx_total / (self.m * abs(kappa)))
            v = min(v_nominal, v_max)
        else:
            v = v_nominal

        # Calculate position - handle circular motion for turns
        if abs(kappa) < 1e-6:
            # Straight segment
            v_dir = np.array([
                np.cos(psi) * np.cos(gamma),
                np.sin(psi) * np.cos(gamma),
                np.sin(gamma)
            ])
            r = r0 + v * t * v_dir
        else:
            # Curved segment - use circular arc parametrization
            # For a curve with curvature κ: r(s) = r0 + (1/κ) * [sin(ψ(s)) - sin(ψ0), -cos(ψ(s)) + cos(ψ0), 0]
            # where ψ(s) = ψ0 + κ*s and s = v*t is arc length
            s = v * t
            r_xy = r0[:2] + (1.0 / kappa) * np.array([
                np.sin(psi) - np.sin(psi0),
                -np.cos(psi) + np.cos(psi0)
            ])
            # Add vertical component
            r = np.array([r_xy[0], r_xy[1], r0[2] + s * np.sin(gamma)])

        # Velocity direction
        v_dir = np.array([
            np.cos(psi) * np.cos(gamma),
            np.sin(psi) * np.cos(gamma),
            np.sin(gamma)
        ])

        # Velocity & acceleration
        if final:
            # At the end, hold the final position with zero velocity/acceleration
            rdot = np.zeros(3)
            rddot = np.zeros(3)
            omega_d = np.zeros(3)
        else:
            rdot = v * v_dir

            rddot = v**2 * kappa * np.array([
                -np.sin(psi) * np.cos(gamma),
                 np.cos(psi) * np.cos(gamma),
                 0.0
            ])
            
            omega_d = np.array([0.0, 0.0, kappa * v])

        # Desired attitude (use velocity direction when moving, else use current direction)
        if np.linalg.norm(rdot) > 1e-6:
            x_b = rdot / np.linalg.norm(rdot)
        else:
            # Use current heading direction when stationary
            x_b = np.array([
                np.cos(psi) * np.cos(gamma),
                np.sin(psi) * np.cos(gamma),
                np.sin(gamma)
            ])
        
        z_b = np.array([0, 0, 1])
        y_b = np.cross(z_b, x_b)
        y_b /= np.linalg.norm(y_b)
        z_b = np.cross(x_b, y_b)

        R_d = np.column_stack((x_b, y_b, z_b))

        qw = 0.5 * np.sqrt(1 + np.trace(R_d))
        q_d = np.array([
            qw,
            (R_d[2,1] - R_d[1,2]) / (4*qw),
            (R_d[0,2] - R_d[2,0]) / (4*qw),
            (R_d[1,0] - R_d[0,1]) / (4*qw)
        ])

        return r, rdot, rddot, q_d, omega_d



class DubinsSegment:
    def __init__(self, seg_type, length, radius, gamma):
        """
        seg_type: 'L', 'R', 'S'
        length: segment length [m]
        radius: turning radius (inf for straight)
        gamma: climb angle [rad]
        """
        self.type = seg_type
        self.length = length
        self.radius = radius
        self.gamma = gamma

class DubinsPath3D:
    def __init__(self, segments, v):
        self.segments = segments
        self.v = v
        self.total_length = sum(seg.length for seg in segments)


def plan_dubins_path_to_target(current_pos, current_heading, target_pos, velocity=0.1, turning_radius=3.0):
    """
    Plan a Dubins-like path from current position to target with three segments (turn-straight-turn).

    Args:
        current_pos: [x, y, z] current position
        current_heading: psi (yaw angle in radians)
        target_pos: [x, y, z] target position
        velocity: forward velocity
        turning_radius: turning radius for Dubins curves

    Returns:
        DubinsPath3D object
    """

    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    dz = target_pos[2] - current_pos[2]
    distance_2d = np.sqrt(dx**2 + dy**2)

    target_heading = np.arctan2(dy, dx)
    heading_change = target_heading - current_heading
    heading_change = np.arctan2(np.sin(heading_change), np.cos(heading_change))  # Normalize to [-pi, pi]

    # Determine first turn direction and length
    if heading_change > 0:
        first_turn_type = 'L'
    else:
        first_turn_type = 'R'

    first_turn_length = turning_radius * abs(heading_change)

    # Straight segment length (distance between turns)
    chord_length = 2 * turning_radius * np.sin(abs(heading_change) / 2)
    straight_length = max(0.1, distance_2d - chord_length)  # Ensure minimum length

    # Second turn aligns with the target heading
    second_turn_type = 'L' if first_turn_type == 'R' else 'R'
    second_turn_length = first_turn_length  # Symmetric turn

    # Climb angle (for vertical component)
    if distance_2d > 0.1:
        gamma = np.arctan2(dz, distance_2d)
    else:
        gamma = np.arctan2(dz, 0.1) if abs(dz) > 1e-6 else 0.0

    # Build path segments
    segments = [
        DubinsSegment(first_turn_type, first_turn_length, turning_radius, gamma),
        DubinsSegment('S', straight_length, np.inf, gamma),
        DubinsSegment(second_turn_type, second_turn_length, turning_radius, gamma)
    ]

    path = DubinsPath3D(segments, velocity)
    return path