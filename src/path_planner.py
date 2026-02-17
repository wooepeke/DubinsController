import numpy as np
from scipy.interpolate import CubicSpline, interp1d


class DubinsPathPlanner:
    """
    Path planner for a Dubins-like swimming robot.
    Generates paths composed of straight lines and circular arcs.
    """
    
    def __init__(self, min_turn_radius=0.5, max_speed=1.0):
        """
        Args:
            min_turn_radius: Minimum turning radius (m) based on robot dynamics
            max_speed: Maximum forward speed (m/s)
        """
        self.min_turn_radius = min_turn_radius
        self.max_speed = max_speed
    
    def plan_path(self, start_pos, start_heading, target_pos, target_heading=None):
        """
        Plan a Dubins path from start to target.
        
        Args:
            start_pos: [x, y, z] starting position
            start_heading: Starting yaw angle (rad)
            target_pos: [x, y, z] target position
            target_heading: Desired yaw at target (rad). If None, face target.
        
        Returns:
            waypoints: Nx3 array of waypoints along the path
            headings: N array of yaw angles along the path
            path_length: Total path length
        """
        # 2D path planning (xy plane) with height blending
        start_xy = start_pos[:2]
        target_xy = target_pos[:2]
        
        # Calculate direct distance and angle to target
        direction = target_xy - start_xy
        dist_to_target = np.linalg.norm(direction)
        
        if dist_to_target < 0.1:  # Already at target
            waypoints = np.array([start_pos, target_pos])
            headings = np.array([start_heading, start_heading])
            return waypoints, headings, 0.0
        
        target_heading_direct = np.arctan2(direction[1], direction[0])
        
        if target_heading is None:
            target_heading = target_heading_direct
        
        # Generate simple Dubins-like path: turn -> straight -> turn
        # Using smooth heading interpolation
        
        # Estimate path segments
        n_segments = max(10, int(dist_to_target / self.min_turn_radius) + 5)
        path_param = np.linspace(0, 1, n_segments)
        
        # Interpolate positions along a smooth curve
        waypoints_xy = self._interpolate_positions(start_xy, target_xy, path_param)
        
        # Calculate headings along the path (smooth transition from start to target)
        headings = self._interpolate_headings(
            start_heading, target_heading, path_param, start_xy, target_xy
        )
        
        # Add height interpolation
        start_z = start_pos[2]
        target_z = target_pos[2]
        z_values = start_z + (target_z - start_z) * path_param
        
        waypoints = np.column_stack([waypoints_xy, z_values])
        
        # Calculate path length
        diffs = np.diff(waypoints, axis=0)
        path_length = np.sum(np.linalg.norm(diffs, axis=1))
        
        return waypoints, headings, path_length
    
    def _interpolate_positions(self, start_xy, target_xy, param):
        """Interpolate positions with slight curvature for Dubins-like motion"""
        n = len(param)
        
        # Simple smooth interpolation (can be enhanced with Dubins curves)
        direction = target_xy - start_xy
        dist = np.linalg.norm(direction)
        
        # Create curved path for smoother turns
        # Use parametric curve: quadratic blend
        positions = np.zeros((n, 2))
        
        for i, t in enumerate(param):
            # Smooth interpolation with ease-in-ease-out
            s = self._smooth_step(t)
            positions[i] = start_xy + s * direction
        
        return positions
    
    def _interpolate_headings(self, start_heading, target_heading, param, 
                              start_xy, target_xy):
        """Generate smooth heading interpolation"""
        n = len(param)
        headings = np.zeros(n)
        
        # Normalize angle difference
        angle_diff = self._normalize_angle(target_heading - start_heading)
        
        for i, t in enumerate(param):
            s = self._smooth_step(t)
            headings[i] = self._normalize_angle(start_heading + s * angle_diff)
        
        return headings
    
    @staticmethod
    def _smooth_step(t):
        """Smooth step function: f(t) = 3t^2 - 2t^3"""
        return 3 * t**2 - 2 * t**3
    
    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))


class TrajectoryGenerator:
    """
    Generates smooth reference trajectories for the controller.
    Converts waypoints into time-based reference commands.
    """
    
    def __init__(self, cruise_speed=0.5, max_acceleration=0.2, dt=0.01):
        """
        Args:
            cruise_speed: Desired forward speed (m/s)
            max_acceleration: Maximum acceleration (m/s^2)
            dt: Time step for discretization
        """
        self.cruise_speed = cruise_speed
        self.max_acceleration = max_acceleration
        self.dt = dt
        self.current_trajectory = None
        self.trajectory_time = None
        self.waypoints = None
        self.headings = None
    
    def generate_trajectory(self, waypoints, headings, path_length):
        """
        Generate time-based trajectory from waypoints.
        
        Args:
            waypoints: Nx3 array of waypoints
            headings: N array of heading angles
            path_length: Total path length
        
        Returns:
            time_array: Time values for trajectory
            trajectory: Dict with position, velocity, acceleration, heading, and angular velocity
        """
        self.waypoints = waypoints
        self.headings = headings
        
        # Calculate time to traverse path
        travel_time = path_length / self.cruise_speed if self.cruise_speed > 0 else 1.0
        travel_time = max(travel_time, 5.0)  # Minimum 5 seconds
        
        n_points = int(travel_time / self.dt) + 1
        time_array = np.linspace(0, travel_time, n_points)
        
        # Interpolate waypoints at each time step
        arc_lengths = self._calculate_arc_lengths(waypoints)
        distance_array = arc_lengths[-1] * time_array / travel_time  # Linear distance progression
        
        # Position interpolation
        position_interp = interp1d(
            arc_lengths, waypoints, axis=0, kind='cubic', fill_value='extrapolate'
        )
        positions = position_interp(distance_array)
        
        # Velocity (derivative of position)
        velocities = np.gradient(positions, time_array, axis=0)
        
        # Acceleration (derivative of velocity)
        accelerations = np.gradient(velocities, time_array, axis=0)
        
        # Heading interpolation
        heading_interp = interp1d(
            arc_lengths, headings, kind='cubic', fill_value='extrapolate'
        )
        heading_array = heading_interp(distance_array)
        
        # Angular velocity (yaw rate)
        heading_diff = np.diff(heading_array)
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))  # Normalize
        omega_z = np.concatenate([[0], heading_diff / np.diff(time_array)])
        
        # Convert headings to quaternions
        quaternions = self._headings_to_quaternions(heading_array)
        
        # Store trajectory for later use
        self.current_trajectory = {
            'time': time_array,
            'position': positions,
            'velocity': velocities,
            'acceleration': accelerations,
            'heading': heading_array,
            'quaternion': quaternions,
            'omega_z': omega_z
        }
        
        self.trajectory_time = travel_time
        
        return time_array, self.current_trajectory
    
    def get_reference(self, current_time):
        """
        Get reference state at a given time.
        
        Args:
            current_time: Current simulation time
        
        Returns:
            ref: [r_d, v_d, a_d, q_d, omega_d]
        """
        if self.current_trajectory is None:
            return None
        
        # Clamp time to trajectory duration
        t = np.clip(current_time, 0, self.trajectory_time)
        
        # Find nearest time index
        time_array = self.current_trajectory['time']
        idx = np.searchsorted(time_array, t)
        idx = np.clip(idx, 0, len(time_array) - 1)
        
        r_d = self.current_trajectory['position'][idx]
        v_d = self.current_trajectory['velocity'][idx]
        a_d = self.current_trajectory['acceleration'][idx]
        q_d = self.current_trajectory['quaternion'][idx]
        
        # Angular velocity: [0, 0, omega_z]
        omega_d = np.array([0, 0, self.current_trajectory['omega_z'][idx]])
        
        return [r_d, v_d, a_d, q_d, omega_d]
    
    def is_trajectory_complete(self, current_time):
        """Check if trajectory is complete"""
        if self.trajectory_time is None:
            return True
        return current_time >= self.trajectory_time
    
    @staticmethod
    def _calculate_arc_lengths(waypoints):
        """Calculate cumulative arc length along waypoints"""
        diffs = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        return arc_lengths
    
    @staticmethod
    def _headings_to_quaternions(headings):
        """Convert yaw angles to quaternions (rotation around z-axis only)"""
        n = len(headings)
        quaternions = np.zeros((n, 4))
        
        for i, yaw in enumerate(headings):
            # Quaternion from yaw angle: [w, x, y, z]
            quaternions[i] = np.array([
                np.cos(yaw / 2),
                0,
                0,
                np.sin(yaw / 2)
            ])
        
        return quaternions
