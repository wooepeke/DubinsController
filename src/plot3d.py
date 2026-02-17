import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from utils import quat_to_rot


def create_robot_box(center, orientation, size=0.3):
    """
    Create a 3D box representing the robot body.
    
    Parameters:
    - center: xyz position of robot center
    - orientation: 3x3 rotation matrix
    - size: characteristic size of the robot
    
    Returns:
    - vertices: 8x3 array of box corners in world frame
    """
    # Define box vertices in body frame (centered at origin)
    half_size = size / 2
    vertices_body = np.array([
        [-half_size, -half_size, -half_size],
        [half_size, -half_size, -half_size],
        [half_size, half_size, -half_size],
        [-half_size, half_size, -half_size],
        [-half_size, -half_size, half_size],
        [half_size, -half_size, half_size],
        [half_size, half_size, half_size],
        [-half_size, half_size, half_size],
    ])
    
    # Rotate and translate to world frame
    vertices_world = vertices_body @ orientation.T + center
    return vertices_world


def draw_robot_frame(ax, center, orientation, size=0.3):
    """
    Draw coordinate frame arrows at robot center.
    
    Parameters:
    - ax: matplotlib 3D axis
    - center: xyz position of robot center
    - orientation: 3x3 rotation matrix
    - size: length of frame arrows
    """
    # X-axis (red)
    ax.quiver(center[0], center[1], center[2],
              orientation[0, 0] * size, orientation[1, 0] * size, orientation[2, 0] * size,
              color='r', arrow_length_ratio=0.3, linewidth=2)
    # Y-axis (green)
    ax.quiver(center[0], center[1], center[2],
              orientation[0, 1] * size, orientation[1, 1] * size, orientation[2, 1] * size,
              color='g', arrow_length_ratio=0.3, linewidth=2)
    # Z-axis (blue)
    ax.quiver(center[0], center[1], center[2],
              orientation[0, 2] * size, orientation[1, 2] * size, orientation[2, 2] * size,
              color='b', arrow_length_ratio=0.3, linewidth=2)


def draw_heading_arrow(ax, position, heading_angle, color='b', size=0.3, label=None):
    """
    Draw a 2D heading arrow on the XY plane at given position.
    
    Parameters:
    - ax: matplotlib 3D axis
    - position: xyz position
    - heading_angle: yaw angle in radians
    - color: color of the arrow
    - size: length of the arrow
    - label: optional label for the arrow
    """
    # Project heading direction onto XY plane
    dx = size * np.cos(heading_angle)
    dy = size * np.sin(heading_angle)
    dz = 0  # Keep at same height
    
    ax.quiver(position[0], position[1], position[2],
              dx, dy, dz,
              color=color, arrow_length_ratio=0.3, linewidth=2, alpha=0.8)


def animate_3d_trajectory(r_log, r_d_log, q_log, t_log, fps=30):
    """
    Create an animated 3D visualization of the robot trajectory with orientation.
    
    Parameters:
    - r_log: Nx3 array of actual robot positions
    - r_d_log: Nx3 array of desired robot positions
    - q_log: Nx4 array of quaternions (w, x, y, z)
    - t_log: N array of time stamps
    - fps: frames per second for the animation
    """
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot full trajectories
    ax.plot(r_log[:, 0], r_log[:, 1], r_log[:, 2], 'b-', linewidth=2, label='Actual trajectory', alpha=0.7)
    ax.plot(r_d_log[:, 0], r_d_log[:, 1], r_d_log[:, 2], 'r--', linewidth=2, label='Desired trajectory', alpha=0.7)
    
    # Initialize scatter plots for start and end points
    ax.scatter(*r_log[0], color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(*r_log[-1], color='red', s=100, marker='s', label='End', zorder=5)
    
    # Initialize artist objects that will be updated
    line_actual, = ax.plot([], [], [], 'b-', linewidth=3, label='Current path segment')
    scatter_current = ax.scatter([], [], [], color='cyan', s=200, marker='*', zorder=10)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Robot Trajectory Animation')
    
    # Set equal aspect ratio for 3D plot
    all_positions = np.vstack([r_log, r_d_log])
    max_range = np.array([all_positions[:, 0].max() - all_positions[:, 0].min(),
                          all_positions[:, 1].max() - all_positions[:, 1].min(),
                          all_positions[:, 2].max() - all_positions[:, 2].min()]).max() / 2.0
    
    mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
    mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
    mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend(loc='upper left', fontsize=10)
    
    # Estimate number of frames (don't exceed reasonable number)
    total_frames = min(len(r_log), int(len(r_log) / max(1, int(len(r_log) / 300))))
    frame_indices = np.linspace(0, len(r_log) - 1, total_frames, dtype=int)
    
    # Need quaternions in proper format for rotation matrix
    # Assuming q_log is in [w, x, y, z] format
    def quat_to_rotation_matrix(q):
        """Convert quaternion [w, x, y, z] to rotation matrix"""
        w, x, y, z = q
        return quat_to_rot(q)  # Using the utility function from utils.py
    
    def update(frame_num):
        nonlocal scatter_current
        idx = frame_indices[frame_num]
        
        # Update trajectory line (show last 50 points for clarity)
        start_idx = max(0, idx - 50)
        line_actual.set_data(r_log[start_idx:idx, 0], r_log[start_idx:idx, 1])
        line_actual.set_3d_properties(r_log[start_idx:idx, 2])
        
        # Update current position marker
        scatter_current.remove()
        scatter_current = ax.scatter([r_log[idx, 0]], [r_log[idx, 1]], [r_log[idx, 2]], 
                                     color='cyan', s=200, marker='*', zorder=10)
        
        # Draw robot body and frame at current position
        current_q = q_log[idx] if hasattr(q_log, '__len__') and len(q_log) > idx else np.array([1, 0, 0, 0])
        current_R = quat_to_rotation_matrix(current_q)
        
        # Draw frame axes
        frame_size = 0.5
        draw_robot_frame(ax, r_log[idx], current_R, size=frame_size)
        
        # Update title with current time and position info
        ax.set_title(f'3D Robot Trajectory Animation (t={t_log[idx]:.2f}s, Frame {frame_num+1}/{total_frames})')
        
        return line_actual, scatter_current
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)
    
    plt.show()


def animate_3d_with_subplots(r_log, r_d_log, q_log, t_log, e_r_log, f_motors_log, fps=30):
    """
    Create a detailed animation with multiple subplots showing trajectory, error, and forces.
    
    Parameters:
    - r_log: Nx3 array of actual robot positions
    - r_d_log: Nx3 array of desired robot positions
    - q_log: Nx4 array of quaternions
    - t_log: N array of time stamps
    - e_r_log: Nx3 array of position errors
    - f_motors_log: NxM array of motor forces
    - fps: frames per second
    """
    
    fig = plt.figure(figsize=(16, 10))
    
    # 3D trajectory plot
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax_3d.plot(r_log[:, 0], r_log[:, 1], r_log[:, 2], 'b-', linewidth=2, alpha=0.6)
    ax_3d.plot(r_d_log[:, 0], r_d_log[:, 1], r_d_log[:, 2], 'r--', linewidth=2, alpha=0.6)
    ax_3d.scatter(*r_log[0], color='green', s=100, marker='o')
    ax_3d.scatter(*r_log[-1], color='red', s=100, marker='s')
    
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Trajectory')
    
    # Set equal aspect ratio
    all_positions = np.vstack([r_log, r_d_log])
    max_range = np.array([all_positions[:, 0].max() - all_positions[:, 0].min(),
                          all_positions[:, 1].max() - all_positions[:, 1].min(),
                          all_positions[:, 2].max() - all_positions[:, 2].min()]).max() / 2.0
    mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
    mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
    mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Position error plot
    ax_err = fig.add_subplot(2, 2, 2)
    error_mag = np.linalg.norm(e_r_log, axis=1)
    ax_err.plot(t_log, error_mag, 'r-', linewidth=2)
    ax_err.set_xlabel('Time (s)')
    ax_err.set_ylabel('Position Error Magnitude (m)')
    ax_err.set_title('Position Error Over Time')
    ax_err.grid(True, alpha=0.3)
    
    # Motor forces plot
    ax_motors = fig.add_subplot(2, 2, 3)
    for i in range(f_motors_log.shape[1]):
        ax_motors.plot(t_log, f_motors_log[:, i], alpha=0.7, label=f'Motor {i+1}')
    ax_motors.set_xlabel('Time (s)')
    ax_motors.set_ylabel('Force (N)')
    ax_motors.set_title('Motor Forces')
    ax_motors.legend(loc='best', fontsize=8)
    ax_motors.grid(True, alpha=0.3)
    
    # XY projection with trail
    ax_xy = fig.add_subplot(2, 2, 4)
    ax_xy.plot(r_log[:, 0], r_log[:, 1], 'b-', linewidth=2, alpha=0.6, label='Actual')
    ax_xy.plot(r_d_log[:, 0], r_d_log[:, 1], 'r--', linewidth=2, alpha=0.6, label='Desired')
    ax_xy.scatter(r_log[0, 0], r_log[0, 1], color='green', s=100, marker='o')
    ax_xy.scatter(r_log[-1, 0], r_log[-1, 1], color='red', s=100, marker='s')
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('XY Projection')
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis('equal')
    
    # For animation
    line_3d, = ax_3d.plot([], [], [], 'b-', linewidth=3)
    scatter_3d = ax_3d.scatter([], [], [], color='cyan', s=200, marker='*', zorder=10)
    
    line_xy, = ax_xy.plot([], [], 'b-', linewidth=3)
    scatter_xy = ax_xy.scatter([], [], color='cyan', s=200, marker='*', zorder=10)
    
    vline_err = ax_err.axvline(t_log[0], color='cyan', linewidth=2, alpha=0.7)
    vline_motors = ax_motors.axvline(t_log[0], color='cyan', linewidth=2, alpha=0.7)
    
    total_frames = min(len(r_log), int(len(r_log) / max(1, int(len(r_log) / 300))))
    frame_indices = np.linspace(0, len(r_log) - 1, total_frames, dtype=int)
    
    def quat_to_rotation_matrix(q):
        return quat_to_rot(q)
    
    def update(frame_num):
        nonlocal scatter_3d, scatter_xy
        idx = frame_indices[frame_num]
        
        # Update 3D trajectory
        start_idx = max(0, idx - 50)
        line_3d.set_data(r_log[start_idx:idx, 0], r_log[start_idx:idx, 1])
        line_3d.set_3d_properties(r_log[start_idx:idx, 2])
        
        scatter_3d.remove()
        scatter_3d = ax_3d.scatter([r_log[idx, 0]], [r_log[idx, 1]], [r_log[idx, 2]], 
                                   color='cyan', s=200, marker='*', zorder=10)
        
        # Update XY projection
        line_xy.set_data(r_log[start_idx:idx, 0], r_log[start_idx:idx, 1])
        scatter_xy.remove()
        scatter_xy = ax_xy.scatter([r_log[idx, 0]], [r_log[idx, 1]], 
                                   color='cyan', s=200, marker='*', zorder=10)
        
        # Update time indicators
        vline_err.set_xdata([t_log[idx]])
        vline_motors.set_xdata([t_log[idx]])
        
        return line_3d, scatter_3d, line_xy, scatter_xy, vline_err, vline_motors
    
    anim = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)
    
    plt.show()
