# plots.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation


def extract_data_from_logs(logs, t_log):
    """
    Extract arrays from logs dictionary for easier manipulation.
    
    Args:
        logs: Dictionary with timestamps as keys and data dicts as values
        t_log: Array of sorted timestamps
        
    Returns:
        Dictionary with extracted arrays
    """
    data = {
        't': t_log,
        'r': np.array([logs[t]['r'] for t in t_log]),
        'r_d': np.array([logs[t]['r_d'] for t in t_log]),
        'v': np.array([logs[t]['v'] for t in t_log]),
        'v_d': np.array([logs[t]['v_d'] for t in t_log]),
        'e_r': np.array([logs[t]['e_r'] for t in t_log]),
        'q_err': np.array([logs[t]['q_err'] for t in t_log]),
        'omega': np.array([logs[t]['omega'] for t in t_log]),
        'omega_d': np.array([logs[t]['omega_d'] for t in t_log]),
        'omega_z': np.array([logs[t]['omega_z'] for t in t_log]),
        'omega_z_d': np.array([logs[t]['omega_z_d'] for t in t_log]),
        'f_motors': np.array([logs[t]['f_motors'] for t in t_log]),
        'heading': np.array([logs[t].get('heading', 0) for t in t_log]),
        'heading_d': np.array([logs[t].get('heading_d', 0) for t in t_log]),
    }
    return data


def plot_trajectory_2d(ax, data, power_stroke_times=None, target=None, targets=None, show_headings=True):
    """Plot 2D top-down trajectory with optional power stroke markers, targets, and heading vectors."""
    ax.plot(data['r_d'][:, 0], data['r_d'][:, 1], 'r--', label='Reference', linewidth=2)
    ax.plot(data['r'][:, 0], data['r'][:, 1], 'b', label='Tracked', linewidth=2)
    ax.plot(data['r'][0, 0], data['r'][0, 1], 'go', markersize=10, label='Start')
    ax.plot(data['r'][-1, 0], data['r'][-1, 1], 'ro', markersize=10, label='End')
    
    # Plot targets if provided (multiple targets take precedence)
    if targets is not None:
        for i, t in enumerate(targets):
            ax.plot(t[0], t[1], 'r*', markersize=15, label=f'Target {i+1}', zorder=10, alpha=0.8)
            # Add semi-transparent circle around target
            circle = plt.Circle((t[0], t[1]), 0.1, color='red', fill=False, linestyle='--', 
                               linewidth=1.5, alpha=0.3, zorder=9)
            ax.add_patch(circle)
    elif target is not None:
        ax.plot(target[0], target[1], 'r*', markersize=20, label='Target', zorder=10)
        # Add semi-transparent circle around target
        circle = plt.Circle((target[0], target[1]), 0.1, color='red', fill=False, linestyle='--',
                           linewidth=1.5, alpha=0.3, zorder=9)
        ax.add_patch(circle)
    
    # Mark power stroke times on trajectory
    if power_stroke_times is not None and len(power_stroke_times) > 0:
        t_log = data['t']
        power_indices = []
        for stroke_t in power_stroke_times:
            # Find closest index to this stroke time
            idx = np.argmin(np.abs(t_log - stroke_t))
            power_indices.append(idx)
        
        # Remove duplicates and sort
        power_indices = sorted(set(power_indices))
        
        # Plot markers at power stroke positions
        if len(power_indices) > 0:
            stroke_positions = data['r'][power_indices]
            ax.scatter(stroke_positions[:, 0], stroke_positions[:, 1], 
                      c='orange', s=30, marker='x', linewidth=2, 
                      label='Power strokes', zorder=5)
    
    # Plot heading vectors
    if show_headings and 'heading' in data and 'heading_d' in data:
        # Sample every Nth point to avoid clutter
        n_arrows = min(20, len(data['t']))
        arrow_indices = np.linspace(0, len(data['t']) - 1, n_arrows, dtype=int)
        arrow_length = 0.3
        
        for idx in arrow_indices:
            pos = data['r'][idx]
            
            # Current heading (blue arrow)
            heading = data['heading'][idx]
            ax.arrow(pos[0], pos[1], 
                    arrow_length * np.cos(heading), 
                    arrow_length * np.sin(heading),
                    head_width=0.1, head_length=0.07, fc='blue', ec='blue', alpha=0.6)
            
            # Desired heading (red arrow)
            heading_d = data['heading_d'][idx]
            ax.arrow(pos[0] + 0.15, pos[1], 
                    arrow_length * np.cos(heading_d), 
                    arrow_length * np.sin(heading_d),
                    head_width=0.1, head_length=0.07, fc='red', ec='red', alpha=0.6)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('2D Position Tracking (Top-Down)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')


def plot_velocities(ax, data):
    """Plot velocity components with desired references."""
    labels = ['x', 'y', 'z']
    for i in range(3):
        ax[i].plot(data['t'], data['v_d'][:, i], 'r--', label=f'Desired', linewidth=2)
        ax[i].plot(data['t'], data['v'][:, i], 'b-', label=f'Actual', linewidth=1.5)
        ax[i].set_ylabel(f'Velocity [m/s]')
        ax[i].set_title(f'Velocity - {labels[i].upper()} Component')
        ax[i].grid(True, alpha=0.3)
        if i == 0:
            ax[i].legend(fontsize=8)
        if i == 2:
            ax[i].set_xlabel('Time [s]')


def plot_position_errors(ax, data):
    """Plot position tracking errors."""
    labels = ['x', 'y', 'z']
    for i in range(3):
        ax[i].plot(data['t'], data['e_r'][:, i], linewidth=1.5)
        ax[i].set_ylabel(f'$e_{labels[i]}$ [m]')
        ax[i].grid(True, alpha=0.3)
        if i == 0:
            ax[i].set_title('Position Error')
        if i == 2:
            ax[i].set_xlabel('Time [s]')


def plot_attitude_error(ax, data):
    """Plot attitude tracking error."""
    ax.plot(data['t'], data['q_err'], linewidth=1.5, color='purple')
    ax.set_ylabel(r'$\|\mathbf{q}_e\|$')
    ax.set_title('Attitude Tracking Error')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time [s]')


def plot_yaw_rate(ax, data):
    """Plot yaw rate tracking."""
    ax.plot(data['t'], data['omega_z_d'], 'b-', label='Desired', linewidth=2)
    ax.plot(data['t'], data['omega_z'], 'r--', label='Actual', linewidth=1.5)
    ax.set_ylabel('Angular velocity [rad/s]')
    ax.set_title('Yaw Rate Tracking')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time [s]')


def plot_heading_angle(ax, data):
    """Plot heading angle tracking."""
    if 'heading' in data and 'heading_d' in data:
        ax.plot(data['t'], data['heading_d'], 'b-', label='Desired', linewidth=2)
        ax.plot(data['t'], data['heading'], 'r--', label='Actual', linewidth=1.5)
        ax.set_ylabel('Heading [rad]')
        ax.set_title('Heading Angle Tracking')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time [s]')


def plot_heading_debug(logs):
    """
    Create a debug figure showing the robot path with heading angles visualized as arrows.
    
    Args:
        logs: Dictionary with timestamps as keys and data dicts as values
    """
    # Extract timestamps
    t_log = np.array(sorted([k for k in logs.keys() if k not in ['target', 'targets']]))
    
    # Extract data
    r = np.array([logs[t]['r'] for t in t_log])
    r_d = np.array([logs[t]['r_d'] for t in t_log])
    heading = np.array([logs[t].get('heading', 0) for t in t_log])
    heading_d = np.array([logs[t].get('heading_d', 0) for t in t_log])
    target = logs.get('target', None)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot trajectories
    ax.plot(r_d[:, 0], r_d[:, 1], 'r--', label='Desired trajectory', linewidth=2, alpha=0.7)
    ax.plot(r[:, 0], r[:, 1], 'b-', label='Actual trajectory', linewidth=2, alpha=0.7)
    
    # Plot start and end
    ax.plot(r[0, 0], r[0, 1], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(r[-1, 0], r[-1, 1], 'ro', markersize=12, label='End', zorder=5)
    
    # Plot target if provided
    if target is not None:
        ax.plot(target[0], target[1], 'r*', markersize=25, label='Target', zorder=10)
    
    # Plot heading vectors at sampled points
    n_arrows = min(100, len(t_log))  # Show more arrows than the main plot
    arrow_indices = np.linspace(0, len(t_log) - 1, n_arrows, dtype=int)
    arrow_length = 0.4
    
    for idx in arrow_indices:
        pos = r[idx]
        
        # Current heading (blue arrow)
        heading_curr = heading[idx]
        ax.arrow(pos[0], pos[1], 
                arrow_length * np.cos(heading_curr), 
                arrow_length * np.sin(heading_curr),
                head_width=0.12, head_length=0.08, fc='blue', ec='blue', alpha=0.7, linewidth=1.5)
        
        # Desired heading (red arrow) - offset slightly
        heading_des = heading_d[idx]
        ax.arrow(pos[0] + 0.2, pos[1], 
                arrow_length * np.cos(heading_des), 
                arrow_length * np.sin(heading_des),
                head_width=0.12, head_length=0.08, fc='red', ec='red', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title('Robot Path with Heading Angles (Blue=Actual, Red=Desired)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    return fig


def plot_motor_forces(ax, data):
    """Plot individual motor/tentacle forces."""
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i in range(data['f_motors'].shape[1]):
        ax.plot(data['t'], data['f_motors'][:, i], label=f'Motor {i+1}', 
                linewidth=1, color=colors[i % len(colors)])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N]')
    ax.set_title('Individual Motor/Tentacle Forces')
    ax.legend(loc='upper right', ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_3d_thrusters(logs, theta, r, l=0.05, a=0.21, b=0.085, c=0.085, animate=True, broken_motors=None, power_stroke_times=None):
    """
    Plot 3D visualization of robot trajectory with thruster forces and robot body.
    
    Args:
        logs: Dictionary with timestamps as keys and data dicts as values
        theta: Array of thruster angles (N_thrusters,)
        r: Radial distance of thrusters from z-axis
        l: Axial offset of thrusters
        a, b, c: Semi-axes of robot ellipsoid
        animate: If True, creates animation; if False, creates static plot
        broken_motors: Set or list of broken motor indices (optional)
        power_stroke_times: List of power stroke start times (optional, for sampling robot poses)
    """
    if broken_motors is None:
        broken_motors = set()
    else:
        broken_motors = set(broken_motors) if not isinstance(broken_motors, set) else broken_motors
    from utils import quat_to_rot
    
    # Extract data
    t_log = np.array(sorted([k for k in logs.keys() if k not in ['target', 'targets']]))
    r_traj = np.array([logs[t]['r'] for t in t_log])
    q_traj = np.array([logs[t]['q'] for t in t_log])
    f_motors = np.array([logs[t]['f_motors'] for t in t_log])
    
    N_thrusters = len(theta)
    thruster_positions_body = np.array([
        [l, r * np.cos(th), r * np.sin(th)] for th in theta
    ])
    
    force_scale = 0.3
    
    # Create semi-ellipsoid mesh 
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, 1, 10)
    u_grid, v_grid = np.meshgrid(u, v)
    
    x_sphere = l + a * v_grid
    y_sphere = b * np.sqrt(1 - v_grid**2) * np.cos(u_grid)
    z_sphere = c * np.sqrt(1 - v_grid**2) * np.sin(u_grid)
    
    if not animate:
        # Static plot with 8 samples and motor forces subplots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(8, 12, figure=fig)
        
        # 3D plot on the left (columns 0-4)
        ax = fig.add_subplot(gs[:, 0:5], projection='3d')
        
        # Create 8 motor force subplots on the right (columns 5-11)
        motor_axes = [fig.add_subplot(gs[i, 5:12]) for i in range(8)]
        
        ax.plot(r_traj[:, 0], r_traj[:, 1], r_traj[:, 2], 'b-', linewidth=2, label='Trajectory', alpha=0.7)
        ax.plot(r_traj[0, 0], r_traj[0, 1], r_traj[0, 2], 'go', markersize=10, label='Start')
        ax.plot(r_traj[-1, 0], r_traj[-1, 1], r_traj[-1, 2], 'ro', markersize=10, label='End')
        
        # Plot targets (either single target or multiple targets)
        if 'targets' in logs:
            for i, target in enumerate(logs['targets']):
                ax.plot(target[0], target[1], target[2], 'r*', markersize=15, alpha=0.8)
                ax.text(target[0], target[1], target[2] + 0.02, f'T{i+1}', fontsize=10, color='darkred')
                # Add semi-transparent circle around target (on xy plane)
                theta_circle = np.linspace(0, 2*np.pi, 100)
                circle_x = target[0] + 0.1 * np.cos(theta_circle)
                circle_y = target[1] + 0.1 * np.sin(theta_circle)
                circle_z = target[2] * np.ones_like(theta_circle)
                ax.plot(circle_x, circle_y, circle_z, 'r--', linewidth=1, alpha=0.3)
        elif 'target' in logs:
            target = logs['target']
            ax.plot(target[0], target[1], target[2], 'r*', markersize=20, label='Target')
            # Add semi-transparent circle around target (on xy plane)
            theta_circle = np.linspace(0, 2*np.pi, 100)
            circle_x = target[0] + 0.1 * np.cos(theta_circle)
            circle_y = target[1] + 0.1 * np.sin(theta_circle)
            circle_z = target[2] * np.ones_like(theta_circle)
            ax.plot(circle_x, circle_y, circle_z, 'r--', linewidth=1, alpha=0.3)
        
        # Determine sample indices: use power stroke times if available, else evenly distributed
        if power_stroke_times is not None and len(power_stroke_times) > 0:
            # Distribute 8 samples evenly across all power strokes to cover all targets
            n_samples = min(8, len(power_stroke_times))
            # Select evenly distributed indices from the power_stroke_times list
            selected_stroke_indices = np.linspace(0, len(power_stroke_times) - 1, n_samples, dtype=int)
            selected_stroke_times = [power_stroke_times[i] for i in selected_stroke_indices]
            
            # Find indices closest to selected power stroke times
            sample_indices = []
            for stroke_time in selected_stroke_times:
                idx = np.argmin(np.abs(t_log - stroke_time))
                if idx not in sample_indices:  # Avoid duplicates
                    sample_indices.append(idx)
            sample_indices = np.array(sorted(sample_indices))
        else:
            # Fallback: evenly distributed samples
            n_samples = min(8, len(t_log))
            sample_indices = np.linspace(0, len(t_log) - 1, n_samples, dtype=int)
        
        colors_robots = plt.cm.viridis(np.linspace(0, 1, len(sample_indices)))
        
        for sample_idx, idx in enumerate(sample_indices):
            robot_pos = r_traj[idx]
            R = quat_to_rot(q_traj[idx])
            
            ellipsoid_points = np.array([x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()])
            rotated_points = R @ ellipsoid_points
            x_rot = rotated_points[0].reshape(x_sphere.shape) + robot_pos[0]
            y_rot = rotated_points[1].reshape(y_sphere.shape) + robot_pos[1]
            z_rot = rotated_points[2].reshape(z_sphere.shape) + robot_pos[2]
            
            ax.plot_surface(x_rot, y_rot, z_rot, alpha=0.15, color=colors_robots[sample_idx])
            
            axis_length = 0.15
            ax.quiver(robot_pos[0], robot_pos[1], robot_pos[2],
                     *(R @ np.array([axis_length, 0, 0])),
                     color='red', arrow_length_ratio=0.3, linewidth=2, alpha=0.8)
            ax.quiver(robot_pos[0], robot_pos[1], robot_pos[2],
                     *(R @ np.array([0, axis_length, 0])),
                     color='green', arrow_length_ratio=0.3, linewidth=2, alpha=0.8)
            ax.quiver(robot_pos[0], robot_pos[1], robot_pos[2],
                     *(R @ np.array([0, 0, axis_length])),
                     color='blue', arrow_length_ratio=0.3, linewidth=2, alpha=0.8)
            
            for i, pos_body in enumerate(thruster_positions_body):
                pos_inertial = robot_pos + R @ pos_body
                # Color defective motors red, others orange
                motor_color = 'red' if i in broken_motors else 'orange'
                ax.scatter(*pos_inertial, s=15, c=motor_color, alpha=0.8)
                
                force_mag = f_motors[idx, i]
                if force_mag > 1e-3:
                    force_inertial = force_scale * force_mag * (R @ np.array([1, 0, 0]))
                    ax.quiver(*pos_inertial, *force_inertial,
                             color='red', alpha=0.6, arrow_length_ratio=0.3, linewidth=1)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('3D Robot Trajectory - Static View')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        max_range = np.array([
            r_traj[:, 0].max() - r_traj[:, 0].min(),
            r_traj[:, 1].max() - r_traj[:, 1].min(),
            r_traj[:, 2].max() - r_traj[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (r_traj[:, 0].max() + r_traj[:, 0].min()) * 0.5
        mid_y = (r_traj[:, 1].max() + r_traj[:, 1].min()) * 0.5
        mid_z = (r_traj[:, 2].max() + r_traj[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Plot individual motor forces on the right (8 separate plots)
        for i in range(N_thrusters):
            motor_axes[i].plot(t_log, f_motors[:, i], 'b-', linewidth=1.5)
            motor_axes[i].fill_between(t_log, 0, f_motors[:, i], alpha=0.3)
            motor_axes[i].set_ylabel(f'M{i} [N]', fontsize=8)
            motor_axes[i].grid(True, alpha=0.3)
            motor_axes[i].set_xlim(t_log[0], t_log[-1])
            motor_axes[i].tick_params(axis='both', labelsize=7)
            motor_axes[i].set_ylim(0, 1.0)

            if i == 0:
                motor_axes[i].set_title('Motor Forces', fontsize=10)
            if i == 7:
                motor_axes[i].set_xlabel('Time [s]', fontsize=8)
            else:
                motor_axes[i].set_xticklabels([])
        
        plt.tight_layout()
        return fig
    
    else:
        # Animated plot with motor forces subplots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(8, 12, figure=fig)
        
        # 3D animation on the left (columns 0-4)
        ax = fig.add_subplot(gs[:, 0:5], projection='3d')
        
        # Create 8 motor force subplots on the right (columns 5-11)
        motor_axes = [fig.add_subplot(gs[i, 5:12]) for i in range(8)]
        
        ax.plot(r_traj[:, 0], r_traj[:, 1], r_traj[:, 2], 'b-', linewidth=2, label='Trajectory', alpha=0.5)
        ax.plot(r_traj[0, 0], r_traj[0, 1], r_traj[0, 2], 'go', markersize=10, label='Start')
        
        # Plot targets (either single target or multiple targets)
        if 'targets' in logs:
            for i, target in enumerate(logs['targets']):
                ax.plot(target[0], target[1], target[2], 'r*', markersize=15, alpha=0.8)
                ax.text(target[0], target[1], target[2] + 0.02, f'T{i+1}', fontsize=9, color='darkred')
                # Add semi-transparent circle around target (on xy plane)
                theta_circle = np.linspace(0, 2*np.pi, 100)
                circle_x = target[0] + 0.1 * np.cos(theta_circle)
                circle_y = target[1] + 0.1 * np.sin(theta_circle)
                circle_z = target[2] * np.ones_like(theta_circle)
                ax.plot(circle_x, circle_y, circle_z, 'r--', linewidth=1, alpha=0.3)
        elif 'target' in logs:
            ax.plot(logs['target'][0], logs['target'][1], logs['target'][2], 'r*', markersize=20, label='Target')
            # Add semi-transparent circle around target (on xy plane)
            theta_circle = np.linspace(0, 2*np.pi, 100)
            circle_x = logs['target'][0] + 0.1 * np.cos(theta_circle)
            circle_y = logs['target'][1] + 0.1 * np.sin(theta_circle)
            circle_z = logs['target'][2] * np.ones_like(theta_circle)
            ax.plot(circle_x, circle_y, circle_z, 'r--', linewidth=1, alpha=0.3)
        
        max_range = np.array([
            r_traj[:, 0].max() - r_traj[:, 0].min(),
            r_traj[:, 1].max() - r_traj[:, 1].min(),
            r_traj[:, 2].max() - r_traj[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (r_traj[:, 0].max() + r_traj[:, 0].min()) * 0.5
        mid_y = (r_traj[:, 1].max() + r_traj[:, 1].min()) * 0.5
        mid_z = (r_traj[:, 2].max() + r_traj[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        
        # Pre-plot all motor forces as lines on the right subplots
        for i in range(N_thrusters):
            motor_axes[i].plot(t_log, f_motors[:, i], 'b-', linewidth=1.5)
            motor_axes[i].fill_between(t_log, 0, f_motors[:, i], alpha=0.3)
            motor_axes[i].set_ylabel(f'M{i} [N]', fontsize=8)
            motor_axes[i].grid(True, alpha=0.3)
            motor_axes[i].set_xlim(t_log[0], t_log[-1])
            motor_axes[i].tick_params(axis='both', labelsize=7)
            if i == 0:
                motor_axes[i].set_title('Motor Forces', fontsize=10)
            if i == 7:
                motor_axes[i].set_xlabel('Time [s]', fontsize=8)
            else:
                motor_axes[i].set_xticklabels([])
        
        plot_objs = {'surfaces': [], 'quivers': [], 'scatters': [], 'v_lines': [None] * N_thrusters}
        
        def animate_frame(frame_idx):
            actual_idx = frame_idx * 5  # Every 5th frame
            if actual_idx >= len(t_log):
                return
            
            # Clean up previous plot
            for surf in plot_objs['surfaces']:
                surf.remove()
            for quiv in plot_objs['quivers']:
                quiv.remove()
            for scatter in plot_objs['scatters']:
                scatter.remove()
            
            # Remove previous vertical lines from all motor plots
            for i in range(N_thrusters):
                if plot_objs['v_lines'][i] is not None:
                    plot_objs['v_lines'][i].remove()
            
            plot_objs['surfaces'].clear()
            plot_objs['quivers'].clear()
            plot_objs['scatters'].clear()
            
            robot_pos = r_traj[actual_idx]
            R = quat_to_rot(q_traj[actual_idx])
            
            # Plot robot body
            ellipsoid_points = np.array([x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()])
            rotated_points = R @ ellipsoid_points
            x_rot = rotated_points[0].reshape(x_sphere.shape) + robot_pos[0]
            y_rot = rotated_points[1].reshape(y_sphere.shape) + robot_pos[1]
            z_rot = rotated_points[2].reshape(z_sphere.shape) + robot_pos[2]
            
            surf = ax.plot_surface(x_rot, y_rot, z_rot, alpha=0.4, color='cyan', edgecolor='none')
            plot_objs['surfaces'].append(surf)
            
            # Plot axes
            axis_length = 0.15
            plot_objs['quivers'].append(ax.quiver(robot_pos[0], robot_pos[1], robot_pos[2],
                                                   *(R @ np.array([axis_length, 0, 0])),
                                                   color='red', arrow_length_ratio=0.3, linewidth=2.5, alpha=0.9))
            plot_objs['quivers'].append(ax.quiver(robot_pos[0], robot_pos[1], robot_pos[2],
                                                   *(R @ np.array([0, axis_length, 0])),
                                                   color='green', arrow_length_ratio=0.3, linewidth=2.5, alpha=0.9))
            plot_objs['quivers'].append(ax.quiver(robot_pos[0], robot_pos[1], robot_pos[2],
                                                   *(R @ np.array([0, 0, axis_length])),
                                                   color='blue', arrow_length_ratio=0.3, linewidth=2.5, alpha=0.9))
            
            # Plot thrusters and forces
            for i, pos_body in enumerate(thruster_positions_body):
                pos_inertial = robot_pos + R @ pos_body
                # Color defective motors red, others orange
                motor_color = 'red' if i in broken_motors else 'orange'
                sc = ax.scatter(*pos_inertial, s=35, c=motor_color, alpha=0.9, edgecolors='darkorange', linewidth=1)
                plot_objs['scatters'].append(sc)
                
                force_mag = f_motors[actual_idx, i]
                if force_mag > 1e-3:
                    force_inertial = force_scale * force_mag * (R @ np.array([1, 0, 0]))
                    q = ax.quiver(*pos_inertial, *force_inertial,
                                 color='red', alpha=0.75, arrow_length_ratio=0.3, linewidth=1.5)
                    plot_objs['quivers'].append(q)
            
            # Add vertical line to all motor forces plots showing current time
            for i in range(N_thrusters):
                plot_objs['v_lines'][i] = motor_axes[i].axvline(x=t_log[actual_idx], color='red', 
                                                                  linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax.set_title(f'3D Robot Animation (t={t_log[actual_idx]:.2f}s)', fontsize=12)
        
        n_frames = (len(t_log) + 4) // 5
        anim = FuncAnimation(fig, animate_frame, frames=n_frames, interval=100, repeat=True, blit=False)
        
        plt.tight_layout()
        return fig, anim


def plot_motor_configuration(ax, theta, r, l, f_max, broken_motors=None):
    """
    Plot the motor configuration as seen from above (ring configuration around robot).
    
    Args:
        ax: Matplotlib axis
        theta: Array of motor angles
        r: Radial distance of motors from center
        l: x-offset of motors (for reference)
        f_max: Max force per motor (for scale)
        broken_motors: Set of broken motor indices (optional)
    """
    if broken_motors is None:
        broken_motors = set()
    
    # Draw robot body (circle)
    circle = plt.Circle((0, 0), 0.05, color='blue', alpha=0.3, label='Robot Body')
    ax.add_patch(circle)
    
    # Draw motors as arrows around the ring
    N = len(theta)
    for i, angle in enumerate(theta):
        # Motor position in body frame (x-y plane, z=0)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        
        # Motor thrust direction (always along x-axis in body frame)
        dx = 0.05 * np.cos(0)  # Thrust is along x-axis
        dy = 0.05 * np.sin(0)
        
        # Color based on broken status
        if i in broken_motors:
            color = 'red'
            marker = 'x'
            label_suffix = " (BROKEN)"
        else:
            color = 'green'
            marker = 'o'
            label_suffix = ""
        
        # Plot motor position
        ax.plot(x, y, marker=marker, markersize=8, color=color)
        
        # Plot thrust direction arrow
        ax.arrow(x, y, dx, dy, head_width=0.015, head_length=0.01, 
                fc=color, ec=color, alpha=0.7)
        
        # Label motor
        ax.text(x * 1.15, y * 1.15, f'M{i}', fontsize=9, ha='center', va='center')
    
    # Set axis properties
    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.15, 0.15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Motor Configuration (Top View)')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Functional Motor'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=10, label='Broken Motor'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)


def plot_all_systems(logs, enable_plots=None, power_stroke_times=None, show_headings=False, 
                     theta=None, r=None, l=None, f_max=None, broken_motors=None):
    """
    Plot all system data in one comprehensive figure using logs dictionary.
    
    Args:
        logs: Dictionary with timestamps as keys and data dicts as values
        enable_plots: Dict controlling which plots to show. 
                     Default shows all. Example:
                     {'trajectory': True, 'velocities': True, ...}
        power_stroke_times: List of times when power strokes occur (optional)
        show_headings: Whether to show heading arrows on trajectory plot (default: False)
        theta: Motor angles array (optional, for motor configuration plot)
        r: Motor radial distance (optional, for motor configuration plot)
        l: Motor x-offset (optional, for motor configuration plot)
        f_max: Motor max force (optional, for motor configuration plot)
        broken_motors: Set of broken motor indices (optional)
    """
    # Default: show all plots
    if enable_plots is None:
        enable_plots = {
            'trajectory': True,
            'velocities': True,
            'position_errors': True,
            'attitude': True,
            'yaw': True,
            'heading': True,
            'motors': True,
        }
    
    # Extract data from logs
    t_log = np.array(sorted([k for k in logs.keys() if k not in ['target', 'targets']]))
    data = extract_data_from_logs(logs, t_log)
    
    # Extract target(s) if they exist in logs
    targets = logs.get('targets', None)
    target = logs.get('target', None)
    
    # Create figure with adaptive grid based on enabled plots
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(6, 6, figure=fig)
    
    plot_idx = 0
    
    # ===== Trajectory =====
    if enable_plots.get('trajectory', True):
        ax2d = fig.add_subplot(gs[0:2, 2:4])
        plot_trajectory_2d(ax2d, data, power_stroke_times=power_stroke_times, target=target, targets=targets, show_headings=show_headings)
        plot_idx += 1
    
    # ===== Velocities =====
    if enable_plots.get('velocities', True):
        axes = [fig.add_subplot(gs[i, 0:2]) for i in range(3)]
        plot_velocities(axes, data)
        plot_idx += 1
    
    # ===== Position Errors =====
    if enable_plots.get('position_errors', True):
        axes = [fig.add_subplot(gs[i, 4:6]) for i in range(3)]
        plot_position_errors(axes, data)
        plot_idx += 1
    
    # ===== Attitude Error =====
    if enable_plots.get('attitude', True):
        ax_q = fig.add_subplot(gs[3, 0:3])
        plot_attitude_error(ax_q, data)
    
    # ===== Yaw Rate =====
    if enable_plots.get('yaw', True):
        ax_yaw = fig.add_subplot(gs[3, 3:6])
        plot_yaw_rate(ax_yaw, data)
    
    # ===== Heading Angle =====
    if enable_plots.get('heading', True):
        ax_heading = fig.add_subplot(gs[4, 0:4])
        plot_heading_angle(ax_heading, data)
    
    # ===== Motor Configuration =====
    if theta is not None and r is not None:
        ax_motors_cfg = fig.add_subplot(gs[4:6, 4:6])
        plot_motor_configuration(ax_motors_cfg, theta, r, l, f_max, broken_motors=broken_motors)
        plot_idx += 1
    
    # ===== Motor Forces =====
    if enable_plots.get('motors', True):
        ax_motors = fig.add_subplot(gs[5, 0:4])
        plot_motor_forces(ax_motors, data)
    
    plt.tight_layout()


# Legacy function for backward compatibility
def plot_all_systems_legacy(t, r, r_d, v, v_d, e_r, q_err, omega_z_d, omega_z, f_motors, power_stroke_times=None):
    """
    [DEPRECATED] Legacy function - use plot_all_systems(logs) instead.
    Plot all system data in one comprehensive figure.
    """
    # Convert arrays back to logs format
    logs = {}
    for idx, t_val in enumerate(t):
        logs[t_val] = {
            'r': r[idx],
            'r_d': r_d[idx],
            'v': v[idx],
            'v_d': v_d[idx],
            'e_r': e_r[idx],
            'q_err': q_err[idx],
            'omega_z': omega_z[idx],
            'omega_z_d': omega_z_d[idx],
            'f_motors': f_motors[idx],
        }
    
    plot_all_systems(logs, power_stroke_times=power_stroke_times)


