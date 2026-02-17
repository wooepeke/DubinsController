import numpy as np
import matplotlib.pyplot as plt
from sympy import true

from controller import SE3Controller
from tentacle import TentacleScheduler
from allocator import RingThrusterAllocator, motors_to_wrench
from dynamics import rigid_body_dynamics
from utils import quat_mul, quat_conj, quat_to_rot, skew, normalize_quat, calculate_inertia, yaw_from_quat, normalize_angle, quat_from_yaw
from plots import plot_all_systems, plot_heading_debug, plot_3d_thrusters

def rk4_step(f, state, u, params, dt):
    """One RK4 integration step"""
    k1 = f(state, *u, params)

    s2 = tuple(state[i] + 0.5 * dt * k1[i] for i in range(4))
    k2 = f(s2, *u, params)

    s3 = tuple(state[i] + 0.5 * dt * k2[i] for i in range(4))
    k3 = f(s3, *u, params)

    s4 = tuple(state[i] + dt * k3[i] for i in range(4))
    k4 = f(s4, *u, params)

    next_state = tuple(
        state[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
        for i in range(4)
    )

    return next_state

def main():
    print("Starting simulation...")
    # Simulation parameters
    dt = 0.01
    T = 60.0
    time = np.arange(0.0, T, dt)
    min_error_dist = 0.1  # Target is considered reached if within 20 cm
    
    # Physical dimensions
    a = 0.21    # Semi-axis along X (the tip)
    b = 0.085   # Semi-axis along Y (base)
    c = 0.085   # Semi-axis along Z (base)
    m = 3.0     # Mass (kg)
    # Linear drag coefficients - different for power vs recovery stroke
    b_power_stroke = np.diag([0.34, 1.54, 1.54])  # Lower drag when tentacles contracted
    b_recovery_stroke = np.diag([2.5, 4.0, 4.0])  # Higher drag when tentacles extended

    # Angular drag coefficients - rotational drag from water resistance
    # Note: z-axis (yaw) has less drag than x/y (pitch/roll) since it rotates around body's long axis
    b_ang_power_stroke = np.diag([0.1, 0.1, 0.1])  # Lower angular drag during power stroke, minimal z-drag
    b_ang_recovery_stroke = np.diag([0.8, 1.2, 0.4])  # Higher angular drag on pitch/roll, but yaw easier
    com_offset_x = (3/8) * a # Center of Mass offset from the flat base

    I = calculate_inertia(a, b, c, m, com_offset_x) # Inertia tensor in the body frame, accounting for COM offset

    # Tentacle actuation parameters
    tentacle_cycle_time = 2  # Total cycle duration (seconds)
    tentacle_thrust_duration = 0.5  # Duration of thrust phase (seconds)
    scheduler = TentacleScheduler(tentacle_cycle_time, tentacle_thrust_duration)

    # Thruster allocator (ring configuration)
    N_thrusters = 8
    r_thrusters = 0.085
    l_thrusters = -0.05
    f_max = 1.0
    phi0 = 22.5
    
    # Specify which motors are broken (empty list means all motors are functional)
    # Example: broken_motors = [0, 3] would disable motors 0 and 3
    broken_motors = []  # Change this to test robustness with broken motors
    
    allocator = RingThrusterAllocator(
        N=N_thrusters,
        r=r_thrusters,
        l=l_thrusters,
        f_max=f_max,
        phi0=np.deg2rad(phi0),
        broken_motors=broken_motors,
    )

    # Controller gains - increased to overcome added angular drag
    Kr = 0.2 * np.eye(3)   # Position tracking gain - increased for better responsiveness
    Kv = 3.0 * np.eye(3)   # Velocity feedback gain - increased to use more available thrust
    Kq = 0.8 * np.eye(3)   # Attitude control - further increased for better turning
    Kw = 1.5 * np.eye(3)   # Angular velocity damping - increased for aggressive rotation
    Ki_omega = 0.0 * np.eye(3)
    Kd_v = 0.0 * np.eye(3)
    Kd_omega = 0.0 * np.eye(3)

    controller = SE3Controller(
        m=m,
        I=I,
        b=b_power_stroke,
        Kr=Kr,
        Kv=Kv,
        Kq=Kq,
        Kw=Kw,
        Ki_omega=Ki_omega,
        Kd_v=Kd_v,
        Kd_omega=Kd_omega
    )

    # Target points - list of sequential targets to visit
    targets = [
        np.array([0.5, 0.2, 0.0]),
        np.array([-1.0, 0.5, 0.0])
    ]
    current_target_idx = 0
    target = targets[current_target_idx].copy()
    
    # Initial state
    r0 = np.array([0.0, 0.0, 0.0])
    v0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
    w0 = np.zeros(3)
    
    state = (r0, v0, q0, w0)
    
    # Data logging using dictionary with timestamps
    logs = {'targets': [t.copy() for t in targets]}  # Store all targets for plotting
    power_stroke_times = []  # Track when power strokes occur
    target_reached_times = []  # Track when each target is reached
    
    # Track previous quaternion for continuity during normalization
    prev_q = q0.copy()
    prev_heading = 0.0  # Track previous heading to handle -π/π discontinuities
    prev_desired_heading = 0.0  # Track previous desired heading
    
    # Store motor forces for the current power stroke to avoid oscillations
    stored_f_motors = np.zeros(N_thrusters)
    
    print("Targets set to:")
    for i, t in enumerate(targets):
        print(f"  Target {i+1}: {t}")
    print("Starting simulation - robot will track each target sequentially...")

    # Simulation loop - position-error driven control with natural heading
    for t in time:
        r_d = target.copy()
        a_d = np.zeros(3) # No feedforward acceleration since target is stationary
        
        # Calculate desired heading toward target (for attitude control to assist turning)
        direction_to_target_2d = target[:2] - state[0][:2]
        direction_norm = np.linalg.norm(direction_to_target_2d)
        
        # Desired velocity: 0.1 m/s toward target (realistic locomotion speed)
        if direction_norm > 0.15:  # Only maintain speed until close to target
            v_d = np.zeros(3)
            v_d[:2] = 0.1 * direction_to_target_2d / direction_norm
        else:
            v_d = np.zeros(3)
        
        desired_heading = np.arctan2(direction_to_target_2d[1], direction_to_target_2d[0])
        
        # Unwrap desired heading to avoid -π/π discontinuities
        if desired_heading - prev_desired_heading > np.pi:
            desired_heading -= 2 * np.pi
        elif desired_heading - prev_desired_heading < -np.pi:
            desired_heading += 2 * np.pi
        prev_desired_heading = desired_heading

        # Convert to quaternion for attitude reference (only yaw rotation, no roll/pitch)
        q_d = quat_from_yaw(desired_heading)  # Create quaternion from yaw angle
        q_d = normalize_quat(q_d) # Ensure desired quaternion stays in same hemisphere as current state for continuity
        
        omega_d = np.zeros(3)
        
        # Create reference
        ref = [r_d, v_d, a_d, q_d, omega_d]
        
        # Check if power stroke is just starting
        if scheduler.power_stroke_just_started(t):
            # Compute control forces ONCE at the start of power stroke
            Fb_burst, tau_burst = controller.compute(state, ref, T_cycle=tentacle_cycle_time, T_burst=tentacle_thrust_duration, dt=dt)
            # Allocate to motors and store
            stored_f_motors = allocator.allocate(Fb_burst, tau_burst)
            power_stroke_times.append(t)  # Track power stroke timing

        # Get smooth thrust profile (0 to 1) for smooth force application
        thrust_profile = scheduler.get_thrust_profile(t)
        
        is_power_stroke = scheduler.power_stroke_start(t)
        if is_power_stroke:
            # Apply smooth profile to STORED motor forces (not recomputed)
            current_f_motors = thrust_profile * stored_f_motors

            Fb, tau = motors_to_wrench(
                current_f_motors,
                allocator.theta,
                allocator.r,
                allocator.l
            )
        else:
            # Default: no actuation (Recovery Stroke)
            Fb = np.zeros(3)
            tau = np.zeros(3)
            current_f_motors = np.zeros(N_thrusters) 
    
        # Select drag coefficients based on stroke phase
        b_current = b_power_stroke if is_power_stroke else b_recovery_stroke
        b_ang_current = b_ang_power_stroke if is_power_stroke else b_ang_recovery_stroke
        
        state = rk4_step(
            rigid_body_dynamics,
            state,
            (Fb, tau),
            (m, I, b_current, b_ang_current),
            dt
        )

        # -------------------------
        # Logging
        # -------------------------
        # Compute log entries
        e_r = state[0] - ref[0]
        q_d = ref[3]
        # q_e = quat_mul(quat_conj(q_d), state[2])
        q_e = quat_mul(quat_conj(state[2]), q_d)

        # if q_e[0] < 0:
        #     q_e = -q_e
        #     print("Warning: Quaternion error in opposite hemisphere, flipping for logging consistency")
        
        # Extract heading angles (yaw) from quaternions using proper formula
        desired_heading_angle = yaw_from_quat(q_d)
        current_heading_angle = yaw_from_quat(state[2])
        
        # Unwrap heading angle to handle -π/π discontinuities
        if current_heading_angle - prev_heading > np.pi:
            current_heading_angle -= 2 * np.pi
        elif current_heading_angle - prev_heading < -np.pi:
            current_heading_angle += 2 * np.pi
        prev_heading = current_heading_angle
                
        # Debug: Print heading mismatch on first few iterations
        # if t < 30.1:
        #     print(f"t={t:.3f}: desired_heading={np.rad2deg(desired_heading_angle):.1f}°, current_heading={np.rad2deg(current_heading_angle):.1f}°, q_d={q_d}, state_q={state[2]}")
        
        # Store all logged data for this timestamp
        logs[t] = {
            'r': state[0].copy(),
            'r_d': ref[0].copy(),
            'v': state[1].copy(),
            'v_d': ref[1].copy(),
            'q': state[2].copy(),
            'q_d': ref[3].copy(),
            'a': ref[2].copy(),
            'e_r': e_r,
            'q_err': np.linalg.norm(q_e[1:]),
            'omega': state[3].copy(),
            'omega_d': ref[4].copy(),
            'omega_z': state[3][2],
            'omega_z_d': ref[4][2],
            'f_motors': current_f_motors.copy(),
            'is_power_stroke': is_power_stroke,
            'heading': current_heading_angle,
            'heading_d': desired_heading_angle
        }
        
        # Check if current target has been reached
        dist_to_target = np.linalg.norm(state[0][:2] - target[:2])
        if dist_to_target < min_error_dist:
            target_reached_times.append(t)
            current_target_idx += 1
            
            if current_target_idx < len(targets):
                # Move to next target
                target = targets[current_target_idx].copy()
                print(f"Target {current_target_idx} reached at t = {t:.2f}s! Distance: {dist_to_target:.4f}m")
                print(f"Moving to target {current_target_idx + 1}: {target}")
            else:
                # All targets reached
                print(f"All {len(targets)} targets reached at t = {t:.2f}s!")
                break

    if True:
        # Plot all systems using the new modular plotting system
        plot_all_systems(logs, power_stroke_times=power_stroke_times, show_headings=False,
                        theta=allocator.theta, r=allocator.r, l=allocator.l, 
                        f_max=allocator.f_max, broken_motors=allocator.broken_motors)
        
        # Plot heading debugging with path and arrows
        # plot_heading_debug(logs)
        #         
        # Plot 3D thruster forces with animation
        result = plot_3d_thrusters(logs, allocator.theta, allocator.r, allocator.l, a=a, b=b, c=c, animate=False,
                                   broken_motors=allocator.broken_motors, power_stroke_times=power_stroke_times)
        if isinstance(result, tuple):
            fig, anim = result
        plt.show()


if __name__ == "__main__":
    main()