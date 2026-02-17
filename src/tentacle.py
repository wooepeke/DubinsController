import numpy as np

class TentacleScheduler:
    def __init__(self, cycle_time, thrust_duration):
        self.cycle_time = cycle_time
        self.thrust_duration = thrust_duration
        self.last_cycle_index = -1
        self.was_in_power_stroke = False

    def cycle_start(self, t):
        cycle_index = int(t // self.cycle_time)
        if cycle_index != self.last_cycle_index:
            self.last_cycle_index = cycle_index
            return True
        return False

    def power_stroke_start(self, t):
        """Returns True if in power stroke phase (for backward compatibility)"""
        t_in_cycle = t % self.cycle_time
        return t_in_cycle >= (self.cycle_time - self.thrust_duration)
    
    def power_stroke_just_started(self, t):
        """Returns True only at the moment the power stroke starts (transition from recovery to power)"""
        is_in_power_stroke = self.power_stroke_start(t)
        just_started = is_in_power_stroke and not self.was_in_power_stroke
        self.was_in_power_stroke = is_in_power_stroke
        return just_started
    
    def get_thrust_profile(self, t):
        """
        Returns a smooth thrust profile (0 to 1) using a cosine ramp.
        Creates a smooth curve that starts high and gradually decreases during power stroke.
        """
        t_in_cycle = t % self.cycle_time
        power_stroke_start_time = self.cycle_time - self.thrust_duration
        
        # If outside power stroke, return 0
        if t_in_cycle < power_stroke_start_time or t_in_cycle >= self.cycle_time:
            return 0.0
        
        # Within power stroke: use cosine ramp for smooth profile
        # t_normalized goes from 0 to 1 during the power stroke
        t_normalized = (t_in_cycle - power_stroke_start_time) / self.thrust_duration
        
        # Cosine profile: starts at 1, gradually decreases to 0
        # Using (1 + cos(π * t_normalized)) / 2 gives a smooth curve from 1 to 0
        profile = 0.5 * (1.0 + np.cos(np.pi * t_normalized))
        
        return profile