"""
Signal Smoothing Filters for 3D Pose Coordinates & Joint Angles
Removes jitter and high-frequency noise from camera frames.
"""

import math
import numpy as np
from collections import deque

class MovingAverageFilter:
    """
    Simple Moving Average (SMA) filter.
    Maintains a rolling window of recent values.
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, val: float) -> float:
        self.history.append(val)
        return float(np.mean(self.history))

    def reset(self):
        self.history.clear()


class LowPassFilter:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.last_val = None

    def update(self, val: float, alpha: float = None) -> float:
        if alpha is None:
            alpha = self.alpha
        if self.last_val is None:
            self.last_val = val
            return val
        filtered = alpha * val + (1.0 - alpha) * self.last_val
        self.last_val = filtered
        return filtered

    def reset(self):
        self.last_val = None


class OneEuroFilter:
    """
    1€ (One Euro) Filter: An adaptive low-pass filter specifically designed
    for human-computer interaction and noisy tracking signals.
    - Adapts cutoff frequency based on movement speed:
      * When moving slowly: Low cutoff -> High smoothing (No jitter/shaking)
      * When moving quickly: High cutoff -> Low lag (Realtime tracking)
    """
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def update(self, val: float, timestamp: float) -> float:
        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filter.update(val, 1.0)

        dt = timestamp - self.last_time
        self.last_time = timestamp
        if dt <= 1e-5:
            dt = 1e-4
        rate = 1.0 / dt

        # Estimate derivative (speed)
        dx = (val - self.x_filter.last_val) * rate if self.x_filter.last_val is not None else 0.0
        edx = self.dx_filter.update(dx, self._alpha(rate, self.d_cutoff))

        # Dynamic cutoff frequency based on speed
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter.update(val, self._alpha(rate, cutoff))

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None
