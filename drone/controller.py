import numpy as np
from ai.utils import center_window, safe_median

class AvoidanceController:
    """
    Rule-based avoidance:
      - Look at median depth in a small center window.
      - If too close (< threshold meters) => "turn"
      - Else => "forward"
    """
    def __init__(self, depth_threshold_m: float = 5.0, window_ratio: float = 0.2):
        self.depth_threshold = float(depth_threshold_m)
        self.window_ratio = float(window_ratio)

    def decide(self, depth_map: np.ndarray) -> str:
        win = center_window(depth_map, self.window_ratio)
        med = safe_median(win)
        if med < self.depth_threshold:
            return "turn"
        return "forward"
