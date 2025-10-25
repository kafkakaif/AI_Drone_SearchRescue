import numpy as np

def center_window(depth: np.ndarray, window_ratio: float = 0.2):
    """
    Returns the center crop of the depth map.
    """
    h, w = depth.shape[:2]
    wr = max(1, int(w * window_ratio))
    hr = max(1, int(h * window_ratio))
    x0 = (w - wr) // 2
    y0 = (h - hr) // 2
    return depth[y0 : y0 + hr, x0 : x0 + wr]

def safe_median(arr: np.ndarray) -> float:
    """Median ignoring invalid/inf/zero values."""
    a = arr[np.isfinite(arr)]
    a = a[a > 0]
    if a.size == 0:
        return float("inf")
    return float(np.median(a))
