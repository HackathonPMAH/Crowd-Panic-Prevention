from __future__ import annotations

from typing import Iterable, List

import numpy as np

from utils.config import GRID_COLS, GRID_ROWS

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None


def _gaussian_kernel(size: int = 5, sigma: float = 1.2) -> np.ndarray:
    axis = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel_sum = float(kernel.sum())
    return kernel / kernel_sum if kernel_sum else kernel


def _smooth_with_numpy(grid: np.ndarray) -> np.ndarray:
    kernel = _gaussian_kernel()
    pad = kernel.shape[0] // 2
    padded = np.pad(grid, pad, mode="edge")
    smoothed = np.zeros_like(grid, dtype=np.float32)

    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            window = padded[row : row + kernel.shape[0], col : col + kernel.shape[1]]
            smoothed[row, col] = float(np.sum(window * kernel))

    return smoothed


def generate_heatmap(points: Iterable[dict], width: int, height: int) -> List[List[float]]:
    """Create a smoothed, normalized density heatmap matrix."""
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    cell_w = max(width / GRID_COLS, 1)
    cell_h = max(height / GRID_ROWS, 1)

    for point in points:
        col = min(int(point["x"] / cell_w), GRID_COLS - 1)
        row = min(int(point["y"] / cell_h), GRID_ROWS - 1)
        grid[row, col] += 1.0

    heatmap = cv2.GaussianBlur(grid, ksize=(0, 0), sigmaX=1.2, sigmaY=1.2) if cv2 else _smooth_with_numpy(grid)
    max_value = float(np.max(heatmap))
    if max_value > 0:
        heatmap = heatmap / max_value

    return np.round(heatmap, 3).tolist()
