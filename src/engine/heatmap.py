"""
Heatmap accumulator — rolling density map of person positions.

Instead of using sv.HeatMapAnnotator (current-frame only), this maintains
a custom accumulator that:
1. Increments a Gaussian at each person's center position every frame
2. Applies exponential decay so old positions fade
3. Normalizes and renders as a color-mapped overlay

This gives a smooth, real-time heatmap showing traffic patterns over a
configurable time window.
"""

import time
import cv2
import numpy as np
from loguru import logger
from src.core.config import settings


class HeatmapAccumulator:
    """Per-camera rolling heatmap of person positions."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.last_decay_time = time.time()
        self.total_weight = 0.0

        # Pre-compute Gaussian kernel for blurring positions
        ksize = settings.HEATMAP_RADIUS * 2 + 1
        self.kernel = self._make_gaussian_kernel(ksize, settings.HEATMAP_RADIUS)

    @staticmethod
    def _make_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """Create a 2D Gaussian kernel."""
        ax = np.arange(size) - size // 2
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def add_position(self, cx: float, cy: float, now: float = None):
        """Add a person's center position to the heatmap."""
        if now is None:
            now = time.time()

        self._decay(now)

        # Convert to integer pixel coordinates
        ix = int(np.clip(cx, 0, self.width - 1))
        iy = int(np.clip(cy, 0, self.height - 1))

        # Stamp Gaussian at this position
        ksize = self.kernel.shape[0]
        half = ksize // 2

        # Compute kernel placement bounds (clamped to image edges)
        y1 = max(0, iy - half)
        y2 = min(self.height, iy + half + 1)
        x1 = max(0, ix - half)
        x2 = min(self.width, ix + half + 1)

        # Kernel slice
        ky1 = y1 - (iy - half)
        ky2 = ky1 + (y2 - y1)
        kx1 = x1 - (ix - half)
        kx2 = kx1 + (x2 - x1)

        self.heatmap[y1:y2, x1:x2] += self.kernel[ky1:ky2, kx1:kx2]
        self.total_weight += 1.0

    def _decay(self, now: float):
        """Apply exponential decay based on elapsed time."""
        dt = now - self.last_decay_time
        if dt <= 0:
            return

        # Decay factor: HEATMAP_DECAY_RATE^dt
        # With default 0.95 and 30fps: ~0.95^0.033 ≈ 0.9983 per frame (very smooth)
        decay = settings.HEATMAP_DECAY_RATE ** dt
        self.heatmap *= decay
        self.total_weight *= decay
        self.last_decay_time = now

    def get_heatmap(self) -> np.ndarray:
        """Return normalized heatmap as uint8 (0-255)."""
        if self.total_weight < 1e-6:
            return np.zeros((self.height, self.width), dtype=np.uint8)

        normalized = self.heatmap / max(self.total_weight, 1e-6)
        return (normalized * 255).astype(np.uint8)

    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Render heatmap as a color-mapped overlay on the frame."""
        if not settings.HEATMAP_ENABLED:
            return frame

        heatmap_uint8 = self.get_heatmap()
        if heatmap_uint8.max() == 0:
            return frame

        # Apply colormap (hot: black -> red -> yellow -> white)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored = cv2.GaussianBlur(colored, (15, 15), 0)

        # Create mask where heatmap has data
        mask = heatmap_uint8 > 5  # threshold to avoid coloring empty areas

        # Blend: only overlay where there's actual heatmap data
        overlay = frame.copy()
        overlay[mask] = cv2.addWeighted(
            frame[mask], 1.0 - settings.HEATMAP_OPACITY,
            colored[mask], settings.HEATMAP_OPACITY,
            0
        )
        return overlay

    def reset(self):
        """Clear the heatmap."""
        self.heatmap.fill(0)
        self.total_weight = 0.0
        self.last_decay_time = time.time()
