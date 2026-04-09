from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None


class CVProcessor:
    """Basic motion extraction fallback for video-based crowd estimation."""

    def __init__(self) -> None:
        self._background_subtractor = (
            cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25) if cv2 else None
        )

    def process_video_frame(self, video_path: str) -> Dict[str, object]:
        if cv2 is None or self._background_subtractor is None:
            return {
                "mode": "video",
                "available": False,
                "people": [],
                "message": "OpenCV is not installed; falling back to simulation mode",
            }

        path = Path(video_path)
        if not path.exists():
            return {"mode": "video", "available": False, "people": [], "message": "Video file not found"}

        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            return {"mode": "video", "available": False, "people": [], "message": "Unable to read video frame"}

        mask = self._background_subtractor.apply(frame)
        _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        people: List[dict] = []
        for index, contour in enumerate(contours, start=1):
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < 150:
                continue
            roi = mask[y : y + h, x : x + w]
            intensity = float(np.mean(roi)) / 255.0
            people.append(
                {
                    "id": index,
                    "x": int(x + w / 2),
                    "y": int(y + h / 2),
                    "vx": round(intensity * 2.5, 3),
                    "vy": round(intensity * 1.5, 3),
                }
            )

        return {
            "mode": "video",
            "available": True,
            "people": people,
            "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        }
