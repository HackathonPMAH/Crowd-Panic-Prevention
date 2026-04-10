from __future__ import annotations

from typing import Dict, List

import numpy as np


class FlowAnalyzer:
    def __init__(self) -> None:
        self._previous_average_speed = 0.0

    def analyze(self, people: List[dict]) -> Dict[str, object]:
        if not people:
            return {
                "average_direction": {"x": 0.0, "y": 0.0},
                "direction_conflict": 0.0,
                "opposing_movement_clusters": 0,
                "sudden_spike": False,
                "avg_speed_delta": 0.0,
                "avg_velocity": 0.0,
                "turbulence": 0.0,
            }

        velocities = np.array([[person["vx"], person["vy"]] for person in people], dtype=np.float32)
        magnitudes = np.linalg.norm(velocities, axis=1)
        avg_velocity = float(np.mean(magnitudes))

        normalized = velocities.copy()
        non_zero = magnitudes > 1e-6
        normalized[non_zero] = normalized[non_zero] / magnitudes[non_zero][:, None]
        average_direction = normalized.mean(axis=0)

        if np.linalg.norm(average_direction) > 1e-6:
            average_direction = average_direction / np.linalg.norm(average_direction)

        direction_consistency = float(np.clip(np.linalg.norm(normalized.mean(axis=0)), 0.0, 1.0))
        direction_conflict = 1.0 - direction_consistency

        headings = np.arctan2(velocities[:, 1], velocities[:, 0])
        positive_cluster = int(np.sum(headings >= 0))
        negative_cluster = int(np.sum(headings < 0))
        opposing_clusters = int(positive_cluster > 0 and negative_cluster > 0)

        avg_speed_delta = avg_velocity - self._previous_average_speed
        sudden_spike = avg_speed_delta > 0.8
        self._previous_average_speed = avg_velocity

        turbulence = float(np.clip(np.std(headings) / np.pi + np.std(magnitudes) / 4.0, 0.0, 1.0))

        return {
            "average_direction": {
                "x": round(float(average_direction[0]), 3),
                "y": round(float(average_direction[1]), 3),
            },
            "direction_conflict": round(float(direction_conflict), 3),
            "opposing_movement_clusters": opposing_clusters,
            "sudden_spike": sudden_spike,
            "avg_speed_delta": round(float(avg_speed_delta), 3),
            "avg_velocity": round(avg_velocity, 3),
            "turbulence": round(turbulence, 3),
        }

    def extract_features(self, people: List[dict]) -> Dict[str, object]:
        return self.analyze(people)
