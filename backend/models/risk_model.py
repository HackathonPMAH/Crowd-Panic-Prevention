from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from utils.config import MAX_REFERENCE_SPEED


@dataclass
class RiskModel:
    """Lightweight feature normalization for the rule-based risk engine."""

    max_density_reference: float = 12.0

    def normalize(self, features: Dict[str, float]) -> Dict[str, float]:
        density_score = np.clip(features.get("max_density", 0.0) / self.max_density_reference, 0.0, 1.0)
        velocity_score = np.clip(features.get("avg_velocity", 0.0) / MAX_REFERENCE_SPEED, 0.0, 1.0)
        conflict_score = np.clip(features.get("direction_conflict", 0.0), 0.0, 1.0)
        turbulence_score = np.clip(features.get("turbulence", 0.0), 0.0, 1.0)
        return {
            "density": float(density_score),
            "velocity": float(velocity_score),
            "direction_conflict": float(conflict_score),
            "turbulence": float(turbulence_score),
        }
