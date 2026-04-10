from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np

from models.risk_model import RiskModel
from utils.config import CRITICAL_THRESHOLD, RISK_WEIGHTS, SAFE_THRESHOLD, WARNING_THRESHOLD


class RiskEngine:
    def __init__(self) -> None:
        self.model = RiskModel()
        self.score_history: Deque[float] = deque(maxlen=12)

    def compute_risk(self, features: Dict[str, float]) -> Dict[str, object]:
        normalized = self.model.normalize(features)
        weighted_score = sum(normalized[name] * weight for name, weight in RISK_WEIGHTS.items())
        risk_score = round(float(np.clip(weighted_score * 100.0, 0.0, 100.0)), 2)
        self.score_history.append(risk_score)

        trend, slope = self._trend()
        time_to_disaster = self._estimate_time_to_disaster(risk_score, slope)

        if risk_score >= WARNING_THRESHOLD:
            status = "CRITICAL" if risk_score >= CRITICAL_THRESHOLD else "WARNING"
        elif risk_score >= SAFE_THRESHOLD:
            status = "WARNING"
        else:
            status = "SAFE"

        return {
            "risk_score": risk_score,
            "status": status,
            "trend": trend,
            "time_to_disaster": time_to_disaster,
            "normalized_factors": normalized,
        }

    def get_recent_scores(self, limit: int = 10) -> list[float]:
        if limit <= 0:
            return []
        history = list(self.score_history)
        return history[-limit:]

    def _trend(self) -> Tuple[str, float]:
        if len(self.score_history) < 3:
            return "stable", 0.0

        x_axis = np.arange(len(self.score_history))
        slope, _ = np.polyfit(x_axis, np.array(self.score_history), 1)
        if slope > 1.2:
            return "increasing", float(slope)
        if slope < -1.2:
            return "decreasing", float(slope)
        return "stable", float(slope)

    def _estimate_time_to_disaster(self, current_score: float, slope: float) -> int | None:
        if current_score >= CRITICAL_THRESHOLD:
            return 0 if current_score >= CRITICAL_THRESHOLD else None

        # Use moving average of recent per-tick deltas to reduce noise.
        if len(self.score_history) >= 4:
            diffs = np.diff(np.array(self.score_history, dtype=np.float32))
            recent = diffs[-3:]
            avg_delta = float(np.mean(recent))
            if avg_delta > 0.25:
                remaining = CRITICAL_THRESHOLD - current_score
                return int(max(round(remaining / avg_delta), 1))

        if slope <= 0:
            return None

        remaining = CRITICAL_THRESHOLD - current_score
        return int(max(round(remaining / slope), 1))
