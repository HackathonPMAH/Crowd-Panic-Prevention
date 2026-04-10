from __future__ import annotations

from typing import Dict, List


class RecommendationEngine:
    def generate(self, risk: Dict[str, object], hotspots: List[str]) -> List[str]:
        status = risk["status"]

        if status == "SAFE":
            return ["Monitor situation"]

        if status == "WARNING":
            zone = hotspots[0] if hotspots else "the busiest zone"
            return [
                f"Redirect crowd from {zone}",
                "Open additional exits",
            ]

        return [
            "Immediate evacuation required",
            "Open all exits",
            "Send alert to authorities",
        ]

    def get_recommendations(self, risk: Dict[str, object], hotspots: List[str] | None = None) -> List[str]:
        return self.generate(risk, hotspots or [])
