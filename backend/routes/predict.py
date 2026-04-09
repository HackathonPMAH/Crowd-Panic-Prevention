from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from fastapi import APIRouter, Query

from backend.routes.simulation import simulation_state
from backend.services.cv_processor import CVProcessor
from backend.services.flow_analyzer import FlowAnalyzer
from backend.services.heatmap import generate_heatmap
from backend.services.recommendation_engine import RecommendationEngine
from backend.services.risk_engine import RiskEngine
from backend.utils.config import FRAME_HEIGHT, FRAME_WIDTH, GRID_COLS, GRID_ROWS, HOTSPOT_THRESHOLD


router = APIRouter(tags=["prediction"])

flow_analyzer = FlowAnalyzer()
risk_engine = RiskEngine()
recommendation_engine = RecommendationEngine()
cv_processor = CVProcessor()


def _extract_features(people: List[dict], width: int, height: int) -> Tuple[Dict[str, object], List[List[float]], List[str]]:
    cell_w = max(width / GRID_COLS, 1)
    cell_h = max(height / GRID_ROWS, 1)
    density_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

    for person in people:
        col = min(int(person["x"] / cell_w), GRID_COLS - 1)
        row = min(int(person["y"] / cell_h), GRID_ROWS - 1)
        density_grid[row, col] += 1.0

    heatmap = generate_heatmap(people, width, height)
    hotspots = _extract_hotspots(heatmap)
    flow = flow_analyzer.analyze(people)

    features = {
        "crowd_size": len(people),
        "avg_density": round(float(np.mean(density_grid)), 3),
        "max_density": round(float(np.max(density_grid)), 3),
        "avg_velocity": flow["avg_velocity"],
        "direction_conflict": flow["direction_conflict"],
        "turbulence": flow["turbulence"],
        "average_direction": flow["average_direction"],
        "opposing_movement_clusters": flow["opposing_movement_clusters"],
        "sudden_spike": flow["sudden_spike"],
        "avg_speed_delta": flow["avg_speed_delta"],
    }
    return features, heatmap, hotspots


def _extract_hotspots(heatmap: List[List[float]]) -> List[str]:
    hotspots: List[str] = []
    matrix = np.array(heatmap)
    for row, col in np.argwhere(matrix >= HOTSPOT_THRESHOLD):
        hotspots.append(f"zone_r{int(row)}_c{int(col)}")
    return hotspots[:4]


def _get_people(mode: str, video_path: str | None) -> Dict[str, object]:
    if mode == "video" and video_path:
        video_data = cv_processor.process_video_frame(video_path)
        if video_data["available"]:
            return video_data

    simulated = simulation_state.get_snapshot()
    return {
        "mode": simulated["mode"],
        "people": simulated["people"],
        "frame_size": simulated["frame_size"],
        "blocked_exits": simulated["blocked_exits"],
        "crowd_size": simulated["crowd_size"],
    }


@router.get("/predict")
def predict(mode: str = Query(default="simulation"), video_path: str | None = Query(default=None)) -> Dict[str, object]:
    crowd_data = _get_people(mode=mode, video_path=video_path)
    frame_size = crowd_data.get("frame_size", {"width": FRAME_WIDTH, "height": FRAME_HEIGHT})
    people = crowd_data.get("people", [])

    features, heatmap, hotspots = _extract_features(people, frame_size["width"], frame_size["height"])
    risk = risk_engine.compute_risk(features)
    recommendations = recommendation_engine.generate(risk, hotspots)

    return {
        "input_mode": crowd_data["mode"],
        "features": features,
        "risk": risk,
        "recommendations": recommendations,
        "hotspots": hotspots,
        "heatmap_preview": heatmap,
    }


@router.get("/heatmap")
def heatmap(mode: str = Query(default="simulation"), video_path: str | None = Query(default=None)) -> Dict[str, object]:
    crowd_data = _get_people(mode=mode, video_path=video_path)
    frame_size = crowd_data.get("frame_size", {"width": FRAME_WIDTH, "height": FRAME_HEIGHT})
    people = crowd_data.get("people", [])
    _, heatmap_matrix, hotspots = _extract_features(people, frame_size["width"], frame_size["height"])

    return {
        "input_mode": crowd_data["mode"],
        "frame_size": frame_size,
        "hotspots": hotspots,
        "heatmap": heatmap_matrix,
    }
