from __future__ import annotations

import asyncio
from typing import Dict, List, Tuple

import numpy as np

try:
    from CrowdControl_DB import crud
    from CrowdControl_DB.database import async_session_maker
    from services.cv_processor import CVProcessor
    from services.flow_analyzer import FlowAnalyzer
    from services.heatmap import generate_heatmap
    from services.recommendation_engine import RecommendationEngine
    from services.risk_engine import RiskEngine
    from utils.config import GRID_COLS, GRID_ROWS, HOTSPOT_THRESHOLD
except ImportError:
    from CrowdControl_DB import crud
    from CrowdControl_DB.database import async_session_maker
    from services.cv_processor import CVProcessor
    from services.flow_analyzer import FlowAnalyzer
    from services.heatmap import generate_heatmap
    from services.recommendation_engine import RecommendationEngine
    from services.risk_engine import RiskEngine
    from utils.config import GRID_COLS, GRID_ROWS, HOTSPOT_THRESHOLD

# ✅ INIT
cv_processor = CVProcessor()
_flow_analyzer = FlowAnalyzer()
_risk_engine = RiskEngine()
_recommendation_engine = RecommendationEngine()


def process_input(image=None, detections=None) -> dict:
    """Process an image or detection list and return crowd insights."""

    if image is not None:
        try:
            # ✅ FIX: use cv_processor instead of undefined function
            detection_result = cv_processor.detect_people(image)
        except Exception as e:
            print("Detection failed:", e)
            detection_result = {"detections": [], "people": []}

        raw_detections = detection_result.get("detections")

        if raw_detections:
            people = [
                {
                    "id": index + 1,
                    "x": float(d["x"] + d["w"] / 2.0),
                    "y": float(d["y"] + d["h"] / 2.0),
                    "vx": 0.0,
                    "vy": 0.0,
                }
                for index, d in enumerate(raw_detections)
            ]
        else:
            people = detection_result.get("people") or []
    else:
        people = detections or []

    features = _flow_analyzer.extract_features(people)

    risk = _risk_engine.compute_risk(
        {
            "max_density": float(features.get("max_density", 0.0)),
            "avg_velocity": float(features.get("avg_velocity", 0.0)),
            "direction_conflict": float(features.get("direction_conflict", 0.0)),
            "turbulence": float(features.get("turbulence", 0.0)),
        }
    )

    recommendations = _recommendation_engine.get_recommendations(risk)

    return {
        "people_count": int(len(people)),
        "features": features,
        "risk": risk,
        "recommendations": recommendations,
    }


def _hotspot_cells(heatmap: List[List[float]]) -> List[tuple[int, int, float]]:
    matrix = np.array(heatmap, dtype=np.float32)
    cells: list[tuple[int, int, float]] = []
    for row, col in np.argwhere(matrix >= HOTSPOT_THRESHOLD):
        cells.append((int(row), int(col), float(matrix[row, col])))
    cells.sort(key=lambda t: t[2], reverse=True)
    return cells[:6]


def _hotspot_points(cells: list[tuple[int, int, float]], width: int, height: int) -> list[dict]:
    cell_w = max(width / GRID_COLS, 1)
    cell_h = max(height / GRID_ROWS, 1)
    points: list[dict] = []
    for row, col, intensity in cells:
        points.append(
            {
                "x": int((col + 0.5) * cell_w),
                "y": int((row + 0.5) * cell_h),
                "intensity": round(float(intensity), 3),
                "cell": {"r": row, "c": col},
            }
        )
    return points


def _zones_from_heatmap(heatmap: List[List[float]]) -> list[dict]:
    zones: list[dict] = []
    matrix = np.array(heatmap, dtype=np.float32)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            val = float(matrix[row, col])
            if val >= 0.8:
                status = "CRITICAL"
            elif val >= 0.5:
                status = "WARNING"
            else:
                status = "SAFE"
            zones.append({"zone_id": row * GRID_COLS + col + 1, "status": status, "value": round(val, 3)})
    return zones


def _density_metrics(people: List[dict], width: int, height: int) -> Tuple[float, float]:
    if not people:
        return 0.0, 0.0

    cell_w = max(width / GRID_COLS, 1)
    cell_h = max(height / GRID_ROWS, 1)
    density_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

    for person in people:
        col = min(int(person["x"] / cell_w), GRID_COLS - 1)
        row = min(int(person["y"] / cell_h), GRID_ROWS - 1)
        density_grid[row, col] += 1.0

    return float(np.mean(density_grid)), float(np.max(density_grid))


async def _persist_async(risk: dict, features: dict, alert_message: str | None) -> None:
    if async_session_maker is None:
        return
    try:
        async with async_session_maker() as db:
            await crud.create_risk_log(
                db,
                {
                    "risk_score": float(risk["risk_score"]),
                    "status": str(risk["status"]),
                    "density": float(features["max_density"]),
                    "velocity": float(features["avg_velocity"]),
                    "turbulence": float(features["turbulence"]),
                    "direction_conflict": float(features["direction_conflict"]),
                },
            )
            if alert_message:
                await crud.create_alert(
                    db,
                    {
                        "risk_score": float(risk["risk_score"]),
                        "message": alert_message,
                        "severity": str(risk["status"]),
                    },
                )
    except Exception:
        return
    
async def run_pipeline(people: List[dict], width: int, height: int) -> Dict[str, object]:
    """
    Central processing pipeline
    """

    avg_density, max_density = _density_metrics(people, width, height)
    flow = _flow_analyzer.analyze(people)

    heatmap = generate_heatmap(people, width, height)
    hotspot_cells = _hotspot_cells(heatmap)
    hotspot_points = _hotspot_points(hotspot_cells, width, height)
    hotspot_cell_ids = [f"zone_r{r}_c{c}" for r, c, _ in hotspot_cells]

    zones = _zones_from_heatmap(heatmap)

    features: Dict[str, object] = {
        "crowd_size": len(people),
        "avg_density": round(avg_density, 3),
        "max_density": round(max_density, 3),
        "avg_velocity": flow["avg_velocity"],
        "direction_conflict": flow["direction_conflict"],
        "turbulence": flow["turbulence"],
        "average_direction": flow["average_direction"],
        "opposing_movement_clusters": flow["opposing_movement_clusters"],
        "sudden_spike": flow["sudden_spike"],
        "avg_speed_delta": flow["avg_speed_delta"],
    }

    risk = _risk_engine.compute_risk(
        {
            "max_density": float(features["max_density"]),
            "avg_velocity": float(features["avg_velocity"]),
            "direction_conflict": float(features["direction_conflict"]),
            "turbulence": float(features["turbulence"]),
        }
    )

    recommendations = _recommendation_engine.generate(risk, hotspot_cell_ids)

    alert_message = None
    if risk["status"] == "CRITICAL":
        alert_message = "; ".join(recommendations)[:500]

    asyncio.create_task(_persist_async(risk, features, alert_message))

    return {
        "features": features,
        "risk": risk,
        "heatmap": heatmap,
        "hotspot_cells": hotspot_cell_ids,
        "hotspots": hotspot_points,
        "zones": zones,
        "recommendations": recommendations,
        "risk_history": _risk_engine.get_recent_scores(10),
    }