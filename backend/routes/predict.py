from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, Query

from routes.simulation import simulation_state
from services.cv_processor import CVProcessor
from services.pipeline import run_pipeline
from utils.config import FRAME_HEIGHT, FRAME_WIDTH


router = APIRouter(tags=["prediction"])

cv_processor = CVProcessor()


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
async def predict(mode: str = Query(default="simulation"), video_path: str | None = Query(default=None)) -> Dict[str, object]:
    crowd_data = _get_people(mode=mode, video_path=video_path)
    frame_size = crowd_data.get("frame_size", {"width": FRAME_WIDTH, "height": FRAME_HEIGHT})
    people = crowd_data.get("people", [])

    result = await run_pipeline(people, frame_size["width"], frame_size["height"])

    return {
        "input_mode": crowd_data["mode"],
        **result,
    }


@router.get("/heatmap")
async def heatmap(mode: str = Query(default="simulation"), video_path: str | None = Query(default=None)) -> Dict[str, object]:
    crowd_data = _get_people(mode=mode, video_path=video_path)
    frame_size = crowd_data.get("frame_size", {"width": FRAME_WIDTH, "height": FRAME_HEIGHT})
    people = crowd_data.get("people", [])
    result = await run_pipeline(people, frame_size["width"], frame_size["height"])

    return {
        "input_mode": crowd_data["mode"],
        "frame_size": frame_size,
        "hotspots": result["hotspots"],
        "heatmap": result["heatmap"],
    }
