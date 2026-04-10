from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from routes.simulation import simulation_state
from services.cv_processor import CVProcessor
from services.pipeline import process_input, run_pipeline
from utils.config import FRAME_HEIGHT, FRAME_WIDTH

import cv2

import numpy as np


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


def _generate_mock_detections() -> List[dict]:
    simulation = simulation_state.get_snapshot()
    return simulation.get("people", [])


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


@router.post("/predict")
async def predict_image(image: UploadFile | None = File(default=None)) -> Dict[str, object]:
    if image is not None:
        if cv2 is None:
            raise HTTPException(status_code=503, detail="OpenCV is required to decode uploaded images")

        contents = await image.read()
        buffer = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Unable to decode uploaded image")

        result = process_input(image=frame)
        return {
            "input_mode": "image",
            "image_filename": image.filename,
            **result,
        }

    mock_detections = _generate_mock_detections()
    result = process_input(detections=mock_detections)
    return {
        "input_mode": "simulation",
        "mock_mode": True,
        **result,
    }


@router.post("/predict-image")
async def predict_image_endpoint(image: UploadFile = File(...)) -> Dict[str, object]:
    if cv2 is None:
        raise HTTPException(status_code=503, detail="OpenCV is required to decode uploaded images")

    contents = await image.read()
    buffer = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unable to decode uploaded image")

    result = process_input(image=frame)
    return result


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
