from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from CrowdControl_DB.database import engine
from CrowdControl_DB.models import Base
from routes.history import router as history_router
from routes.predict import router as predict_router
from routes.simulation import router as simulation_router
from routes.simulation import simulation_state
from services.pipeline import run_pipeline
from utils.config import FRAME_HEIGHT, FRAME_WIDTH, WEBSOCKET_INTERVAL_SECONDS


app = FastAPI(
    title="Crowd Panic Prediction and Prevention Platform",
    description="Real-time crowd risk prediction backend with simulation, heatmaps, and recommendations.",
    version="1.0.0",
)

app.include_router(simulation_router)
app.include_router(predict_router)
app.include_router(history_router)


@app.on_event("startup")
async def startup() -> None:
    if engine is None:
        return
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception:
        # Keep realtime endpoints alive even if DB is temporarily unavailable.
        return


@app.get("/")
def root() -> dict:
    return {
        "service": app.title,
        "status": "running",
        "endpoints": ["/simulate", "/predict", "/heatmap", "/ws/live"],
    }


@app.websocket("/ws/live")
async def live_updates(
    websocket: WebSocket,
    crowd_size: int | None = Query(default=None, ge=5, le=300),
    increase_by: int = Query(default=0, ge=0, le=100),
    block_exits: str | None = Query(default=None),
) -> None:
    await websocket.accept()
    simulation_state.configure(crowd_size=crowd_size, increase_by=increase_by, block_exits=block_exits)

    try:
        while True:
            snapshot = simulation_state.get_snapshot()
            frame_size = snapshot.get("frame_size", {"width": FRAME_WIDTH, "height": FRAME_HEIGHT})
            people = snapshot.get("people", [])

            result = await run_pipeline(people, frame_size["width"], frame_size["height"])
            risk = result["risk"]
            print(f"[ws/live] risk={risk['risk_score']} status={risk['status']}")

            await websocket.send_json(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "crowd": {
                        "size": len(people),
                        "blocked_exits": snapshot.get("blocked_exits", []),
                    },
                    "risk_score": risk["risk_score"],
                    "status": risk["status"],
                    "time_to_disaster": risk["time_to_disaster"],
                    "risk_history": result.get("risk_history", []),
                    **result,
                }
            )
            await asyncio.sleep(WEBSOCKET_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        return
