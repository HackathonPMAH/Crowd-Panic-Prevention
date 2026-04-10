from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from CrowdControl_DB import crud
from CrowdControl_DB.database import get_db


router = APIRouter(tags=["history"])


@router.get("/history")
async def history(limit: int = Query(default=50, ge=1, le=200), db: AsyncSession = Depends(get_db)) -> list[dict]:
    try:
        rows = await crud.get_recent_risk_logs(db, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc.__class__.__name__}") from exc
    return [
        {
            "id": row.id,
            "timestamp": row.timestamp.isoformat(),
            "risk_score": row.risk_score,
            "status": row.status,
            "density": row.density,
            "velocity": row.velocity,
            "turbulence": row.turbulence,
            "direction_conflict": row.direction_conflict,
        }
        for row in rows
    ]


@router.get("/alerts")
async def alerts(limit: int = Query(default=50, ge=1, le=200), db: AsyncSession = Depends(get_db)) -> list[dict]:
    try:
        rows = await crud.get_recent_alerts(db, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc.__class__.__name__}") from exc
    return [
        {
            "id": row.id,
            "timestamp": row.timestamp.isoformat(),
            "risk_score": row.risk_score,
            "message": row.message,
            "severity": row.severity,
        }
        for row in rows
    ]

