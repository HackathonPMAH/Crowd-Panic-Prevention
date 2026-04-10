from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from CrowdControl_DB.models import AlertLog, RiskLog


async def create_risk_log(db: AsyncSession, data: Dict[str, Any]) -> RiskLog:
    row = RiskLog(**data)
    db.add(row)
    await db.commit()
    return row


async def create_alert(db: AsyncSession, data: Dict[str, Any]) -> AlertLog:
    row = AlertLog(**data)
    db.add(row)
    await db.commit()
    return row


async def get_recent_risk_logs(db: AsyncSession, limit: int = 50) -> List[RiskLog]:
    result = await db.execute(select(RiskLog).order_by(desc(RiskLog.timestamp)).limit(limit))
    return list(result.scalars().all())


async def get_recent_alerts(db: AsyncSession, limit: int = 50) -> List[AlertLog]:
    result = await db.execute(select(AlertLog).order_by(desc(AlertLog.timestamp)).limit(limit))
    return list(result.scalars().all())

