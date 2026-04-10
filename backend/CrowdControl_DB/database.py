from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncGenerator

from fastapi import HTTPException
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:CHANGE_ME@localhost:5432/CrowdControl_DB",
)

try:
    engine = create_async_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )
except Exception:
    # Allows the app to run without DB deps installed.
    engine = None

async_session_maker = (
    async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession) if engine is not None else None
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    if async_session_maker is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not configured. Install asyncpg and set DATABASE_URL.",
        )
    async with async_session_maker() as session:
        yield session

