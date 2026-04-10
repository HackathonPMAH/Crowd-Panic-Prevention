from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Dict, List

from fastapi import APIRouter, Query

from utils.config import EXIT_ZONES, FRAME_HEIGHT, FRAME_WIDTH, SIMULATED_DATA_PATH


router = APIRouter(tags=["simulation"])


class CrowdSimulation:
    def __init__(self, seed: int = 7) -> None:
        self.random = Random(seed)
        self.frame_width = FRAME_WIDTH
        self.frame_height = FRAME_HEIGHT
        self.crowd_size = 35
        self.blocked_exits: List[str] = []
        self.people = self._bootstrap_people(self.crowd_size)
        self._ensure_seed_file()

    def _ensure_seed_file(self) -> None:
        path = Path(SIMULATED_DATA_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps({"people": self.people}, indent=2), encoding="utf-8")

    def _bootstrap_people(self, count: int) -> List[dict]:
        people = []
        for index in range(1, count + 1):
            people.append(
                {
                    "id": index,
                    "x": self.random.randint(60, self.frame_width - 60),
                    "y": self.random.randint(60, self.frame_height - 60),
                    "vx": round(self.random.uniform(-1.8, 1.8), 3),
                    "vy": round(self.random.uniform(-1.8, 1.8), 3),
                }
            )
        return people

    def configure(self, crowd_size: int | None = None, increase_by: int = 0, block_exits: str | None = None) -> None:
        if crowd_size is not None and crowd_size > 0:
            self.crowd_size = crowd_size
        self.crowd_size = max(5, self.crowd_size + increase_by)

        if block_exits is not None:
            requested = [exit_name.strip() for exit_name in block_exits.split(",") if exit_name.strip()]
            self.blocked_exits = [exit_name for exit_name in requested if exit_name in EXIT_ZONES]

        if len(self.people) < self.crowd_size:
            self.people.extend(self._bootstrap_people(self.crowd_size - len(self.people)))
        elif len(self.people) > self.crowd_size:
            self.people = self.people[: self.crowd_size]

    def get_snapshot(self) -> Dict[str, object]:
        self._step()
        return {
            "mode": "simulation",
            "people": self.people,
            "blocked_exits": self.blocked_exits,
            "crowd_size": len(self.people),
            "frame_size": {"width": self.frame_width, "height": self.frame_height},
        }

    def _step(self) -> None:
        congestion_factor = min(len(self.people) / 80.0, 1.5)
        exit_pressure = 1.0 + 0.25 * len(self.blocked_exits)

        for person in self.people:
            jitter_x = self.random.uniform(-0.35, 0.35) * exit_pressure
            jitter_y = self.random.uniform(-0.35, 0.35) * exit_pressure

            person["vx"] = round(person["vx"] * 0.72 + jitter_x, 3)
            person["vy"] = round(person["vy"] * 0.72 + jitter_y, 3)

            if congestion_factor > 0.9:
                person["vx"] = round(person["vx"] + self.random.uniform(-0.25, 0.25), 3)
                person["vy"] = round(person["vy"] + self.random.uniform(-0.25, 0.25), 3)

            next_x = person["x"] + person["vx"] * (1.2 + congestion_factor)
            next_y = person["y"] + person["vy"] * (1.2 + congestion_factor)

            if next_x <= 15 or next_x >= self.frame_width - 15:
                person["vx"] = round(-person["vx"], 3)
            if next_y <= 15 or next_y >= self.frame_height - 15:
                person["vy"] = round(-person["vy"], 3)

            person["x"] = int(max(10, min(self.frame_width - 10, person["x"] + person["vx"] * 2.0)))
            person["y"] = int(max(10, min(self.frame_height - 10, person["y"] + person["vy"] * 2.0)))

        if self.blocked_exits:
            for person in self.people:
                for blocked_exit in self.blocked_exits:
                    zone = EXIT_ZONES[blocked_exit]
                    inside = zone["x1"] <= person["x"] <= zone["x2"] and zone["y1"] <= person["y"] <= zone["y2"]
                    if inside:
                        person["vx"] = round(-person["vx"] * 1.2, 3)
                        person["vy"] = round(-person["vy"] * 1.2, 3)


simulation_state = CrowdSimulation()


@router.get("/simulate")
def simulate(
    crowd_size: int | None = Query(default=None, ge=5, le=300),
    increase_by: int = Query(default=0, ge=0, le=100),
    block_exits: str | None = Query(default=None, description="Comma-separated exit names"),
) -> Dict[str, object]:
    simulation_state.configure(crowd_size=crowd_size, increase_by=increase_by, block_exits=block_exits)
    return simulation_state.get_snapshot()
