from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SIMULATED_DATA_PATH = DATA_DIR / "simulated_data.json"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
GRID_ROWS = 12
GRID_COLS = 16
WEBSOCKET_INTERVAL_SECONDS = 1.0

RISK_WEIGHTS = {
    "density": 0.4,
    "velocity": 0.2,
    "direction_conflict": 0.2,
    "turbulence": 0.2,
}

SAFE_THRESHOLD = 40.0
WARNING_THRESHOLD = 70.0
CRITICAL_THRESHOLD = 90.0

MAX_REFERENCE_SPEED = 6.0
HOTSPOT_THRESHOLD = 0.65

EXIT_ZONES = {
    "north_exit": {"x1": 250, "y1": 0, "x2": 390, "y2": 40},
    "south_exit": {"x1": 250, "y1": 440, "x2": 390, "y2": 480},
    "east_exit": {"x1": 600, "y1": 170, "x2": 640, "y2": 310},
    "west_exit": {"x1": 0, "y1": 170, "x2": 40, "y2": 310},
}
