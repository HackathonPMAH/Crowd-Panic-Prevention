"""
Pre-trained Crowd Panic Risk Model
===================================
1. Drop risk_model.pkl, scaler.pkl, and this file into your backend/models/ folder.
2. Import and call predict_risk_class() anywhere in your backend.

pip install scikit-learn joblib numpy
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib, os

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "risk_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
LABEL_MAP   = {0: "SAFE", 1: "WARNING", 2: "CRITICAL"}

def _load_or_train():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("[RiskModel] Loaded from disk.")
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    print("[RiskModel] PKL not found — training from scratch …")
    return _train()

def _build_data():
    np.random.seed(42)
    X, y = [], []
    for _ in range(500):
        X.append([np.random.uniform(0,0.35), np.random.uniform(0,0.30),
                  np.random.uniform(0,0.25), np.random.uniform(0,0.25)]); y.append(0)
    for _ in range(500):
        X.append([np.random.uniform(0.25,0.65), np.random.uniform(0.20,0.60),
                  np.random.uniform(0.20,0.65), np.random.uniform(0.20,0.60)]); y.append(1)
    for _ in range(500):
        X.append([np.random.uniform(0.55,1.0)]*4); y.append(2)
    for _ in range(150):
        X.append([np.random.uniform(0.85,1.0)]*4); y.append(2)
    for _ in range(150):
        X.append([np.random.uniform(0,0.08)]*4); y.append(0)
    for _ in range(200):
        X.append([np.random.uniform(0.30,0.45)]*4); y.append(1)
    return np.array(X), np.array(y)

def _train():
    X, y   = _build_data()
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    model  = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                 learning_rate=0.07, subsample=0.85, random_state=42)
    model.fit(Xs, y)
    joblib.dump(model, MODEL_PATH); joblib.dump(scaler, SCALER_PATH)
    print(f"[RiskModel] Trained & saved. Accuracy: {np.mean(model.predict(Xs)==y):.4f}")
    return model, scaler

_model, _scaler = _load_or_train()


# ── Public API ────────────────────────────────────────────────────────────────

def predict_risk_class(features: dict) -> dict:
    """
    Args:
        features = {
            "density_score":      float 0-1,  # people per cell, normalised
            "velocity_score":     float 0-1,  # avg crowd speed, normalised
            "direction_conflict": float 0-1,  # 0=aligned, 1=opposing chaos
            "turbulence":         float 0-1   # variance in individual speeds
        }
    Returns:
        { "class": int, "label": str, "probabilities": {SAFE, WARNING, CRITICAL} }
    """
    X = np.array([[
        features["density_score"],
        features["velocity_score"],
        features["direction_conflict"],
        features["turbulence"],
    ]])
    Xs    = _scaler.transform(X)
    cls   = int(_model.predict(Xs)[0])
    proba = _model.predict_proba(Xs)[0]
    return {
        "class": cls,
        "label": LABEL_MAP[cls],
        "probabilities": {
            "SAFE":     round(float(proba[0]), 4),
            "WARNING":  round(float(proba[1]), 4),
            "CRITICAL": round(float(proba[2]), 4),
        }
    }


def batch_predict(feature_list: list) -> list:
    """Predict for a list of feature dicts — useful for processing video frames."""
    return [predict_risk_class(f) for f in feature_list]


def retrain(extra_X=None, extra_y=None):
    """
    Retrain with optional real-world data you collected.
    extra_X: np.ndarray shape (N, 4)
    extra_y: np.ndarray shape (N,)  — labels 0/1/2
    """
    global _model, _scaler
    X, y = _build_data()
    if extra_X is not None:
        X = np.vstack([X, extra_X])
        y = np.concatenate([y, extra_y])
    _model, _scaler = _train()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        {"density_score": 0.1, "velocity_score": 0.1, "direction_conflict": 0.05, "turbulence": 0.05},
        {"density_score": 0.5, "velocity_score": 0.4, "direction_conflict": 0.45, "turbulence": 0.40},
        {"density_score": 0.9, "velocity_score": 0.85,"direction_conflict": 0.90, "turbulence": 0.88},
    ]
    for t in tests:
        r = predict_risk_class(t)
        print(f"{r['label']:8s}  proba={r['probabilities']}")
