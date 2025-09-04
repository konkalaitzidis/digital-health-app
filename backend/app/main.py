# backend/app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

import numpy as np
import joblib
from pathlib import Path
import sys

# ---- Import shared feature code from ml/src ----
ML_SRC = Path(__file__).resolve().parents[2] / "ml" / "src"
sys.path.append(str(ML_SRC))
from features import extract_single_window  # single source of truth for inference features

app = FastAPI(title="ADAMMA Inference API", version="0.2.1")

# ---- CORS (open for local dev; tighten later if needed) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model artifacts on startup ----
ART = ML_SRC / "model.pkl"
if not ART.exists():
    raise RuntimeError(f"Model artifact not found: {ART}")
art = joblib.load(ART)

MODEL = art["model"]
SCALER = art["scaler"]
LABELS = art["label_encoder"]
FS = float(art.get("fs", 20.0))
WIN_SEC = float(art.get("win_sec", 5.0))
OVERLAP = float(art.get("overlap", 0.5))
N_FEATURES = int(art.get("n_features", SCALER.mean_.shape[0]))
CLASSES = list(art.get("classes", LABELS.classes_))
WIN = int(FS * WIN_SEC)

# ---- Schemas ----
class Sample(BaseModel):
    accel_x: float = Field(..., description="Acceleration X (g from Expo)")
    accel_y: float = Field(..., description="Acceleration Y (g from Expo)")
    accel_z: float = Field(..., description="Acceleration Z (g from Expo)")
    timestamp: Optional[float] = Field(None, description="Optional timestamp (s or ms)")

class PredictRequest(BaseModel):
    samples: List[Sample] = Field(..., description=f"One window of ~{WIN} samples (FS={FS}Hz, win={WIN_SEC}s)")

class PredictResponse(BaseModel):
    met_class: str
    proba: Optional[Dict[str, float]] = None

# ---- Helpers ----
def _calibrate_to_wisdm(seg: np.ndarray) -> np.ndarray:
    """
    Auto-detect input units:
      - If it already looks like WISDM (m/s^2; z baseline ~9, amplitudes > ~5),
        return as-is (maybe a tiny amplitude floor only).
      - If it looks like Expo iPhone data (g, centered around 0, small amplitudes),
        convert g→m/s^2, align z baseline, and (lightly) boost amplitude if too flat.
    """
    x = seg.astype(np.float64, copy=True)

    # Heuristics to detect WISDM-like data (already m/s^2):
    z_mean = float(x[:, 2].mean())
    abs95 = float(np.percentile(np.abs(x), 95))  # typical scale snapshot
    # WISDM-ish if z baseline ~9 OR overall magnitudes already > ~5 m/s^2
    looks_like_wisdm = (7.0 <= z_mean <= 12.0) or (abs95 > 5.0)

    if looks_like_wisdm:
        # Optional tiny floor to avoid "always Sedentary" on super-flat windows
        mag = np.linalg.norm(x, axis=1)
        if mag.std() < 0.25:
            scale = 0.75 / max(mag.std(), 1e-6)
            x *= np.clip(scale, 1.0, 2.0)
        return x

    # Otherwise treat it as Expo g-units → convert and align
    x *= 9.81  # g → m/s^2

    # Align z baseline toward ~9 if centered near 0
    z_mean = float(x[:, 2].mean())
    if -3.0 < z_mean < 3.0:
        x[:, 2] += 9.0

    # Light amplitude boost only if extremely flat
    mag = np.linalg.norm(x, axis=1)
    mag_std = float(mag.std())
    target_std = 1.0
    if mag_std < 0.3:
        scale = target_std / max(mag_std, 1e-6)
        x *= np.clip(scale, 1.0, 3.0)

    return x


# ---- Routes ----
@app.get("/ping")
def ping():
    return {"ok": True, "service": "adamma-api", "version": "0.2.1"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Parse samples -> ndarray [n,3]
    arr = np.array([[s.accel_x, s.accel_y, s.accel_z] for s in req.samples], dtype=np.float64)
    n = arr.shape[0]
    if n < WIN:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough samples ({n}). Need at least {WIN} (FS={FS}Hz * {WIN_SEC}s).",
        )

    # Use the last full window (allows rolling buffers from client)
    seg = arr[-WIN:, :]

    # Calibration shim (Expo -> WISDM-like)
    seg = _calibrate_to_wisdm(seg)

    # Features (must match training exactly)
    x = extract_single_window(seg)  # shape [N_FEATURES]
    if x.shape[0] != N_FEATURES:
        raise HTTPException(
            status_code=500,
            detail=f"Feature length mismatch: got {x.shape[0]} expected {N_FEATURES}",
        )

    # Scale & predict
    xs = SCALER.transform(x.reshape(1, -1))
    pred_idx = MODEL.predict(xs)[0]
    met = LABELS.inverse_transform([pred_idx])[0]

    proba = None
    if hasattr(MODEL, "predict_proba"):
        p = MODEL.predict_proba(xs)[0]
        proba = {CLASSES[i]: float(p[i]) for i in range(len(CLASSES))}

    return PredictResponse(met_class=met, proba=proba)
