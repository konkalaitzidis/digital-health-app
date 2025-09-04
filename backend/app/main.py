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
    Adjust Expo accelerometer window to resemble WISDM characteristics.
    Steps:
      1) Convert g -> m/s^2 (×9.81)
      2) If z baseline is near 0, shift it toward ~9 (gravity DC offset)
      3) If overall motion is very low, scale up to a target std (heuristic)
    """
    out = seg.astype(np.float64).copy()

    # 1) units: g -> m/s^2
    out *= 9.81

    # 2) gravity alignment on z if centered around ~0
    z_mean = out[:, 2].mean()
    if -3.0 < z_mean < 3.0:
        out[:, 2] += 9.0

    # 3) amplitude boost if too flat
    mag = np.linalg.norm(out, axis=1)
    mag_std = mag.std()
    target_std = 2.0  # heuristic from WISDM walking
    if mag_std < 0.6:
        scale = target_std / max(mag_std, 1e-6)
        out *= np.clip(scale, 1.0, 6.0)

    return out

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
