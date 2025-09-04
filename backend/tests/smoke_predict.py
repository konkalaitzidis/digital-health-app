# backend/tests/smoke_from_wisdm.py
import pandas as pd, requests
from pathlib import Path

RAW = Path("../../ml/data/raw/WISDM_clean.txt")  # adjust if needed
API_URL = "http://127.0.0.1:8000/predict"
FS, WIN_SEC = 20, 5
N = FS * WIN_SEC

df = pd.read_csv(RAW, header=None,
                 names=["user","activity","timestamp","accel_x","accel_y","accel_z"],
                 sep=",", engine="python")
for c in ["accel_x","accel_y","accel_z"]:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(";", "", regex=False).str.strip(), errors="coerce")
df = df.dropna(subset=["accel_x","accel_y","accel_z"]).reset_index(drop=True)

# pick a window of a specific activity from raw labels (e.g., Walking)
win = df[df["activity"].str.strip() == "Walking"].iloc[:N]
payload = {"samples": win[["accel_x","accel_y","accel_z"]].to_dict(orient="records")}
r = requests.post(API_URL, json=payload, timeout=10)
print("Walking window →", r.status_code, r.json())
