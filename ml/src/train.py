from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from features import extract_features
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------
# Config
# -----------------
FS = 20.0            # Hz (WISDM ≈ 20Hz)
WIN_SEC = 5.0        # window length in seconds
OVERLAP = 0.5        # 50% overlap
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "WISDM_clean.txt"

# -----------------
# Load + quick clean
# -----------------
# WISDM rows: user,activity,timestamp,accel_x,accel_y,accel_z;
df = pd.read_csv(
    RAW,
    header=None,
    names=["user", "activity", "timestamp", "accel_x", "accel_y", "accel_z"],
    sep=","
)

# Remove trailing semicolons in accel_z if present
if df["accel_z"].dtype == object:
    df["accel_z"] = df["accel_z"].astype(str).str.replace(";", "", regex=False).astype(float)

# -----------------
# Map activities → MET classes
# -----------------
MET_MAP = {
    "Sitting": "Sedentary",
    "Standing": "Light",
    "Walking": "Moderate",
    "Upstairs": "Moderate",
    "Downstairs": "Moderate",
    "Jogging": "Vigorous",
}
df = df[df["activity"].isin(MET_MAP.keys())].copy()
df["met_class"] = df["activity"].map(MET_MAP)

# -----------------
# Windowing + features extraction
# -----------------

X, y = extract_features(df, fs=FS, win_sec=WIN_SEC, overlap=OVERLAP, label_col="met_class")

# -----------------
# Split, scale, train
# -----------------
le = LabelEncoder()
y_enc = le.fit_transform(y)

Xtr, Xte, ytr, yte = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(Xtr_s, ytr)

pred = clf.predict(Xte_s)
acc = accuracy_score(yte, pred)
f1 = f1_score(yte, pred, average="macro")
print(f"Baseline RF — Acc: {acc:.3f} | F1(macro): {f1:.3f}")
print(classification_report(yte, pred, target_names=le.classes_))

# -----------------
# Confusion Matrix
# -----------------
cm = confusion_matrix(yte, pred, labels=range(len(le.classes_)), normalize="true")

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)

plt.title("Confusion Matrix (Normalized)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# -----------------
# Save artifacts
# -----------------
ART = Path(__file__).resolve().parent / "model.pkl"
joblib.dump(
    {"model": clf, "scaler": scaler, "label_encoder": le, "fs": FS, "win_sec": WIN_SEC, "overlap": OVERLAP},
    ART
)
print(f"Saved model to: {ART}")

# Save to docs/figures
FIG = Path(__file__).resolve().parents[1] / ".." / "docs" / "figures" / "confusion_matrix.png"
FIG.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(FIG, dpi=150)
plt.show()

print(f"Confusion matrix saved to {FIG}")
