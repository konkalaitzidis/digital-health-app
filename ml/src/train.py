from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from features import extract_features

# -----------------
# Config
# -----------------
FS = 20.0             # Hz (WISDM ≈ 20Hz)
WIN_SEC = 5.0         # window length in seconds
OVERLAP = 0.5         # 50% overlap
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "WISDM_clean.txt"

# Reproducibility
RNG_SEED = 42
np.random.seed(RNG_SEED)

# Paths
SRC_DIR = Path(__file__).resolve().parent
ML_DIR = SRC_DIR.parent
REPO_ROOT = ML_DIR.parent
REPORTS_DIR = ML_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = REPO_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = SRC_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
    "Standing": "Sedentary",   # fixed to Sedentary per project mapping
    "Walking": "Light",
    "Upstairs": "Moderate",
    "Downstairs": "Moderate",
    "Jogging": "Vigorous",
}
df = df[df["activity"].isin(MET_MAP.keys())].copy()
df["met_class"] = df["activity"].map(MET_MAP)

# -----------------
# Windowing + feature extraction
# -----------------
X, y = extract_features(df, fs=FS, win_sec=WIN_SEC, overlap=OVERLAP, label_col="met_class")

# -----------------
# Split, scale
# -----------------
le = LabelEncoder()
y_enc = le.fit_transform(y)

Xtr, Xte, ytr, yte = train_test_split(
    X, y_enc, test_size=0.2, random_state=RNG_SEED, stratify=y_enc
)

scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

# -----------------
# Models to compare
# -----------------
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=RNG_SEED, n_jobs=-1
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=2000, multi_class="multinomial", solver="lbfgs", class_weight="balanced", random_state=RNG_SEED
    ),
    "SVM_RBF": SVC(
        kernel="rbf", probability=True, gamma="scale", C=3.0,
        class_weight="balanced", random_state=RNG_SEED
    ),
    "MLP_64x32": MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=1e-4, max_iter=400, random_state=RNG_SEED, early_stopping=True,
        n_iter_no_change=10, validation_fraction=0.1
    ),
}

# -----------------
# Train + evaluate helper
# -----------------
def evaluate_model(name, model, Xtr, ytr, Xte, yte, labels_text):
    t0 = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t0

    ypred = model.predict(Xte)
    acc = accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro")

    # Confusion matrix (row-normalized)
    cm = confusion_matrix(yte, ypred, labels=np.arange(len(labels_text)), normalize="true")

    # Save confusion matrix figure
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels_text, yticklabels=labels_text)
    plt.title(f"Confusion Matrix (Normalized) — {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    fig_path = FIG_DIR / f"confusion_matrix_{name}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # Classification report to file
    clsrep_txt = classification_report(yte, ypred, target_names=labels_text)
    (REPORTS_DIR / f"classification_report_{name}.txt").write_text(clsrep_txt)

    # Save per-model artifact (for inspection / optional backend swap)
    art_path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(
        {"model": model, "scaler": scaler, "label_encoder": le, "fs": FS, "win_sec": WIN_SEC, "overlap": OVERLAP},
        art_path
    )

    return {
        "Model": name,
        "Accuracy": acc,
        "F1_macro": f1m,
        "TrainTime_sec": train_time,
        "confmat_path": str(fig_path),
        "artifact_path": str(art_path),
    }

# -----------------
# Run comparisons
# -----------------
results = []
for name, mdl in models.items():
    print(f"\n=== Training {name} ===")
    res = evaluate_model(name, mdl, Xtr_s, ytr, Xte_s, yte, labels_text=le.classes_)
    results.append(res)
    print(f"{name} — Acc: {res['Accuracy']:.3f} | F1(macro): {res['F1_macro']:.3f} | Train: {res['TrainTime_sec']:.2f}s")
    print(f"Saved: {res['artifact_path']}")
    print(f"Confusion matrix → {res['confmat_path']}")
    print(f"Classification report → {REPORTS_DIR / f'classification_report_{name}.txt'}")

# -----------------
# Results table (print + save)
# -----------------
res_df = pd.DataFrame(results).sort_values(by="F1_macro", ascending=False).reset_index(drop=True)
print("\n\n=== Results (sorted by F1_macro) ===")
print(res_df[["Model", "Accuracy", "F1_macro", "TrainTime_sec"]].to_string(index=False))

# Save CSV + TXT table
res_csv = REPORTS_DIR / "model_results.csv"
res_txt = REPORTS_DIR / "model_results.txt"
res_df.to_csv(res_csv, index=False)
res_txt.write_text(res_df[["Model", "Accuracy", "F1_macro", "TrainTime_sec"]].to_string(index=False))

print(f"\nResults saved to:\n- {res_csv}\n- {res_txt}")

# -----------------
# Save BEST model to standard path for backend
# -----------------
best_row = res_df.iloc[0]
best_name = best_row["Model"]
best_artifact = best_row["artifact_path"]

# Load best artifact and re-dump to canonical path
best_payload = joblib.load(best_artifact)

ART = SRC_DIR / "model.pkl"
joblib.dump(best_payload, ART)
print(f"\nBest model: {best_name} (F1_macro={best_row['F1_macro']:.3f})")
print(f"Saved canonical artifact for backend to: {ART}")
