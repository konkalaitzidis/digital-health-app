import numpy as np
import pandas as pd

# -----------------
# Windowing + features extraction
# -----------------
def window_indices(n: int, win: int, step: int):
    """
    Generator for sliding window start/end indices.
    n    = total number of samples
    win  = window length (samples)
    step = step size (samples)
    """
    for start in range(0, n - win + 1, step):
        yield start, start + win


def extract_features(
    frame: pd.DataFrame,
    fs: float,
    win_sec: float = 5.0,
    overlap: float = 0.5,
    label_col: str = "met_class"
):
    """
    Extract windowed statistical features from accelerometer DataFrame.
    Returns X (n_windows, n_features), y (labels).
    
    Features per window:
    - For each axis (x,y,z): mean, std, min, max, median, IQR
    - For magnitude vector: mean, std
    Total = 20 features per window.
    """
    win = int(win_sec * fs)
    step = int(win * (1 - overlap))
    X, y = [], []

    # Ensure numeric
    A = frame[["accel_x", "accel_y", "accel_z"]].to_numpy(dtype=np.float64)
    L = frame[label_col].to_numpy()

    for i, j in window_indices(len(frame), win, step):
        seg = A[i:j]   # [win, 3]
        labs = L[i:j]

        # majority label in window
        vals, cnts = np.unique(labs, return_counts=True)
        majority = vals[np.argmax(cnts)]

        feats = []
        for k in range(3):  # per-axis stats
            a = seg[:, k]
            feats += [
                float(a.mean()),
                float(a.std()),
                float(a.min()),
                float(a.max()),
                float(np.median(a)),
                float(np.percentile(a, 75) - np.percentile(a, 25)),  # IQR
            ]

        # magnitude stats
        mag = np.linalg.norm(seg, axis=1)
        feats += [float(mag.mean()), float(mag.std())]

        feats = np.nan_to_num(np.array(feats), nan=0.0, posinf=0.0, neginf=0.0)
        X.append(feats)
        y.append(majority)

    return np.vstack(X), np.array(y)


def extract_single_window(seg: np.ndarray) -> np.ndarray:
    """
    Compute features for a single window [win, 3].
    Used for inference (backend).
    Returns: 1D feature vector of length 20.
    """
    feats = []
    for k in range(3):
        a = seg[:, k]
        feats += [
            float(a.mean()),
            float(a.std()),
            float(a.min()),
            float(a.max()),
            float(np.median(a)),
            float(np.percentile(a, 75) - np.percentile(a, 25)),
        ]

    mag = np.linalg.norm(seg, axis=1)
    feats += [float(mag.mean()), float(mag.std())]

    feats = np.nan_to_num(np.array(feats), nan=0.0, posinf=0.0, neginf=0.0)
    return feats
