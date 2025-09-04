# Activity-to-MET Classifier

Prototype system that classifies smartphone accelerometer signals into 4 MET categories: **Sedentary, Light, Moderate, Vigorous**.  

---

## 📊 Progress
- **Dataset**: WISDM v1.1 accelerometer dataset ([link](https://www.cis.fordham.edu/wisdm/dataset.php)).  
- **ML Pipeline** (`ml/src/train.py`):
  - Cleans WISDM data, maps activities → MET classes.
  - 5s windows (50% overlap), statistical + magnitude features.
  - RandomForest model (~99% accuracy), saved as `ml/src/model.pkl` (Raw data came from a controlled collection process - check ml/data/raw/readme.txt for more information).
- **Backend** (`backend/app/main.py`):
  - FastAPI with `/ping` and `/predict`.
  - Calibration shim (Expo g-units → WISDM m/s²).
  - Returns class + probabilities.
- **Frontend App** (`frontend/adamma-frontend/App.js`):
  - Expo React Native with iPhone accelerometer (~20 Hz).
  - Sends 5s windows to backend, smooths predictions (majority of last 3).
  - Displays current class (color-coded) + per-class timers.
  - Session summary: Total, Active, MVPA, Active%, MVPA%.
  - Reset button + footer attribution.

---

## ▶️ How to run
```bash
# Train model
cd ml/src
python train.py

# Run backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run frontend
cd frontend/adamma-frontend
npm install
npm start

In App.js, set API_URL to your LAN IP (e.g. http://192.168.x.x:8000/predict).
Open with Expo Go on iOS/Android.

in progress...
