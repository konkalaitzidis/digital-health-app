# Activity-to-MET Classifier

Prototype system for classifying smartphone accelerometer signals into 4 MET categories: **Sedentary, Light, Moderate, Vigorous**.  

---

## 📊 What’s done so far
- **Dataset**: WISDM v1.1 accelerometer dataset (downloaded locally from [link](https://www.cis.fordham.edu/wisdm/dataset.php)).  
- **ML Pipeline** (`ml/src/train.py`):
  - Cleans raw WISDM data.
  - Maps activities to MET classes.
  - Windowing (5s, 50% overlap).
  - Basic statistical features (mean, std, min, max, etc.).
  - Trains a RandomForest baseline model.
  - Achieves ~99% accuracy / F1. *raw data came from a controlled collection process - check `ml/data/raw/readme.txt` for more information.*
  - Saves model in `ml/src/model.pkl`.
- **Frontend App** (`frontend/adamma-frontend/App.js`):
  - Expo React Native skeleton.
  - Shows current MET class (dummy value).
  - Timers for cumulative time per class (increments every second).

---

## ▶️ How to run

### ML pipeline
```bash
cd ml/src
python train.py

### Frontend Expo App
cd frontend/adamma-frontend
npm install
npm start
