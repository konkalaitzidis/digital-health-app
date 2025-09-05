# ADAMMA - Live MET Tracker

Mobile app that classifies smartphone accelerometer signals into **4 MET categories**:  
**Sedentary, Light, Moderate, Vigorous** — in near real time.  

---

## 📊 Project Overview
- **Dataset:** WISDM v1.1 accelerometer dataset ([link](https://www.cis.fordham.edu/wisdm/dataset.php)).  
- **ML Pipeline** (`ml/src/train.py`):  
  - Cleans WISDM data, maps activities to MET classes.  
  - 5 s windows (50% overlap), statistical + magnitude features.  
  - Random Forest (~98–99% accuracy) chosen as best model.  
  - Model artifact saved at `ml/src/model.pkl`.  
- **Backend** (`backend/app/main.py`):  
  - FastAPI with `/ping` and `/predict`.  
  - Calibrates Expo accelerometer data (g-units → m/s²).  
  - Returns predicted MET class + probabilities.  
- **Frontend App** (`frontend/adamma-frontend/App.js`):  
  - Expo React Native app streaming accelerometer (~20 Hz).  
  - Buffers 5 s windows, sends to backend, applies 3-prediction majority smoothing.  
  - Displays current class (color-coded) + per-class timers.  
  - Session summary: **Total, Active, MVPA, Active%, MVPA%**.  
  - Reset button + settings button for user to input custom URL. 

---

## 📱 How to Use

You will need:  
- An **Android phone** (for APK installation).  
- A **computer** (macOS/Linux/Windows, Python 3.9+) to run the backend.  
- Both on the **same Wi-Fi network**.  

### 1. Run the backend - open a terminal on an IDE
```bash
git clone https://github.com/konkalaitzidis/digital-health-app.git
cd digital-health-app

python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)

pip install -r requirements.txt

cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Find your computer’s IP - open new terminal window

**macOS/Linux:**
```bash
ipconfig getifaddr en0  
```

**Windows:**
```cmd
ipconfig
```

Check from your phone’s browser:
```
http://<YOUR-IP>:8000/ping
```

Should return:
```json
{"status":"ok"}
```

### 3. Install the Android APK
APK is provided here:
```
docs/releases/ADAMMA-v1-android.apk
```

Or open this link to download and install: ([link](https://expo.dev/accounts/guskk/projects/adamma-frontend/builds/4621f0f3-52b8-42be-ad6b-ccb40418fac3))

Or transfer to your phone and tap to install.  
Or via ADB:
```bash
adb install -r docs/releases/ADAMMA-v1-android.apk
```

### 4. Run the app
- Open **ADAMMA** on your phone.  
- Tap ⚙ **Settings** → enter your computer’s IP (e.g. `http://192.168.1.45:8000`).  
- Save.  
- Move with your phone (sit → walk → jog).  
- Status shows: `Predicting → OK: <class>`.  
- Timers increment by class.  
- Session summary updates with Active% and MVPA%.  

---

## ⚙️ Developer Instructions

**Train the model**
```bash
cd ml/src
python train.py
```

Outputs:
- Model artifacts (`.pkl`) in `ml/src/models/`  
- Confusion matrices in `docs/figures/`  
- Reports in `ml/reports/`  

**Run backend**
```bash
cd backend/app
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Run frontend in dev mode**
```bash
cd frontend/adamma-frontend
npm install
npx expo start -c
```

Open in **Expo Go** (iOS/Android) or emulator.  

---

## 📑 Deliverables
- ✅ Android APK (`docs/releases/ADAMMA-v1-android.apk`)  
- ✅ Model + code (`ml/src/train.py`, `ml/src/model.pkl`, `features.py`)  
- ✅ Technical report (`technical_report.pdf`)  
- ✅ Demo video (`docs/demo.mp4`)  

---

## ⚠️ Limitations
- App doesn't work on background. Screen must be on. 
- iOS distribution not applicable as requires paid apple developer account
- WISDM dataset bias: controlled lab collection.

---

## 📜 License
MIT
