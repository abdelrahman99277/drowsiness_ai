# AI Drowsiness Detection

This version is **ML-first**. The final drowsy/normal decision comes from a user-trained Random Forest. EAR/MAR, blink/yawn statistics, PERCLOS and head pose are feature inputs; they are not used as a direct final drowsiness rule.

## One-time setup

1. Make sure Python 3.12 is installed. Python 3.14 can remain installed and is not modified.
2. Double-click `START.bat` once. It creates/repairs `venv` and installs packages **only if the import check says they are missing**.
3. Do not manually run `pip` from system Python.

All BAT files call the same `setup_env.bat`, so `collect_*` and `train.bat` will not accidentally run with Python 3.14 or with a broken environment.

## Training

If your existing `data.csv` was created by the previous 10-feature version, delete only that file and recollect once.

Recommended:

- `collect_normal.bat` — 3 separate normal sessions
- `collect_drowsy.bat` — 3 separate drowsy sessions
- `train.bat`
- `START.bat`

`train.bat` uses grouped cross-validation by recording session, learns the probability threshold from out-of-fold predictions, prints Accuracy/F1/Confusion Matrix/Classification Report, then trains the final model on all data and saves `model.pkl`.

## Runtime

- Up to 5 faces are analyzed independently.
- Each person's temporal feature history is separate.
- The app smooths the **Random Forest probability**, not a hand-written eye/mouth rule.
- The alarm is separate from classification and only sounds after the AI remains drowsy for the configured duration.

## Run without START.bat

```powershell
cd D:\path\to\drowsiness
.\venv\Scripts\Activate.ps1
python -m streamlit run app.py --browser.gatherUsageStats false
```
"# drowsiness_ai" 
"# drowsiness_ai" 
"# drowsiness_ai" 
