"""AI Drowsiness Detection: user-trained Random Forest + multi-face temporal smoothing."""
import os
import threading
import time
import cv2
import joblib
import numpy as np
import streamlit as st
from features import FaceAnalyzer, TemporalTracker, calibrate_threshold, FEATURE_NAMES

st.set_page_config(page_title="AI Drowsiness Detection", page_icon="😴", layout="wide")
st.title("😴 AI Drowsiness Detection")


@st.cache_resource
def load_model_bundle():
    if not os.path.exists("model.pkl"):
        return None, "model.pkl not found"
    try:
        obj = joblib.load("model.pkl")
        if isinstance(obj, dict) and "model" in obj:
            model = obj["model"]
            threshold = float(obj.get("threshold", 0.5))
            features = obj.get("features", FEATURE_NAMES)
        else:
            model = obj
            threshold = 0.5
            features = FEATURE_NAMES
        if list(features) != list(FEATURE_NAMES):
            return None, "model.pkl was trained with a different feature set. Run train.bat again."
        if getattr(model, "n_features_in_", len(FEATURE_NAMES)) != len(FEATURE_NAMES):
            return None, "model.pkl has an old feature count. Run train.bat again."
        return (model, threshold), ""
    except Exception as e:
        return None, f"Could not load model.pkl: {e}"


def open_camera(index):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, None):
        cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
        cap.release()
    return None


def play_alarm():
    import winsound
    for _ in range(2):
        winsound.Beep(1200, 250)
        time.sleep(0.05)


bundle, model_error = load_model_bundle()
with st.sidebar:
    st.header("AI Settings")
    camera_index = st.number_input("Camera index", 0, 5, 0)
    smoothing = st.slider("AI prediction window", 5, 25, 12)
    alarm_delay = st.slider("Alarm after continuous AI drowsiness (s)", 0.5, 5.0, 2.0, 0.5)
    if bundle:
        model, learned_threshold = bundle
        st.success("AI mode: Random Forest")
        st.caption(f"Learned probability threshold: {learned_threshold:.2f}")
        st.caption("Final state is produced by the trained model and temporal smoothing. No fixed EAR/MAR drowsiness rule is used.")
    else:
        model = None
        st.error(model_error)

run = st.toggle("Start camera", value=False)
video = st.empty(); banner = st.empty(); metrics = st.empty()

if run:
    if model is None:
        st.stop()
    cap = open_camera(int(camera_index))
    if cap is None:
        st.error("Camera could not be opened. Close Camera/Zoom/Teams and check Windows camera permissions.")
        st.stop()

    analyzer = FaceAnalyzer(max_num_faces=5)
    known = {}       # person id -> center, threshold, baseline
    trackers = {}
    probs = {}
    last_seen = {}
    next_id = 0
    alarm_started = {}
    alarm_last = 0.0

    # Per-face 5-second calibration.
    calibration = {}
    end = time.time() + 5
    while time.time() < end:
        ok, frame = cap.read()
        if not ok:
            break
        faces = analyzer.process_all(frame)
        for i, face in enumerate(faces):
            calibration.setdefault(i, []).append(face["ear"])
        left = max(0.0, end - time.time())
        cv2.putText(frame, f"AI calibration - eyes OPEN - {left:.1f}s", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 200, 255), 2)
        video.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    for i, ears in calibration.items():
        baseline = float(np.median(ears)) if ears else 0.21
        known[i] = {"center": None, "thr": calibrate_threshold(ears), "baseline": baseline}
    next_id = len(known)

    while run:
        ok, frame = cap.read()
        if not ok:
            banner.error("Camera frame read failed.")
            break
        t = time.time()
        faces = analyzer.process_all(frame)
        current = []

        for face in faces:
            center = np.array(face["center"], dtype=float)
            best_id, best_d = None, 1e9
            for fid, info in known.items():
                if info["center"] is None:
                    continue
                d = float(np.linalg.norm(center - info["center"]))
                if d < best_d and d < 0.20:
                    best_id, best_d = fid, d
            if best_id is None:
                best_id = next_id
                next_id += 1
                known[best_id] = {
                    "center": center,
                    "thr": 0.72 * max(face["ear"], 0.2),
                    "baseline": max(face["ear"], 0.2),
                }
            known[best_id]["center"] = center
            last_seen[best_id] = t
            info = known[best_id]
            if best_id not in trackers:
                trackers[best_id] = TemporalTracker(ear_thr=info["thr"], baseline_ear=info["baseline"])
                probs[best_id] = []
                alarm_started[best_id] = None

            feats = trackers[best_id].update(t, face["ear"], face["mar"], face["pitch"], face["yaw"], face["roll"])
            proba = model.predict_proba([feats])[0]
            classes = list(model.classes_)
            d_idx = classes.index(1)
            dprob = float(proba[d_idx])

            # Smooth the model probability itself. The classification threshold was learned during training.
            probs[best_id].append(dprob)
            probs[best_id] = probs[best_id][-int(smoothing):]
            smooth_prob = float(np.mean(probs[best_id]))
            ai_state = int(smooth_prob >= learned_threshold)

            if ai_state:
                if alarm_started[best_id] is None:
                    alarm_started[best_id] = t
            else:
                alarm_started[best_id] = None

            current.append((best_id, ai_state, dprob, smooth_prob, feats, face["box"]))
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in face["box"]]
            color = (0, 0, 255) if ai_state else (0, 200, 0)
            cv2.rectangle(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), color, 2)
            cv2.putText(frame, f"Person {best_id+1}: {'DROWSY' if ai_state else 'NORMAL'} {smooth_prob:.0%}",
                        (max(5, int(x1*w)), max(30, int(y1*h)-10)), cv2.FONT_HERSHEY_SIMPLEX, .58, color, 2)

        for fid in list(last_seen):
            if t - last_seen[fid] > 1.5:
                last_seen.pop(fid, None)
                trackers.pop(fid, None)
                probs.pop(fid, None)
                alarm_started.pop(fid, None)
                known.pop(fid, None)

        video.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        if current:
            drowsy_people = [x for x in current if x[1]]
            alarm_ready = any(
                alarm_started.get(x[0]) is not None and t - alarm_started[x[0]] >= alarm_delay
                for x in drowsy_people
            )
            if alarm_ready and t - alarm_last > alarm_delay:
                banner.error("⚠️ AI: DROWSINESS DETECTED")
                threading.Thread(target=play_alarm, daemon=True).start()
                alarm_last = t
            elif drowsy_people:
                banner.warning("AI detects increasing drowsiness probability...")
            else:
                banner.success(f"AI monitoring {len(current)} face(s)")

            rows = []
            for fid, state, raw_p, smooth_p, f, _ in current:
                rows.append(
                    f"**Person {fid+1}: {'DROWSY' if state else 'NORMAL'} — AI probability {smooth_p:.0%}**  \n"
                    f"EAR {f[0]:.2f} | MAR {f[1]:.2f} | Closure {f[3]:.1f}s | "
                    f"Blinks/min {f[4]:.0f} | PERCLOS {f[5]:.0%} | Yawns/min {f[6]:.1f} | "
                    f"Mouth-open {f[7]:.0%} | Pitch {f[8]:.0f}° | Yaw {f[9]:.0f}° | Roll {f[10]:.0f}°"
                )
            metrics.markdown("\n\n".join(rows))
        else:
            banner.warning("No face detected")

    cap.release()
