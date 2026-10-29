"""Collect labeled ML training data. Press Q to finish a session."""
import csv
import os
import sys
import time
import cv2
import numpy as np
from features import FEATURE_NAMES, FaceAnalyzer, TemporalTracker, calibrate_threshold

LABELS = {"normal": 0, "drowsy": 1}
CSV = "data.csv"
MAX_FACES = 5


def open_camera(index=0):
    # Try the common Windows capture backends. The first backend that can read a frame wins.
    candidates = [(index, cv2.CAP_DSHOW), (index, cv2.CAP_MSMF), (index, None)]
    for idx, backend in candidates:
        cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
        cap.release()
    return None


def existing_csv_is_compatible():
    if not os.path.exists(CSV):
        return True
    try:
        with open(CSV, "r", encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
        required = FEATURE_NAMES + ["label", "session"]
        return header == required
    except Exception:
        return False


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in LABELS:
        print(__doc__)
        return 2

    label_name = sys.argv[1]
    label = LABELS[label_name]

    if os.path.exists(CSV) and not existing_csv_is_compatible():
        print("[ERROR] data.csv belongs to an older project version.")
        print("Delete only data.csv, then collect the new data once.")
        return 1

    session = f"{label_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    new_file = not os.path.exists(CSV)
    cap = open_camera(0)
    if cap is None:
        print("[ERROR] Camera could not be opened.")
        print("Close Windows Camera, Zoom, Teams, browser camera tabs, then try again.")
        print("Also check Windows Settings > Privacy & security > Camera.")
        return 1

    analyzer = FaceAnalyzer(max_num_faces=MAX_FACES)
    calibration = {}
    calibration_centers = {}
    print("Calibration: keep your eyes OPEN for 5 seconds. Up to 5 faces can be detected.")
    end = time.time() + 5
    while time.time() < end:
        ok, frame = cap.read()
        if not ok:
            cap.release(); cv2.destroyAllWindows()
            print("[ERROR] Camera frame read failed during calibration.")
            return 1
        faces = analyzer.process_all(frame)
        for i, face in enumerate(faces):
            calibration.setdefault(i, []).append(face["ear"])
            calibration_centers[i] = np.array(face["center"], dtype=float)
        remaining = max(0, end - time.time())
        cv2.putText(frame, f"Calibration - eyes OPEN - {remaining:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("AI training data collection - Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release(); cv2.destroyAllWindows(); return 0

    trackers = {}
    last_centers = {}
    for i, ears in calibration.items():
        baseline = float(np.median(ears)) if ears else 0.21
        trackers[i] = TemporalTracker(
            ear_thr=calibrate_threshold(ears), baseline_ear=baseline
        )
        last_centers[i] = calibration_centers.get(i)

    if not trackers:
        cap.release(); cv2.destroyAllWindows()
        print("[ERROR] No face detected during calibration. Try again with your face visible.")
        return 1

    with open(CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(FEATURE_NAMES + ["label", "session"])

        last_write = 0.0
        n = 0
        print(f"Recording {label_name.upper()} | session={session} | Press Q to stop")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Camera frame read failed.")
                break
            t = time.time()
            faces = analyzer.process_all(frame)

            # Simple nearest-center identity for the short collection session.
            matched = {}
            for face in faces:
                center = np.array(face["center"], dtype=float)
                best_id, best_d = None, 1e9
                for fid in trackers:
                    prev = last_centers.get(fid)
                    if prev is None:
                        continue
                    d = float(np.linalg.norm(center - prev))
                    if d < best_d and d < 0.20:
                        best_id, best_d = fid, d
                if best_id is None:
                    unused = [i for i in range(MAX_FACES) if i not in trackers and i not in matched]
                    if unused:
                        best_id = unused[0]
                        trackers[best_id] = TemporalTracker(
                            ear_thr=calibrate_threshold([face["ear"]]),
                            baseline_ear=max(float(face["ear"]), 0.1),
                        )
                if best_id is None:
                    continue
                last_centers[best_id] = center
                matched[best_id] = True
                feats = trackers[best_id].update(
                    t, face["ear"], face["mar"], face["pitch"], face["yaw"], face["roll"]
                )
                if t - last_write >= 0.15:
                    writer.writerow([round(float(x), 6) for x in feats] + [label, session])
                    n += 1

            if t - last_write >= 0.15:
                f.flush(); last_write = t

            cv2.putText(frame, f"{label_name.upper()} | samples: {n} | faces: {len(faces)} | Q = stop",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("AI training data collection - Q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release(); cv2.destroyAllWindows()
    print(f"Saved {n} samples to {CSV} (session {session})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
