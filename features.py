"""Face/temporal feature extraction for AI drowsiness detection.

The classifier, not a hand-written drowsiness rule, makes the final decision.
Thresholds here are only used to derive blink/eye-closure/yawn statistics.
"""
from collections import deque
import cv2
import mediapipe as mp
import numpy as np

FEATURE_NAMES = [
    "ear", "mar", "ear_ratio", "closure_dur", "blink_rate", "perclos",
    "yawn_rate", "mouth_open_ratio", "pitch", "yaw", "roll",
    "ear_mean", "ear_std", "mar_mean", "mar_std"
]

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_H = (61, 291)
MOUTH_V = [(81, 178), (13, 14), (311, 402)]
POSE_IDX = [1, 152, 33, 263, 61, 291]
MODEL_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)
], dtype=np.float64)


def _pt(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float64)


def eye_aspect_ratio(lm, idx, w, h):
    p = [_pt(lm, i, w, h) for i in idx]
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    hz = np.linalg.norm(p[0] - p[3])
    return float(v / (2.0 * hz + 1e-6))


def mouth_aspect_ratio(lm, w, h):
    v = np.mean([
        np.linalg.norm(_pt(lm, a, w, h) - _pt(lm, b, w, h))
        for a, b in MOUTH_V
    ])
    hz = np.linalg.norm(_pt(lm, MOUTH_H[0], w, h) - _pt(lm, MOUTH_H[1], w, h))
    return float(v / (hz + 1e-6))


def head_pose(lm, w, h):
    pts2d = np.array([_pt(lm, i, w, h) for i in POSE_IDX], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, pts2d, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rmat)[0]
    pitch, yaw, roll = [float(x) for x in angles]
    if pitch > 90: pitch -= 180
    elif pitch < -90: pitch += 180
    return pitch, yaw, roll


class FaceAnalyzer:
    def __init__(self, max_num_faces=5):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process_all(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return []
        out = []
        for face in res.multi_face_landmarks:
            lm = face.landmark
            ear = (eye_aspect_ratio(lm, LEFT_EYE, w, h) + eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2
            mar = mouth_aspect_ratio(lm, w, h)
            pitch, yaw, roll = head_pose(lm, w, h)
            xs = np.array([p.x for p in lm]); ys = np.array([p.y for p in lm])
            out.append({
                "ear": float(ear), "mar": float(mar), "pitch": pitch, "yaw": yaw, "roll": roll,
                "landmarks": lm, "center": (float(xs.mean()), float(ys.mean())),
                "box": (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            })
        return out

    def process(self, frame_bgr):
        faces = self.process_all(frame_bgr)
        if not faces:
            return None
        r = faces[0]
        return r["ear"], r["mar"], r["pitch"], r["yaw"], r["roll"], r["landmarks"]


class TemporalTracker:
    """Converts face measurements into a rolling feature vector for the ML model."""
    def __init__(self, window=30.0, ear_thr=0.21, mar_thr=0.60, baseline_ear=None):
        self.window = window
        self.ear_thr = float(ear_thr)
        self.mar_thr = float(mar_thr)
        self.baseline_ear = baseline_ear
        self.hist = deque()       # t, ear, mar, closed, mouth_open
        self.blinks = deque()
        self.yawns = deque()
        self.closed_since = None
        self.yawn_since = None
        self.t0 = None

    def update(self, t, ear, mar, pitch, yaw, roll):
        if self.t0 is None:
            self.t0 = t
        if self.baseline_ear is None:
            self.baseline_ear = max(float(ear), 1e-3)

        # The model receives a continuous, person-normalized EAR as well as raw EAR.
        ear_ratio = float(ear / max(self.baseline_ear, 1e-3))
        closed = ear < self.ear_thr
        mouth_open = mar > self.mar_thr

        if closed and self.closed_since is None:
            self.closed_since = t
        if not closed and self.closed_since is not None:
            dur = t - self.closed_since
            if 0.05 <= dur <= 0.80:
                self.blinks.append(t)
            self.closed_since = None
        closure_dur = (t - self.closed_since) if closed else 0.0

        if mouth_open and self.yawn_since is None:
            self.yawn_since = t
        if not mouth_open and self.yawn_since is not None:
            dur = t - self.yawn_since
            if 0.8 <= dur <= 8.0:
                self.yawns.append(t)
            self.yawn_since = None

        self.hist.append((t, float(ear), float(mar), closed, mouth_open))
        while self.hist and t - self.hist[0][0] > self.window:
            self.hist.popleft()
        while self.blinks and t - self.blinks[0] > self.window:
            self.blinks.popleft()
        while self.yawns and t - self.yawns[0] > self.window:
            self.yawns.popleft()

        span = max(min(self.window, t - self.t0), 5.0)
        perclos = float(np.mean([c for _, _, _, c, _ in self.hist])) if self.hist else 0.0
        mouth_ratio = float(np.mean([m for _, _, _, _, m in self.hist])) if self.hist else 0.0
        ears = np.array([e for _, e, _, _, _ in self.hist], dtype=float)
        mars = np.array([m for _, _, m, _, _ in self.hist], dtype=float)
        blink_rate = len(self.blinks) * 60.0 / span
        yawn_rate = len(self.yawns) * 60.0 / span

        return [
            float(ear), float(mar), ear_ratio, float(closure_dur), float(blink_rate),
            perclos, float(yawn_rate), mouth_ratio, float(pitch), float(yaw), float(roll),
            float(ears.mean()) if len(ears) else float(ear),
            float(ears.std()) if len(ears) else 0.0,
            float(mars.mean()) if len(mars) else float(mar),
            float(mars.std()) if len(mars) else 0.0,
        ]


def calibrate_threshold(ears, default=0.21):
    if len(ears) < 20:
        return default
    return float(np.clip(0.72 * np.median(ears), 0.14, 0.32))
