"""Prints what the system sees for 15 seconds. Run once without glasses, once with:
    python3 scripts/diag_glasses.py
Seconds 0-5 look straight at the screen, 5-10 turn your head to the side, 10-15 look down."""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv_modules.attention import AttentionEstimator
from cv_modules.head_pose import eye_distance
from cv_modules.mp_pitch import MediaPipePitch
from run_live import SimpleTracker

fmt = lambda v: "-" if v is None else f"{v:+.0f}"
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
h, w = frame.shape[:2]
det = cv2.FaceDetectorYN.create("cv_modules/models/face_detection_yunet_2023mar.onnx", "", (w, h), 0.5, 0.3, 5000)
mp, tracker, est = MediaPipePitch(), SimpleTracker(), AttentionEstimator(down_hold_frames=90)
print("time   eyes_px  mediapipe  yaw  pitch  state")
t0 = last = time.time()
while time.time() - t0 < 15:
    ok, frame = cap.read()
    if not ok:
        continue
    _, dets = det.detect(frame)
    faces = tracker.update(dets if dets is not None else [])
    for f in faces:
        f["mp_pitch"] = mp.pitch(frame, f["box"])
    res = est.update(faces, (w, h))
    if time.time() - last >= 1:
        last = time.time()
        if not faces:
            print(f"{last - t0:4.0f}s  NO FACE FOUND")
        for f, r in zip(faces, res):
            m = f["mp_pitch"]
            print(f"{last - t0:4.0f}s  {eye_distance(f['landmarks']):6.0f}  {'LOST' if m is None else fmt(m):>9}  {fmt(r['yaw']):>3}  {fmt(r['pitch']):>5}  {r['state']}")
cap.release()
