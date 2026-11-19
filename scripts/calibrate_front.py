"""Measure your 'looking straight at the board' baseline for MediaPipe pitch.
Run from the project root, sitting where a student would sit:
    python3 scripts/calibrate_front.py
Look straight at the screen for 5 seconds after the countdown. Saves mp_front_pitch
into data/pose_calibration.json (other settings are kept)."""
import json
import sys
import time
from pathlib import Path
from statistics import median

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv_modules.mp_pitch import MediaPipePitch

OUT = Path("data/pose_calibration.json")
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
if not ok:
    raise SystemExit("Could not read from the camera.")
h, w = frame.shape[:2]
det = cv2.FaceDetectorYN.create("cv_modules/models/face_detection_yunet_2023mar.onnx", "", (w, h), 0.5, 0.3, 5000)
mp = MediaPipePitch()
values, t0 = [], time.time()
while time.time() - t0 < 8:
    ok, frame = cap.read()
    if not ok:
        continue
    el = time.time() - t0
    _, dets = det.detect(frame)
    if dets is not None and len(dets):
        row = max(dets, key=lambda d: d[2] * d[3])
        if el >= 3:
            p = mp.pitch(frame, tuple(int(v) for v in row[0:4]))
            if p is not None:
                values.append(p)
    text = "Get ready" if el < 3 else "Look straight at the screen"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if el >= 3 else (255, 255, 255), 2)
    cv2.imshow("Calibrate front", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        raise SystemExit("Aborted. Nothing saved.")
cap.release()
cv2.destroyAllWindows()
if len(values) < 15:
    raise SystemExit(f"Only {len(values)} usable frames. Check lighting and that your face is in view, then run again.")
data = json.loads(OUT.read_text()) if OUT.exists() else {}
old = data.get("mp_front_pitch")
data["mp_front_pitch"] = round(median(values), 1)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, indent=2))
print(f"mp_front_pitch: {old} -> {data['mp_front_pitch']}  ({len(values)} frames)")
