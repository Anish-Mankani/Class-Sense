"""Eyes and movement probe: how does WRITING differ from SLEEPING? Run from the project root:
    python3 scripts/probe_eyes.py
Four poses, 3 s get-ready + 5 s recording each. Esc = abort."""
import sys
import time
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv_modules.mp_pitch import forward_pitch

MODEL = "cv_modules/models/face_landmarker.task"
STEPS = [("front", "Look straight at the screen"),
         ("writing", "Write on paper on your desk, eyes open, like a normal lesson"),
         ("eyes_closed", "Head down a little, EYES CLOSED, as if dozing"),
         ("head_on_arms", "Rest your head on your arms on the desk, eyes closed")]

if not Path(MODEL).exists():
    raise SystemExit(f"Missing {MODEL}.")
opts = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
    running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1,
    min_face_detection_confidence=0.3, min_face_presence_confidence=0.3, min_tracking_confidence=0.3,
    output_face_blendshapes=True, output_facial_transformation_matrixes=True)
landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
if not ok:
    raise SystemExit("Could not read from the camera.")
rows = {name: {"frames": 0, "face": 0, "pitch": [], "blink": [], "down": [], "x": [], "y": []} for name, _ in STEPS}
aborted = False
for name, text in STEPS:
    t0 = time.time()
    while time.time() - t0 < 8:
        ok, frame = cap.read()
        if not ok:
            continue
        el = time.time() - t0
        if el >= 3:
            rgb = cv2.cvtColor(cv2.resize(frame, (960, 540)), cv2.COLOR_BGR2RGB)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            r = rows[name]
            r["frames"] += 1
            if res.face_landmarks and res.face_blendshapes:
                r["face"] += 1
                bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
                r["blink"].append((bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2)
                r["down"].append((bs.get("eyeLookDownLeft", 0) + bs.get("eyeLookDownRight", 0)) / 2)
                if res.facial_transformation_matrixes:
                    r["pitch"].append(forward_pitch(res.facial_transformation_matrixes[0]))
                nose = res.face_landmarks[0][1]
                r["x"].append(nose.x)
                r["y"].append(nose.y)
        msg = f"{'RECORDING ' + name if el >= 3 else 'Get ready'}: {text}"
        cv2.putText(frame, msg[:90], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if el >= 3 else (255, 255, 255), 2)
        cv2.imshow("Eyes probe", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            aborted = True
            break
    if aborted:
        break
cap.release()
cv2.destroyAllWindows()
if aborted:
    raise SystemExit("Aborted.")
med = lambda v: "-" if not v else f"{median(v):+.2f}"
print(f"\n{'pose':<14}{'face found':>12}{'pitch(up+)':>12}{'eye blink':>11}{'look down':>11}{'motion':>9}")
for name, _ in STEPS:
    r = rows[name]
    motion = "-" if len(r["x"]) < 5 else f"{100 * float(np.hypot(np.std(r['x']), np.std(r['y']))):.2f}"
    pitch = "-" if not r["pitch"] else f"{median(r['pitch']):+.0f}"
    print(f"{name:<14}{r['face']:>5}/{r['frames']:<6}{pitch:>12}{med(r['blink']).lstrip('+'):>11}{med(r['down']).lstrip('+'):>11}{motion:>9}")
