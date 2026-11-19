"""Calibrate head-pose limits for YOUR camera. Run from the project root:
    python3 scripts/calibrate_pose.py
Sit where a student would sit. For each step, get ready during the countdown,
then hold the pose while it records. Esc = abort.
Saves data/pose_calibration.json (read automatically by cv_modules/attention.py)."""
import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv_modules.attention import suggest_limits  # noqa: E402
from cv_modules.head_pose import eye_distance, head_pose  # noqa: E402

OUT_FILE = Path("data/pose_calibration.json")
STEPS = [
    ("front", "Look straight at the board / screen"),
    ("left", "Turn your head to your LEFT, as far as a student would"),
    ("right", "Turn your head to your RIGHT, as far as a student would"),
    ("down", "Look DOWN at a desk, like writing or reading"),
    ("up", "Tilt your head UP, as if looking at the ceiling"),
]
GET_READY_SECONDS = 3.0
RECORD_SECONDS = 4.0
MIN_EYE_DISTANCE = 25  # ignore tiny faces; they give noisy angles


def largest_face(dets):
    if dets is None or len(dets) == 0:
        return None
    return max(dets, key=lambda r: r[2] * r[3])


def banner(frame, lines, color=(255, 255, 255)):
    y = 40
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        y += 38


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cv_modules/models/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from the camera.")
    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(args.model, "", (w, h), 0.5, 0.3, 5000)

    samples = {name: [] for name, _ in STEPS}
    aborted = False
    for name, instruction in STEPS:
        start = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            elapsed = time.time() - start
            recording = elapsed >= GET_READY_SECONDS
            _, dets = detector.detect(frame)
            row = largest_face(dets)
            if row is not None:
                x, y, bw, bh = (int(v) for v in row[0:4])
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 0), 2)
                if recording:
                    lm = np.asarray(row[4:14], dtype=np.float64).reshape(5, 2)
                    if eye_distance(lm) >= MIN_EYE_DISTANCE:
                        pose = head_pose(lm, (w, h))
                        if pose is not None:
                            samples[name].append((pose[0], pose[1]))
            if recording:
                left = RECORD_SECONDS - (elapsed - GET_READY_SECONDS)
                banner(frame, [f"RECORDING {name.upper()}  {left:0.1f}s", instruction], (0, 0, 255))
            else:
                left = GET_READY_SECONDS - elapsed
                banner(frame, [f"Get ready: {left:0.0f}", instruction])
            cv2.imshow("Calibrate", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                aborted = True
                break
            if elapsed >= GET_READY_SECONDS + RECORD_SECONDS:
                break
        if aborted:
            break
    cap.release()
    cv2.destroyAllWindows()
    if aborted:
        raise SystemExit("Aborted. Nothing saved.")

    print(f"\n{'pose':<8}{'samples':>8}{'yaw':>9}{'pitch':>9}")
    for name, _ in STEPS:
        s = samples[name]
        if s:
            print(f"{name:<8}{len(s):>8}{median(y for y, _ in s):>9.1f}{median(p for _, p in s):>9.1f}")
        else:
            print(f"{name:<8}{0:>8}{'-':>9}{'-':>9}")

    settings = suggest_limits(samples)
    missing = {"yaw_limit", "pitch_down_limit", "pitch_offset", "pitch_up_limit"} - set(settings)
    if missing:
        print("\nNot enough usable frames for:", ", ".join(sorted(missing)))
        print("Check lighting and that your face stays big enough in the picture, then run again.")
    if not settings:
        raise SystemExit("Nothing to save.")

    existing = {}
    if OUT_FILE.exists():
        try:
            existing = json.loads(OUT_FILE.read_text())
        except ValueError:
            pass
    existing.update(settings)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(existing, indent=2))
    print("\nSaved", OUT_FILE)
    for k, v in settings.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
