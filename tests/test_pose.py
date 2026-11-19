"""Checks for head pose and attention. They use a simulated face, so no camera is needed.
Run:  python tests/test_pose.py"""
import itertools
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project folder: lets this file import cv_modules, backend, client

import numpy as np

from cv_modules import attention
from cv_modules.attention import AttentionEstimator, class_attention, suggest_limits
from cv_modules.head_pose import (EYE_HALF, MOUTH_DEPTH, MOUTH_DROP, NOSE_DEPTH, NOSE_DROP,
                                  eye_distance, head_pose)

FACE_3D = np.array([(-EYE_HALF, 0, 0), (EYE_HALF, 0, 0), (0, -NOSE_DROP, NOSE_DEPTH),
                    (-25, -MOUTH_DROP, MOUTH_DEPTH), (25, -MOUTH_DROP, MOUTH_DEPTH)], float)


def make_landmarks(yaw, pitch, roll=0.0, distance=3000.0, focal=1280.0, centre=(640, 360)):
    """The 5 landmarks a camera would see when a model face is turned by yaw, pitch and roll (degrees)."""
    ty, tp, tr = map(math.radians, (yaw, pitch, roll))
    Ry = np.array([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(tp), -math.sin(tp)], [0, math.sin(tp), math.cos(tp)]])
    Rz = np.array([[math.cos(tr), -math.sin(tr), 0], [math.sin(tr), math.cos(tr), 0], [0, 0, 1]])
    points = (Rz @ Rx @ Ry @ FACE_3D.T).T * np.array([1, -1, 1])
    points[:, 2] = distance - points[:, 2]
    return [(centre[0] + focal * x / z, centre[1] + focal * y / z) for x, y, z in points]


passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    print(("[PASS] " if condition else "[FAIL] ") + name + (f"   {detail}" if detail else ""))
    passed += bool(condition)
    failed += not condition


# ---------- head pose geometry ----------
yaw, pitch, roll = head_pose(make_landmarks(0, 0))
check("a face looking straight ahead gives about 0 / 0 / 0", max(abs(yaw), abs(pitch), abs(roll)) < 1,
      f"yaw {yaw:.1f} pitch {pitch:.1f} roll {roll:.1f}")

worst = 0.0
for y, p, r in itertools.product(range(-60, 61, 15), range(-30, 31, 15), (-15, 0, 15)):
    estimate = head_pose(make_landmarks(y, p, r))
    worst = max(worst, abs(estimate[0] - y), abs(estimate[1] - p), abs(estimate[2] - r))
check("angles are recovered for yaw -60..60, pitch -30..30, roll -15..15 (clean landmarks)", worst < 1.5, f"worst error {worst:.2f} degrees")
check("turning left and right gives opposite signs", head_pose(make_landmarks(-35, 0))[0] < -30 and head_pose(make_landmarks(35, 0))[0] > 30)
check("tipping the head down gives positive pitch", head_pose(make_landmarks(0, 25))[1] > 20)
check("unusable landmarks give None", head_pose([(5, 5)] * 5) is None)
check("eye distance is measured", abs(eye_distance(make_landmarks(0, 0)) - 2 * EYE_HALF * 1280 / 3000) < 0.5)


# ---------- attention logic ----------
def face(fid, yaw, pitch, **kw):
    return {"id": fid, "box": (500, 300, 100, 120), "score": 0.9, "landmarks": make_landmarks(yaw, pitch, **kw)}


def run(estimator, yaw, pitch, frames, **kw):
    state = None
    for _ in range(frames):
        state = estimator.update([face(1, yaw, pitch, **kw)])[0]["state"]
    return state


est = AttentionEstimator(yaw_limit=30, pitch_down_limit=25, pitch_offset=0, hold_frames=10, smoothing=0.5)
check("facing the front is attentive", run(est, 0, 0, 20) == "attentive")
check("a quick glance away (3 frames) is ignored", run(est, 50, 0, 3) == "attentive")
run(est, 0, 0, 20)
check("looking away for a while is flagged", run(est, 50, 0, 25) == "looking away")
check("turning back clears the flag", run(est, 0, 0, 25) == "attentive")
check("head down for a while is flagged", run(est, 0, 40, 25) == "head down")
run(est, 0, 0, 25)

two = AttentionEstimator(hold_frames=5, smoothing=1.0)
for _ in range(10):
    out = two.update([face(1, 0, 0), face(2, 55, 0)])
by_id = {r["id"]: r["state"] for r in out}
check("two faces are judged separately", by_id == {1: "attentive", 2: "looking away"}, str(by_id))
check("class attention = percent attentive", class_attention(out) == 50.0)

tiny = AttentionEstimator(hold_frames=1)
small = tiny.update([face(1, 50, 0, distance=30000.0)])[0]
check("a face that is too small/far is 'unknown', not guessed", small["state"] == "unknown")
check("class attention ignores unknown faces", class_attention([small]) is None)
check("a face without landmarks is 'unknown'", tiny.update([{"id": 9, "box": (0, 0, 10, 10), "score": 0.9}])[0]["state"] == "unknown")

# the pitch offset (camera angle) moves 'front' to 0
shifted = AttentionEstimator(pitch_offset=15, hold_frames=1, smoothing=1.0)
check("pitch offset: a face tipped 15 degrees counts as front", shifted.update([face(1, 0, 15)])[0]["state"] == "attentive")

# calibration file is read
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "pose.json"
    path.write_text(json.dumps({"yaw_limit": 12.0, "pitch_down_limit": 9.0, "pitch_offset": 3.0}))
    original = attention.CALIBRATION_FILE
    attention.CALIBRATION_FILE = path
    loaded = AttentionEstimator()
    attention.CALIBRATION_FILE = original
check("limits are loaded from the calibration file", (loaded.yaw_limit, loaded.pitch_down_limit, loaded.pitch_offset) == (12.0, 9.0, 3.0))
check("explicit limits beat the calibration file", AttentionEstimator(yaw_limit=44).yaw_limit == 44)

# calibration advice
samples = {"front": [(2.0, 8.0)] * 20, "left": [(-48.0, 8.0)] * 20, "right": [(52.0, 9.0)] * 20, "down": [(1.0, 38.0)] * 20}
advice = suggest_limits(samples)
check("calibration: pitch offset = the 'front' pitch", advice["pitch_offset"] == 8.0, str(advice))
check("calibration: yaw limit sits between 'front' and 'turned'", 20 <= advice["yaw_limit"] <= 30, str(advice))
check("calibration: down limit is halfway to the 'down' pose", abs(advice["pitch_down_limit"] - 15.0) < 0.1, str(advice))
check("calibration: too few samples gives no advice", suggest_limits({"front": [(0, 0)] * 3}) == {})

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
