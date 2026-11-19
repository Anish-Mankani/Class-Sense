import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project folder: lets this file import cv_modules, backend, client

"""Compare phone-detection settings on the SAME recorded clips.
Needs the clips in data/ (see CLIPS below).
Run:  python evaluate_phones.py
Each number = percent of frames where a phone was reported.
phone_back / phone_front: higher is better.  no_phone / charger: lower is better."""
import time
from pathlib import Path

from cv_modules.video_input import VideoSource
from cv_modules.phone_detection import PhoneDetector

CONFIGS = [  # (model, image size, confidence)
    ("phone_best2.pt", 640, 0.72),
    ("phone_best3.pt", 640, 0.72),
]
CLIPS = ["phone_back", "phone_front", "no_phone", "charger"]


def run(detector, clip):
    """Return (percent of frames with a phone found, seconds used, frames)."""
    video = VideoSource(f"data/{clip}.mp4")
    frames = hits = 0
    start = time.time()
    while True:
        frame = video.read()
        if frame is None:
            break
        frames += 1
        hits += 1 if detector.detect(frame) else 0
    video.release()
    return (100 * hits / frames if frames else 0.0), time.time() - start, frames


missing = [c for c in CLIPS if not Path(f"data/{c}.mp4").exists()]
if missing:
    raise SystemExit(f"Missing clips in data/: {missing}. Record them with record_clip.py first.")

print(f"{'model':<14}{'size':>5}{'conf':>6} | " + "".join(f"{c:>13}" for c in CLIPS) + f"{'fps':>8}")
for model, size, conf in CONFIGS:
    detector = PhoneDetector(model, conf, size)
    results = [run(detector, clip) for clip in CLIPS]
    fps = sum(r[2] for r in results) / max(sum(r[1] for r in results), 0.001)
    print(f"{model:<14}{size:>5}{conf:>6} | " + "".join(f"{r[0]:>13.0f}" for r in results) + f"{fps:>8.1f}")
