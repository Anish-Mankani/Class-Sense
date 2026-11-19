import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project folder: lets this file import cv_modules, backend, client

"""Compare phone models on your recorded clips, in one table.
Run:  python compare_models.py phone_best2.pt phone_ft1.pt
Optional:  --conf 0.72   (the confidence limit, default 0.72)
Numbers = percent of frames where a phone was reported.
phone_back / phone_front: want HIGH.   no_phone / charger / calculator / door: want LOW."""
import sys
import time
from pathlib import Path

from cv_modules.video_input import VideoSource
from cv_modules.phone_detection import PhoneDetector

args = sys.argv[1:]
conf = 0.72
if "--conf" in args:
    i = args.index("--conf")
    conf = float(args[i + 1])
    del args[i:i + 2]
models = args or sorted(p.name for p in Path("models").glob("phone_*.pt"))
first = ["phone_back", "phone_front", "no_phone", "charger"]
found = {p.stem for p in Path("data").glob("*.mp4") if not p.stem.startswith(("neg", "walk"))}
clips = [c for c in first if c in found] + sorted(found - set(first))   # every test clip in data/
if not clips:
    raise SystemExit("No clips found in data/. Record them with record_clip.py first.")
print(f"Confidence {conf}. Clips: {clips}")
print(f"{'model':<18}" + "".join(f"{c:>13}" for c in clips) + f"{'fps':>8}")

for model in models:
    if not (Path(model).exists() or (Path("models") / model).exists()):
        print(f"{model:<18} (file not found in this folder)")
        continue
    detector = PhoneDetector(model, conf, 640)
    row, frames_total, seconds_total = [], 0, 0.0
    for clip in clips:
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
        seconds_total += time.time() - start
        frames_total += frames
        row.append(100 * hits / max(frames, 1))
    print(f"{model:<18}" + "".join(f"{v:>13.0f}" for v in row) + f"{frames_total / max(seconds_total, 0.001):>8.1f}")
