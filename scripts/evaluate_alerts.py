import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project folder: lets this file import cv_modules, backend, client

"""Count ALERTS (not single frames) on the recorded clips, like the real system.
The model runs ONCE per clip; then we try several "frames in a row" settings.
Run:  python evaluate_alerts.py [model] [confidence]
Example:  python evaluate_alerts.py phone_best2.pt 0.7"""
import sys
from pathlib import Path

from cv_modules.video_input import VideoSource
from cv_modules.phone_detection import PhoneDetector, PhoneAlertFilter

model = sys.argv[1] if len(sys.argv) > 1 else "phone_best2.pt"
conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
CLIPS = ["phone_back", "phone_front", "no_phone", "charger"]
FRAMES_IN_A_ROW = [5, 10, 15, 20, 30]

detector = PhoneDetector(model, conf, 640)
print(f"model {model}, confidence {conf}")
print("Alerts per clip, for different 'frames in a row' settings (cooldown 60 frames):")
print(f"{'clip':<13}{'frames':>8}" + "".join(f"{'N=' + str(n):>7}" for n in FRAMES_IN_A_ROW))
for clip in CLIPS:
    if not Path(f"data/{clip}.mp4").exists():
        print(f"{clip:<13} (clip missing)")
        continue
    video = VideoSource(f"data/{clip}.mp4")
    seen = []  # one True/False per frame: was a phone found?
    while True:
        frame = video.read()
        if frame is None:
            break
        seen.append(bool(detector.detect(frame)))
    video.release()
    row = []
    for needed in FRAMES_IN_A_ROW:
        alert_filter = PhoneAlertFilter(frames_needed=needed)
        row.append(sum(1 for found in seen if alert_filter.update([1] if found else [])))
    print(f"{clip:<13}{len(seen):>8}" + "".join(f"{count:>7}" for count in row))
print("phone_back / phone_front: want at least 1.   no_phone / charger: want 0.")
