import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project folder: lets this file import cv_modules, backend, client

"""Which objects does the phone detector mistake for a phone?
Hold each object in front of the camera for 5 seconds. The program runs EVERY model
in MODELS on the same frames and prints one table.
Run:  python test_objects.py
  or: python test_objects.py charger calculator "door lock"      (your own list)
In the video window:  SPACE = start the 5-second test,  S = skip this object,  Q = quit."""
import csv
import sys
import time

import cv2
from cv_modules.video_input import VideoSource
from cv_modules.phone_detection import PhoneDetector, draw_phones

MODELS = [          # (model file, confidence). Add the models you want to compare.
    ("phone_best2.pt", 0.7),
    ("yolov8n.pt", 0.4),
]
SECONDS = 5
DEFAULT_OBJECTS = ["phone_back", "phone_front", "charger", "calculator", "door lock",
                   "wall", "power bank", "wallet", "notebook", "remote", "tablet", "keyboard"]
POSITIVE = ("phone_back", "phone_front")   # these SHOULD be detected

objects = sys.argv[1:] or DEFAULT_OBJECTS
detectors = [PhoneDetector(model, conf, 640) for model, conf in MODELS]
video = VideoSource(0, width=1280, height=720)
results = []  # (object, [(percent flagged, best confidence) per model])


def show(frame, text, color=(255, 255, 0)):
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Object test (SPACE start, S skip, Q quit)", frame)
    return cv2.waitKey(1) & 0xFF


quit_all = False
for name in objects:
    # wait until the user is ready
    started = skipped = False
    while not started and not skipped:
        frame = video.read()
        if frame is None:
            quit_all = True
            break
        draw_phones(frame, detectors[0].detect(frame))
        key = show(frame, f"Object: {name}  -  press SPACE to start")
        started, skipped, quit_all = key == ord(" "), key == ord("s"), key == ord("q")
        if quit_all:
            break
    if quit_all:
        break
    if skipped:
        continue

    frames = 0
    flagged = [0] * len(detectors)
    best = [0.0] * len(detectors)
    begin = time.time()
    while time.time() - begin < SECONDS:
        frame = video.read()
        if frame is None:
            break
        frames += 1
        for i, detector in enumerate(detectors):
            phones = detector.detect(frame)
            if phones:
                flagged[i] += 1
                best[i] = max(best[i], max(p["score"] for p in phones))
        show(frame, f"{name}: recording {SECONDS - int(time.time() - begin)}s", (0, 0, 255))
    results.append((name, [(100 * flagged[i] / max(frames, 1), best[i]) for i in range(len(detectors))]))

video.release()
cv2.destroyAllWindows()

header = f"{'object':<14}" + "".join(f"{model + ' @' + str(conf):>24}" for model, conf in MODELS)
print("\nPercent of frames flagged as a phone (best confidence). Non-phone objects should be 0.")
print(header)
for name, values in results:
    tag = "  <- should be HIGH" if name in POSITIVE else ""
    print(f"{name:<14}" + "".join(f"{f'{p:.0f}% ({c:.2f})':>24}" for p, c in values) + tag)

with open("data/object_test_results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["object"] + [f"{m} {c} percent" for m, c in MODELS] + [f"{m} {c} best conf" for m, c in MODELS])
    for name, values in results:
        writer.writerow([name] + [round(p) for p, _ in values] + [round(c, 2) for _, c in values])
print("\nSaved data/object_test_results.csv")
