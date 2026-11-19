import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project folder: lets this file import cv_modules, backend, client

"""Record a short test clip from the webcam.
Run:  python record_clip.py phone_back 20
Saves data/phone_back.mp4. Press Q to stop early."""
import sys
import time
from pathlib import Path

import cv2
from cv_modules.video_input import VideoSource

name = sys.argv[1] if len(sys.argv) > 1 else "clip"
seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
Path("data").mkdir(exist_ok=True)
path = f"data/{name}.mp4"

video = VideoSource(0, width=1280, height=720)
first = video.read()
height, width = first.shape[:2]
writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height))

print("Get ready... recording starts in 3 seconds")
start = time.time()
while time.time() - start < 3:
    frame = video.read()
    shown = frame.copy()
    cv2.putText(shown, f"Starting in {3 - int(time.time() - start)}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow("Recording (press Q to stop)", shown)
    cv2.waitKey(1)

print(f"RECORDING {name} for {seconds} seconds")
start = time.time()
while time.time() - start < seconds:
    frame = video.read()
    if frame is None:
        break
    writer.write(frame)
    shown = frame.copy()
    cv2.putText(shown, f"REC {seconds - int(time.time() - start)}s", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imshow("Recording (press Q to stop)", shown)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

writer.release()
video.release()
cv2.destroyAllWindows()
print("Saved", path)
