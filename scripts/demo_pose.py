"""Live head pose and attention cues. ADVISORY ONLY.
Run:  python scripts/demo_pose.py                 (webcam)
      python scripts/demo_pose.py data/clip.mp4   (video file)
Green box = attentive, orange = looking away, red = head down, grey = too small or too far.
Press Q in the video window to quit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project folder: lets this file import cv_modules, backend, client

import cv2
from cv_modules.attention import AttentionEstimator, class_attention
from cv_modules.face_detection_yunet import YuNetFaceDetector
from cv_modules.tracking import FaceTracker
from cv_modules.video_input import VideoSource

COLORS = {"attentive": (0, 200, 0), "looking away": (0, 140, 255), "head down": (0, 0, 255), "unknown": (160, 160, 160)}

source = sys.argv[1] if len(sys.argv) > 1 else "0"
if source.isdigit():
    source = int(source)
video = VideoSource(source, width=1280, height=720)
detector = YuNetFaceDetector()
tracker = FaceTracker()
attention = AttentionEstimator()
print(f"Limits: yaw {attention.yaw_limit}, head down {attention.pitch_down_limit}, "
      f"pitch offset {attention.pitch_offset}. Click the video window and press Q to quit.")

while True:
    frame = video.read()
    if frame is None:
        break
    height, width = frame.shape[:2]
    faces = tracker.update(detector.detect(frame))
    results = attention.update(faces, (width, height))
    for result, face in zip(results, faces):
        x, y, w, h = result["box"]
        color = COLORS[result["state"]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{result['id']}: {result['state']}"
        if result["yaw"] is not None:
            text += f"  yaw {result['yaw']:+.0f} pitch {result['pitch']:+.0f}"
        cv2.putText(frame, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        for px, py in face.get("landmarks") or []:
            cv2.circle(frame, (int(px), int(py)), 2, color, -1)
    percent = class_attention(results)
    shown = "-" if percent is None else f"{percent:.0f}%"
    cv2.putText(frame, f"Faces: {len(results)}   Attentive: {shown}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.imshow("ClassSense head pose (press Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
