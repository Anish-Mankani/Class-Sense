import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project folder: lets this file import cv_modules, backend, client

"""Run:  python demo_phones.py                 (webcam)
       python demo_phones.py video.mp4         (video file)
Shows faces (green) and phones (red). A phone seen in 5 frames in a row
raises an ALERT. Press Q in the video window to quit."""
import sys
import time
import cv2
from cv_modules.video_input import VideoSource
from cv_modules.face_detection import draw_faces
from cv_modules.face_detection_yunet import YuNetFaceDetector
from cv_modules.phone_detection import PhoneDetector, PhoneAlertFilter, draw_phones
from cv_modules.tracking import FaceTracker

source = sys.argv[1] if len(sys.argv) > 1 else "0"
if source.isdigit():
    source = int(source)

video = VideoSource(source, width=1280, height=720)
faces_model = YuNetFaceDetector()
model_name = sys.argv[2] if len(sys.argv) > 2 else "phone_best2.pt"
image_size = int(sys.argv[3]) if len(sys.argv) > 3 else 640
min_conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.72
print("Phone model:", model_name, "| image size:", image_size, "| confidence:", min_conf)
phones_model = PhoneDetector(model_name, min_conf, image_size)
alert_filter = PhoneAlertFilter(frames_needed=20)
face_tracker = FaceTracker()
alert_count = 0
alert_shown_until = 0
print("Started. Hold a phone in front of the camera. Click the video window and press Q to quit.")

while True:
    frame = video.read()
    if frame is None:
        print("No frame received from the video source. Stopping.")
        break
    faces = face_tracker.update(faces_model.detect(frame))
    phones = phones_model.detect(frame)
    if alert_filter.update(phones):
        alert_count += 1
        alert_shown_until = time.time() + 3
        print(f"ALERT {alert_count}: possible phone use at {time.strftime('%H:%M:%S')}")
    frame = draw_phones(draw_faces(frame, faces), phones)
    cv2.putText(frame, f"Faces: {len(faces)}  Phones: {len(phones)}  Alerts: {alert_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    if time.time() < alert_shown_until:
        cv2.putText(frame, "ALERT: possible phone use", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imshow("ClassSense - faces and phones (press Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
print("Closed.")
