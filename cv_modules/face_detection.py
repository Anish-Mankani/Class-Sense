"""Module 2 - Face detection (MediaPipe Tasks).
Finds all faces in a frame and returns their boxes.
The model file is downloaded once into cv_modules/models/.
"""
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
MODEL_PATH = Path(__file__).parent / "models" / "blaze_face_short_range.tflite"


def ensure_model():
    """Download the face model the first time. Returns the file path."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 10_000:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as error:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the face model. Run this in Terminal from the "
            f"ClassSense folder instead:\n  curl -L -o {MODEL_PATH} {MODEL_URL}"
        ) from error
    return MODEL_PATH


def to_face(detection):
    """Turn a MediaPipe detection into {"box": (x, y, w, h), "score": float}."""
    b = detection.bounding_box
    return {"box": (max(0, int(b.origin_x)), max(0, int(b.origin_y)),
                    int(b.width), int(b.height)),
            "score": float(detection.categories[0].score)}


class FaceDetector:
    def __init__(self, min_confidence=0.5):
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model())),
            min_detection_confidence=min_confidence,
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame):
        """Return a list of faces: {"box": (x, y, w, h), "score": 0.0-1.0}."""
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # MediaPipe wants RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(image)
        return [to_face(d) for d in result.detections]


def draw_faces(frame, faces):
    """Draw a green box and a number on each face."""
    for number, face in enumerate(faces, start=1):
        x, y, w, h = face["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Face {face.get('id', number)}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame
