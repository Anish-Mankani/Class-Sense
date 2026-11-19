"""Module 2b - Face detection with YuNet (OpenCV).
Same job as face_detection.py, but it looks at the picture at full size,
so it can find smaller (more distant) faces.
The model file is downloaded once into cv_modules/models/.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
MODEL_PATH = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"


def ensure_model():
    """Download the YuNet model the first time. Returns the file path."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 50_000:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        if MODEL_PATH.stat().st_size <= 50_000:
            raise RuntimeError("downloaded file is too small")
    except Exception as error:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the YuNet model. Run this in Terminal from the "
            f"ClassSense folder instead:\n  curl -L -o {MODEL_PATH} {MODEL_URL}"
        ) from error
    return MODEL_PATH


def to_face(row):
    """YuNet row = x, y, w, h, 10 landmark numbers, score."""
    x, y, w, h = row[:4]
    return {"box": (max(0, int(x)), max(0, int(y)), int(w), int(h)),
            "score": float(row[-1]),
            "landmarks": [(float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5)]}


def looks_like_face(face):
    """Reject boxes that are too thin or too wide to be a human face."""
    _, _, w, h = face["box"]
    return w > 0 and h > 0 and 0.5 <= w / h <= 1.4


class YuNetFaceDetector:
    def __init__(self, min_confidence=0.6):
        self.detector = cv2.FaceDetectorYN.create(
            str(ensure_model()), "", (320, 320), min_confidence, 0.3, 5000
        )

    def detect(self, frame):
        """Return a list of faces: {"box": (x, y, w, h), "score": 0.0-1.0}."""
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))  # use the full picture size
        _, rows = self.detector.detect(frame)
        if rows is None:
            return []
        faces = [to_face(row) for row in rows]
        return [face for face in faces if looks_like_face(face)]
