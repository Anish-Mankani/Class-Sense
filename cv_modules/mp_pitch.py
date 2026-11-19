"""Up/down head angle from MediaPipe Face Landmarker, run on a crop around one YuNet face.
pitch(frame, box) -> degrees (positive = looking UP), or None if MediaPipe cannot find the face."""
import math

import cv2
import numpy as np
import mediapipe as mp

MODEL = "cv_modules/models/face_landmarker.task"


def square_crop(frame, box, margin=0.4, size=256):
    """Square crop centred on the face box, padded with black where it leaves the frame."""
    x, y, w, h = box
    side = int(max(w, h) * (1 + 2 * margin))
    side = max(side, 16)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad = side
    padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = padded[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    return cv2.resize(crop, (size, size))


def forward_pitch(matrix):
    """Pitch in degrees (up positive) from MediaPipe's 4x4 face transform."""
    f = np.asarray(matrix, dtype=np.float64).reshape(4, 4)[:3, 2]
    f = f / np.linalg.norm(f)
    return math.degrees(math.atan2(f[1], f[2]))


class MediaPipePitch:
    def __init__(self, model_path=MODEL, margin=0.4):
        self.margin = margin
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_facial_transformation_matrixes=True)
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)

    def pitch(self, frame_bgr, box):
        crop = square_crop(frame_bgr, box, self.margin)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.facial_transformation_matrixes:
            return None
        return forward_pitch(result.facial_transformation_matrixes[0])
