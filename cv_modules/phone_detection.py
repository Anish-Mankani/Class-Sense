"""Module 3 - Mobile phone detection (YOLOv8n).
YOLOv8n is already trained on 80 kinds of objects, and "cell phone" is one of
them, so we do not train anything. The weights file (yolov8n.pt, about 6 MB)
downloads by itself the first time.
"""
import cv2
from pathlib import Path

from ultralytics import YOLO

CELL_PHONE_CLASS = 67  # class number of "cell phone" in the COCO dataset


def find_model(name):
    """Look for a model file in the current folder, then in models/."""
    for candidate in (Path(name), Path("models") / name):
        if candidate.exists():
            return str(candidate)
    return name   # not found: YOLO may download a standard model


PHONE_NAMES = ("cell phone", "phone", "mobile", "mobile phone", "smartphone")


def phone_class_ids(names):
    """Class numbers that mean 'phone'. A model with a single class is assumed to be the phone class."""
    ids = [n for n, name in names.items() if name.lower().replace("_", " ").replace("-", " ").strip() in PHONE_NAMES]
    if not ids and len(names) == 1:
        ids = list(names)
    return ids


class PhoneDetector:
    def __init__(self, model_name="yolov8n.pt", min_confidence=0.4, image_size=640):
        self.model = YOLO(find_model(model_name))
        # find the phone class number in this model (67 in the standard model, 0 in ours)
        self.classes = phone_class_ids(self.model.names)
        if not self.classes:
            raise ValueError(f"No phone class found in {model_name}. Its classes are: {dict(self.model.names)}")
        self.min_confidence = min_confidence
        self.image_size = image_size

    def detect(self, frame):
        """Return a list of phones: {"box": (x, y, w, h), "score": 0.0-1.0}."""
        results = self.model.predict(
            frame, classes=self.classes, conf=self.min_confidence,
            imgsz=self.image_size, verbose=False,
        )
        height, width = frame.shape[:2]
        phones = [to_phone(box) for box in results[0].boxes]
        return [phone for phone in phones if plausible_phone(phone, width, height)]


def to_phone(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return {"box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            "score": float(box.conf[0])}


def plausible_phone(phone, frame_width, frame_height):
    """Reject boxes that cannot be a phone: far too big or tiny, or the wrong shape."""
    _, _, w, h = phone["box"]
    if w <= 0 or h <= 0:
        return False
    area = (w * h) / (frame_width * frame_height)
    shape = min(w, h) / max(w, h)  # 1.0 = square, near 0 = a thin stick
    return 0.0003 <= area <= 0.30 and 0.25 <= shape <= 0.90


def draw_phones(frame, phones):
    """Draw a red box on each phone."""
    for phone in phones:
        x, y, w, h = phone["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"Phone {phone['score']:.2f}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


class PhoneAlertFilter:
    """Raise an alert only when a phone is seen in several frames in a row.
    This avoids false alarms from one bad frame (SRS requirement FR-12)."""

    def __init__(self, frames_needed=5, cooldown_frames=60):
        self.frames_needed = frames_needed
        self.cooldown_frames = cooldown_frames
        self.seen_in_a_row = 0
        self.cooldown = 0

    def update(self, phones):
        """Call once per frame. Returns True on the frame an alert should be raised."""
        if self.cooldown > 0:
            self.cooldown -= 1
        self.seen_in_a_row = self.seen_in_a_row + 1 if phones else 0
        if self.seen_in_a_row >= self.frames_needed and self.cooldown == 0:
            self.cooldown = self.cooldown_frames
            return True
        return False
