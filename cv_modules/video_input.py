"""Module 1 - Video input.
Opens the laptop webcam (source=0) or a video file (source="class.mp4")
and gives us one frame at a time.
"""
import cv2


class VideoSource:
    def __init__(self, source=0, width=640, height=480):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        if isinstance(source, int):  # webcam only: ask for a smaller picture (faster)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        """Return one frame (BGR image) or None when the video has ended."""
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()
