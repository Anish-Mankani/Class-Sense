"""Live webcam check: YuNet -> head pose -> attention boxes.
Run from the project root (ClassSense):
    python3 run_live.py --model cv_modules/models/face_detection_yunet_2023mar.onnx
Keys: q = quit. Boxes: green attentive, orange looking away, red head down, grey unknown.
Shows yaw/pitch above each face so you can check the signs on a real face."""
import argparse
import cv2
import numpy as np

from cv_modules.attention import AttentionEstimator

COLORS = {  # BGR
    "attentive": (0, 200, 0),
    "looking away": (0, 140, 255),
    "head down": (0, 0, 255),
    "unknown": (140, 140, 140),
    "writing": (230, 130, 40),
}


class SimpleTracker:
    """Gives each face a stable id by matching box centers between frames.
    A detection only becomes a student after min_hits frames in a row, and boxes smaller than
    min_size pixels are ignored, so flickering false detections (curtains, patterns) are not counted.
    A confirmed face that disappears is remembered for max_age frames and keeps its id when it returns."""
    def __init__(self, max_dist=80, max_age=45, min_hits=8, min_size=40):
        self.max_dist, self.max_age, self.min_hits, self.min_size = max_dist, max_age, min_hits, min_size
        self.next_id = 1      # public ids, given only to confirmed faces
        self.next_key = 0
        self.tracks = {}      # internal key -> track details

    def update(self, detections):
        faces, used = [], set()
        for row in detections:
            size = max(row[2], row[3])
            if size < self.min_size:
                continue
            cx, cy = row[0] + row[2] / 2, row[1] + row[3] / 2
            best, best_d = None, None
            for key, tr in self.tracks.items():
                d = np.hypot(cx - tr["cx"], cy - tr["cy"])
                if key not in used and d < max(self.max_dist, 0.8 * tr["size"]) and (best_d is None or d < best_d):
                    best, best_d = key, d
            if best is None:
                best, self.next_key = self.next_key, self.next_key + 1
                self.tracks[best] = {"hits": 0, "pid": None}
            tr = self.tracks[best]
            tr.update(cx=cx, cy=cy, size=size, age=0, hits=tr["hits"] + 1)
            used.add(best)
            if tr["pid"] is None and tr["hits"] >= self.min_hits:
                tr["pid"], self.next_id = self.next_id, self.next_id + 1
            if tr["pid"] is not None:
                faces.append({"id": tr["pid"],
                              "box": tuple(int(v) for v in row[0:4]),
                              "landmarks": np.asarray(row[4:14], dtype=np.float64).reshape(5, 2)})
        for key in list(self.tracks):
            if key in used:
                continue
            tr = self.tracks[key]
            tr["age"] = tr.get("age", 0) + 1
            if tr["pid"] is None or tr["age"] > self.max_age:
                del self.tracks[key]
        return faces

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cv_modules/models/face_detection_yunet_2023mar.onnx")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from the camera.")
    h, w = frame.shape[:2]
    print("camera opened:", w, "x", h)
    detector = cv2.FaceDetectorYN.create(args.model, "", (w, h), 0.5, 0.3, 5000)
    tracker = SimpleTracker()
    estimator = AttentionEstimator(down_hold_frames=90)
    try:
        from cv_modules.mp_pitch import MediaPipePitch
        mp_pitch = MediaPipePitch()
    except Exception as e:
        print("MediaPipe pitch disabled:", e)
        mp_pitch = None

    failures = 0
    frames_seen = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            failures += 1
            if failures > 30:
                print("Camera stopped giving frames.")
                break
            continue
        failures = 0
        _, dets = detector.detect(frame)
        faces = tracker.update(dets if dets is not None else [])
        if mp_pitch is not None:
            for f in faces:
                f["mp_pitch"] = mp_pitch.pitch(frame, f["box"])
        for r in estimator.update(faces, (w, h)):
            x, y, bw, bh = r["box"]
            color = COLORS[r["state"]]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            label = r["state"] if r["yaw"] is None else f'{r["state"]}  y{r["yaw"]:+.0f} p{r["pitch"]:+.0f}'
            cv2.putText(frame, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("ClassSense", frame)
        if not getattr(main, "_shown", False):
            print("window shown, press q in it to quit")
            main._shown = True
        key = cv2.waitKey(1) & 0xFF
        frames_seen += 1
        if frames_seen < 60:
            key = 255  # ignore stray keys in the first 2 seconds
        if key in (ord("q"), 27):
            print("quit key pressed")
            break
    print("loop ended")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
