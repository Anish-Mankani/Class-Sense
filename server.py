"""ClassSense live server: camera -> YuNet -> attention -> WebSocket -> dashboard.
Run from the project root:   python3 server.py        then open http://localhost:8000
No camera? Try:               python3 server.py --demo
Phone alerts and roster matching are placeholders until those modules are connected."""
import argparse
import asyncio
import json
import os
import random
import threading
import time
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from cv_modules.attention import AttentionEstimator, class_attention
from run_live import SimpleTracker

MODEL = "cv_modules/models/face_detection_yunet_2023mar.onnx"
PAGE = Path("web/dashboard.html")
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET"], allow_headers=["*"])
ANALYSIS = Path("data/analysis")
ANALYSIS.mkdir(parents=True, exist_ok=True)
app.mount("/analysis-files", StaticFiles(directory=ANALYSIS), name="analysis-files")
STATE = {"msg": {"faces": [], "class_attention": None}}


COLORS = {"attentive": (0, 200, 0), "looking away": (0, 140, 255),
          "head down": (0, 0, 255), "unknown": (140, 140, 140),
          "writing": (230, 130, 40)}  # BGR


def annotate(frame, results, phones=()):
    """Draw the coloured boxes and return a JPEG (smaller copy for streaming)."""
    img = frame.copy()
    for r in results:
        x, y, bw, bh = r["box"]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), COLORS[r["state"]], 3)
    for ph in phones:
        x, y, bw, bh = ph["box"]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(img, "Phone", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    scale = 960 / img.shape[1]
    img = cv2.resize(img, (960, int(img.shape[0] * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes() if ok else None


def assign_phones(phones, results, max_faces_away=5.0):
    """Give each phone to the nearest student face. Returns the set of student ids."""
    hit = set()
    for ph in phones:
        px, py = ph["box"][0] + ph["box"][2] / 2, ph["box"][1] + ph["box"][3] / 2
        best, best_d = None, None
        for r in results:
            x, y, w, h = r["box"]
            d = ((px - (x + w / 2)) ** 2 + (py - (y + h / 2)) ** 2) ** 0.5
            if d <= max_faces_away * max(w, h) and (best_d is None or d < best_d):
                best, best_d = r["id"], d
        if best is not None:
            hit.add(best)
    return hit


def to_message(results, phone_ids=()):
    faces = [{"id": r["id"], "name": f"Student {r['id']}", "state": r["state"],
              "yaw": None if r["yaw"] is None else float(r["yaw"]),
              "pitch": None if r["pitch"] is None else float(r["pitch"]),
              "phone": r["id"] in phone_ids,
              "roster": "verified"}    # TODO: connect roster verification
             for r in results]
    return {"faces": faces, "class_attention": class_attention(results)}


def camera_loop(camera_index, use_mp, phone_model=None, phone_conf=0.7):
    cap = cv2.VideoCapture(camera_index)
    ok, frame = cap.read()
    if not ok:
        print("Could not read from the camera.")
        return
    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(MODEL, "", (w, h), 0.6, 0.3, 5000)
    tracker, estimator, mp_pitch = SimpleTracker(), AttentionEstimator(down_seconds=float(os.environ.get("CLASSSENSE_DOWN_SECONDS", 60))), None
    if use_mp:
        try:
            from cv_modules.mp_pitch import MediaPipePitch
            mp_pitch = MediaPipePitch()
        except Exception as e:
            print("MediaPipe pitch disabled:", e)
    phone_detector, phones, phone_score, frame_no = None, [], {}, 0
    if phone_model:
        try:
            from cv_modules.phone_detection import PhoneDetector
            phone_detector = PhoneDetector(phone_model, min_confidence=phone_conf)
        except Exception as e:
            print("Phone detection disabled:", e)
    print("Camera running.")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        _, dets = detector.detect(frame)
        faces = tracker.update(dets if dets is not None else [])
        if mp_pitch is not None:
            for f in faces:
                f["mp_pitch"] = mp_pitch.pitch(frame, f["box"])
        results = estimator.update(faces, (w, h))
        frame_no += 1
        if phone_detector is not None and frame_no % 3 == 0:   # every 3rd frame keeps video smooth
            phones = phone_detector.detect(frame)
            hit = assign_phones(phones, results)
            for fid in set(phone_score) | hit:
                score = phone_score.get(fid, 0)
                phone_score[fid] = min(score + 1, 6) if fid in hit else max(score - 1, 0)
        phone_ids = {fid for fid, sc in phone_score.items() if sc >= 2}
        STATE["msg"] = to_message(results, phone_ids)
        STATE["jpg"] = annotate(frame, results, phones)


def demo_loop():
    states = ["attentive"] * 12
    while True:
        results = []
        for i in range(12):
            r = random.random()
            if states[i] == "attentive":
                states[i] = "looking away" if r < .03 else "head down" if r < .045 else "attentive"
            elif r < .15:
                states[i] = "attentive"
            results.append({"id": i + 1, "state": states[i],
                            "yaw": random.uniform(-6, 6) if states[i] != "looking away" else 35.0,
                            "pitch": random.uniform(-4, 4) if states[i] != "head down" else 20.0})
        STATE["msg"] = to_message(results)
        time.sleep(1)


@app.get("/")
def index():
    if PAGE.exists():
        return FileResponse(PAGE)
    return {"status": "ClassSense server running", "dashboard": "http://localhost:5173",
            "endpoints": ["/ws", "/video", "/api/state"]}


@app.get("/video")
def video():
    def frames():
        while True:
            jpg = STATE.get("jpg")
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.07)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/state")
def state():
    return STATE["msg"]


@app.get("/api/analyses")
def list_analyses():
    out = []
    for f in sorted(ANALYSIS.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            d = json.loads(f.read_text())
        except ValueError:
            continue
        out.append({"name": f.stem, "video": d.get("video"), "duration_seconds": d.get("duration_seconds"),
                    "processed_until": d.get("processed_until"), "finished": d.get("finished"),
                    "summary": d.get("summary")})
    return out


@app.get("/api/analyses/{name}")
def get_analysis(name: str):
    f = ANALYSIS / (Path(name).name + ".json")
    if not f.exists():
        raise HTTPException(404, "Analysis not found")
    return json.loads(f.read_text())


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    try:
        while True:
            await sock.send_text(json.dumps(STATE["msg"]))
            await asyncio.sleep(0.5)
    except Exception:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="stream fake students, no camera")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-mediapipe", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--phone-model", default="phone_best2.pt", help="phone model file")
    ap.add_argument("--phone-conf", type=float, default=0.70, help="phone confidence, 0-1")
    ap.add_argument("--no-phones", action="store_true")
    a = ap.parse_args()
    target = demo_loop if a.demo else (lambda: camera_loop(a.camera, not a.no_mediapipe, None if a.no_phones else a.phone_model, a.phone_conf))
    threading.Thread(target=target, daemon=True).start()
    print(f"Open http://localhost:{a.port}")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
