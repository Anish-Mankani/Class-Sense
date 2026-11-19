"""Analyse a recorded lesson video. Run from the project root:
    python3 analyze_video.py path/to/lesson.mp4
Samples one frame every --every seconds (default 2), so a 3-hour lesson is ~5,400 frames.
Saves data/analysis/<name>.json (timeline, events, per-student seconds) and a snapshot
image for every event in data/analysis/<name>/. Progress is saved as it goes; use
--resume to continue an interrupted run (student numbers may restart after a resume)."""
import argparse
import json
import time
from pathlib import Path

import cv2

from cv_modules.attention import AttentionEstimator, class_attention
from run_live import SimpleTracker
from server import annotate, assign_phones

MODEL = "cv_modules/models/face_detection_yunet_2023mar.onnx"
OUT = Path("data/analysis")


def clock(seconds):
    s = int(seconds)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between analysed frames")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds a cue must last")
    ap.add_argument("--down-hold", type=float, default=60.0, help="seconds head-down must last without a lift")
    ap.add_argument("--phone-model", default="phone_best2.pt")
    ap.add_argument("--phone-conf", type=float, default=0.70, help="phone confidence, 0-1")
    ap.add_argument("--no-phones", action="store_true")
    ap.add_argument("--no-mediapipe", action="store_true")
    ap.add_argument("--limit-minutes", type=float, default=0, help="stop after this many minutes (for tests)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)

    video = Path(a.video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    step = max(1, round(a.every * fps))
    every = step / fps                                  # real seconds between samples
    ok, first = cap.read()
    if not ok:
        raise SystemExit("The video has no readable frames.")
    h, w = first.shape[:2]

    detector = cv2.FaceDetectorYN.create(MODEL, "", (w, h), 0.6, 0.3, 5000)
    estimator = AttentionEstimator(hold_frames=max(1, round(a.hold / every)),
                                   down_seconds=a.down_hold, write_seconds=a.hold, lift_frames=max(1, round(4 / every)), smoothing=0.5,
                                   lost_hold_frames=max(1, round(a.hold / every)),
                                   forget_frames=max(5, round(60 / every)),
                                   lost_min_seen=max(2, round(6 / every)))
    tracker = SimpleTracker(max_dist=150, max_age=max(2, round(20 / every)), min_hits=2)
    mp_pitch = phone_detector = None
    if not a.no_mediapipe:
        try:
            from cv_modules.mp_pitch import MediaPipePitch
            mp_pitch = MediaPipePitch()
        except Exception as e:
            print("MediaPipe pitch disabled:", e)
    if not a.no_phones:
        try:
            from cv_modules.phone_detection import PhoneDetector
            phone_detector = PhoneDetector(a.phone_model, min_confidence=a.phone_conf)
        except Exception as e:
            print("Phone detection disabled:", e)

    OUT.mkdir(parents=True, exist_ok=True)
    result_file, snap_dir = OUT / f"{video.stem}.json", OUT / video.stem
    snap_dir.mkdir(exist_ok=True)
    timeline, events, students, start_t = [], [], {}, 0.0
    if a.resume and result_file.exists():
        old = json.loads(result_file.read_text())
        timeline, events, students = old["timeline"], old["events"], old["students"]
        start_t = old["processed_until"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_t * fps))
        print("Resuming from", clock(start_t))
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def save(t, finished):
        usable = [r["engaged"] for r in timeline if r["engaged"] is not None]
        result_file.write_text(json.dumps({
            "video": video.name, "duration_seconds": round(duration, 1), "every_seconds": round(every, 2),
            "processed_until": round(t, 1), "finished": finished,
            "summary": {"average_engagement": round(sum(usable) / len(usable), 1) if usable else None,
                        "events": len(events), "phone_events": sum(1 for e in events if e["type"] == "phone")},
            "timeline": timeline, "events": events, "students": students}, indent=1))

    prev_state, prev_phone, phone_score = {}, set(), {}
    frame_idx, count, started, t = int(start_t * fps), 0, time.time(), start_t
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps
            if a.limit_minutes and t - start_t > a.limit_minutes * 60:
                break
            _, dets = detector.detect(frame)
            faces = tracker.update(dets if dets is not None else [])
            if mp_pitch is not None:
                for f in faces:
                    f["mp_pitch"] = mp_pitch.pitch(frame, f["box"])
            results = estimator.update(faces, (w, h), now=t)
            phones = phone_detector.detect(frame) if phone_detector else []
            hit = assign_phones(phones, results)
            for fid in set(phone_score) | hit:
                sc = phone_score.get(fid, 0)
                phone_score[fid] = min(sc + 1, 6) if fid in hit else max(sc - 1, 0)
            phone_ids = {fid for fid, sc in phone_score.items() if sc >= 2}

            for r in results:
                sid = str(r["id"])
                st = students.setdefault(sid, {"attentive": 0, "looking away": 0, "head down": 0,
                                               "unknown": 0, "phone_events": 0})
                st[r["state"]] = round(st.get(r["state"], 0) + every, 1)
                new = []
                if r["state"] in ("looking away", "head down") and prev_state.get(r["id"]) != r["state"]:
                    new.append(r["state"])
                if r["id"] in phone_ids and r["id"] not in prev_phone:
                    new.append("phone")
                    st["phone_events"] += 1
                for kind in new:
                    name = f"{len(events) + 1:04d}_{kind.replace(' ', '_')}.jpg"
                    jpg = annotate(frame, results, phones)
                    if jpg:
                        (snap_dir / name).write_bytes(jpg)
                    events.append({"time": round(t, 1), "clock": clock(t), "student": r["id"],
                                   "type": kind, "snapshot": name})
                prev_state[r["id"]] = r["state"]
            prev_phone = phone_ids

            usable = [r for r in results if r["state"] != "unknown"]
            bad = sum(1 for r in usable if r["state"] not in ("attentive", "writing"))
            att = class_attention(results)
            timeline.append({"t": round(t, 1), "faces": len(results),
                             "engaged": None if att is None else round(att, 1),
                             "distracted": None if not usable else round(100 * bad / len(usable), 1),
                             "phones": len(phone_ids)})

            count += 1
            if count % 20 == 0:
                done = (t - start_t) / max(duration - start_t, 1)
                eta = (time.time() - started) / max(done, 1e-6) * (1 - done)
                print(f"{100 * done:5.1f}%  at {clock(t)} of {clock(duration)}  ETA {clock(eta)}")
            if count % 50 == 0:
                save(t, False)
            for _ in range(step - 1):
                cap.grab()
            frame_idx += step
    finally:
        save(t, t + every >= duration - 1)
        print(f"Saved {result_file}  ({len(timeline)} samples, {len(events)} events)")


if __name__ == "__main__":
    main()
