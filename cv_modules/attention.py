import json
import time
from pathlib import Path
 
from .head_pose import eye_distance, head_pose
 
CALIBRATION_FILE = Path("data/pose_calibration.json")
DEFAULTS = {"yaw_limit": 30.0, "pitch_down_limit": 25.0, "pitch_offset": 0.0,
            "pitch_up_limit": 20.0,
            "mp_front_pitch": 0.0}
 
 
def load_settings():
    settings = dict(DEFAULTS)
    if CALIBRATION_FILE.exists():
        try:
            settings.update(json.loads(CALIBRATION_FILE.read_text()))
        except ValueError:
            pass
    return settings
 
 
class AttentionEstimator:
    def __init__(self, yaw_limit=None, pitch_down_limit=None, pitch_offset=None,
                 pitch_up_limit=None, hold_frames=30, down_hold_frames=None, smoothing=0.25, min_eye_distance=14,
                 mp_front_pitch=None, mp_min_eye=30, lost_hold_frames=45, forget_frames=600, lost_min_seen=60,
                 down_seconds=60.0, write_seconds=3.0, lift_frames=20):
        settings = load_settings()
        self.yaw_limit = settings["yaw_limit"] if yaw_limit is None else yaw_limit
        self.pitch_down_limit = settings["pitch_down_limit"] if pitch_down_limit is None else pitch_down_limit
        self.pitch_offset = settings["pitch_offset"] if pitch_offset is None else pitch_offset
        self.pitch_up_limit = settings["pitch_up_limit"] if pitch_up_limit is None else pitch_up_limit
        self.hold_frames = hold_frames          # frames a cue must last (30 = about 1 second)
        # frames "head down" must last; defaults to hold_frames. Use a larger value
        # (e.g. 150-300) so students who are writing or reading are not flagged.
        self.down_hold_frames = hold_frames if down_hold_frames is None else down_hold_frames
        self.smoothing = smoothing              # 0.25 = new angle counts 25%, old 75% (less jitter)
        self.min_eye_distance = min_eye_distance  # pixels; smaller faces are too tiny for a good angle
        self.mp_front_pitch = settings["mp_front_pitch"] if mp_front_pitch is None else mp_front_pitch
        self.mp_min_eye = mp_min_eye   # below this eye distance (px) a MediaPipe miss is not trusted
        self.lost_hold_frames = lost_hold_frames   # frames a face may vanish before it counts as looking away
        self.forget_frames = forget_frames         # frames after which a vanished face is dropped
        self.lost_min_seen = lost_min_seen         # frames a face must have been seen before it can count as lost
        self.down_seconds = down_seconds     # head down this long without a lift = Head down (distracted)
        self.write_seconds = write_seconds   # head down this long = Writing / reading (counts as engaged)
        self.lift_frames = lift_frames       # frames with the head up before the head-down timer restarts
        self.people = {}
 
    def update(self, faces, frame_size=(1280, 720), now=None):
        """faces = tracker output (id, box, landmarks). Returns one dict per face:
        id, box, yaw, pitch, state ('attentive', 'looking away', 'head down' or 'unknown')."""
        t = time.monotonic() if now is None else now
        results, seen = [], set()
        for face in faces:
            fid = face["id"]
            seen.add(fid)
            memory = self.people.setdefault(fid, {"yaw": None, "pitch": None, "cue": "front", "count": 0, "missing": 0})
            memory["missing"] = 0
            memory["box"] = face["box"]
            memory["seen"] = memory.get("seen", 0) + 1
            landmarks = face.get("landmarks")
            pose = None
            if landmarks is not None and len(landmarks) > 0 and eye_distance(landmarks) >= self.min_eye_distance:
                pose = head_pose(landmarks, frame_size)
            if (pose is not None and "mp_pitch" in face and face["mp_pitch"] is None
                    and eye_distance(landmarks) < self.mp_min_eye):
                pose = None   # face too small to trust a MediaPipe miss
            if pose is None:
                results.append({"id": fid, "box": face["box"], "yaw": None, "pitch": None, "state": "unknown"})
                continue
            yaw = pose[0]
            if "mp_pitch" in face:
                if face["mp_pitch"] is None:
                    # YuNet sees a face but MediaPipe cannot: a deep nod hides the face.
                    pitch = self.pitch_down_limit + 5.0
                else:
                    pitch = -(face["mp_pitch"] - self.mp_front_pitch)   # down positive, relative to front
            else:
                pitch = pose[1] - self.pitch_offset
            a = self.smoothing
            memory["yaw"] = yaw if memory["yaw"] is None else a * yaw + (1 - a) * memory["yaw"]
            memory["pitch"] = pitch if memory["pitch"] is None else a * pitch + (1 - a) * memory["pitch"]
 
            if abs(memory["yaw"]) > self.yaw_limit:
                cue = "looking away"
            elif memory["pitch"] > self.pitch_down_limit:
                cue = "head down"
            elif memory["pitch"] < -self.pitch_up_limit:
                cue = "looking away"   # head tipped up
            else:
                cue = "front"
            if cue == "front":
                memory["count"] = 0
            else:
                memory["count"] = memory["count"] + 1 if cue == memory["cue"] else 1
            memory["cue"] = cue
            if cue == "head down":
                memory["up_count"] = 0
                if memory.get("down_since") is None:
                    memory["down_since"] = t
            else:
                memory["up_count"] = memory.get("up_count", 0) + 1
                if memory["up_count"] >= self.lift_frames:
                    memory["down_since"] = None   # the head came up: writing students do this often
            if memory.get("down_since") is not None and cue != "looking away":
                down_for = t - memory["down_since"]
                if down_for >= self.down_seconds:
                    state = "head down"
                elif down_for >= self.write_seconds:
                    state = "writing"
                else:
                    state = "attentive"
            else:
                state = cue if cue != "front" and memory["count"] >= self.hold_frames else "attentive"
            results.append({"id": fid, "box": face["box"], "yaw": memory["yaw"], "pitch": memory["pitch"], "state": state})
 
        for fid in list(self.people):
            if fid in seen:
                continue
            memory = self.people[fid]
            memory["missing"] += 1
            if memory["missing"] > self.forget_frames:
                del self.people[fid]
            elif (memory["missing"] >= self.lost_hold_frames and memory.get("box") is not None
                  and memory.get("seen", 0) >= self.lost_min_seen):
                x, y, w, h = memory["box"]
                reach = max(w, h)
                if any(abs(f["box"][0] + f["box"][2] / 2 - (x + w / 2)) < reach
                       and abs(f["box"][1] + f["box"][3] / 2 - (y + h / 2)) < reach for f in faces):
                    del self.people[fid]   # the same person is back under a new id: no ghost box
                else:
                    # The camera cannot see this student any more. If they were head down, they are probably
                    # bent lower over the desk, so the head-down timer keeps running; otherwise they turned away.
                    if memory.get("down_since") is not None:
                        down_for = t - memory["down_since"]
                        lost_state = "head down" if down_for >= self.down_seconds else "writing"
                    else:
                        lost_state = "looking away"
                    results.append({"id": fid, "box": memory["box"], "yaw": None, "pitch": None, "state": lost_state})
        return results
 
 
def class_attention(results):
    """Percent of the faces with a usable angle that are attentive, or None if there are none."""
    usable = [r for r in results if r["state"] != "unknown"]
    if not usable:
        return None
    return 100.0 * sum(1 for r in usable if r["state"] in ("attentive", "writing")) / len(usable)
 
 
def suggest_limits(samples, minimum=15):
    """Work out good limits from calibration clips.
    samples = {"front": [(yaw, pitch), ...], "left": [...], "right": [...], "down": [...]}
    Returns a dict with only the settings that could be worked out."""
    from statistics import median
 
    def clamp(value, low, high):
        return max(low, min(high, value))
 
    front, left, right, down, up = (samples.get(k, []) for k in ("front", "left", "right", "down", "up"))
    settings = {}
    offset = 0.0
    if len(front) >= minimum:
        offset = median(p for _, p in front)
        settings["pitch_offset"] = round(offset, 1)
    if min(len(front), len(left), len(right)) >= minimum:
        straight = median(abs(y) for y, _ in front)
        turned = min(median(abs(y) for y, _ in left), median(abs(y) for y, _ in right))
        settings["yaw_limit"] = round(clamp((straight + turned) / 2, 15, 45), 1)
    if len(down) >= minimum:
        settings["pitch_down_limit"] = round(clamp((median(p for _, p in down) - offset) / 2, 10, 35), 1)
    if len(up) >= minimum:
        settings["pitch_up_limit"] = round(clamp((offset - median(p for _, p in up)) / 2, 10, 35), 1)
    return settings
 