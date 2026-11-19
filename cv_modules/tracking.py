"""Face tracking: gives each face a steady ID and hides one-frame false faces.
A face is shown only after it has been seen in several frames (min_hits).
A real face stays in place for many frames; a false one flickers and vanishes.
"""


def overlap(a, b):
    """How much two boxes (x, y, w, h) overlap, from 0 (none) to 1 (same box)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    def __init__(self, min_hits=5, max_missed=10, min_overlap=0.3):
        self.min_hits = min_hits        # frames a face must be seen before it counts
        self.max_missed = max_missed    # frames a face may vanish before its ID is dropped
        self.min_overlap = min_overlap
        self.tracks = []
        self.next_id = 1

    def update(self, faces):
        """Give this frame's detections to the tracker.
        Returns only the confirmed faces, each with a steady "id"."""
        # match existing tracks to new detections, best overlap first
        pairs = sorted(
            ((overlap(t["box"], f["box"]), ti, fi)
             for ti, t in enumerate(self.tracks) for fi, f in enumerate(faces)),
            reverse=True,
        )
        used_tracks, used_faces = set(), set()
        for score, ti, fi in pairs:
            if score < self.min_overlap:
                break
            if ti in used_tracks or fi in used_faces:
                continue
            used_tracks.add(ti)
            used_faces.add(fi)
            track = self.tracks[ti]
            track.update(box=faces[fi]["box"], score=faces[fi]["score"], missed=0,
                         landmarks=faces[fi].get("landmarks"))
            track["hits"] += 1

        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track["missed"] += 1
        for fi, face in enumerate(faces):
            if fi not in used_faces:
                self.tracks.append({"id": self.next_id, "box": face["box"],
                                    "score": face["score"], "hits": 1, "missed": 0,
                                    "landmarks": face.get("landmarks")})
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]
        return [{"id": t["id"], "box": t["box"], "score": t["score"], "landmarks": t.get("landmarks")}
                for t in self.tracks if t["hits"] >= self.min_hits and t["missed"] == 0]
