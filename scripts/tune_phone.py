"""Tune phone detection on YOUR test videos. Run from the project root:
    python3 scripts/tune_phone.py
Videos directly inside data/: a name containing 'phone' (but not starting with 'no') counts as
"a phone is present"; everything else (calculator, charger, negatives, no_phone...) as "no phone".
Each model is run once at a low confidence, then every confidence threshold is scored.
Options:  --every N (analyse every Nth frame, default 3)   --max-seconds S (per video, default 40)
          --models a.pt b.pt   --sizes 640 960   --run K (consecutive frames that make an alert, default 3)"""
import argparse
import csv
import sys
from pathlib import Path
from statistics import mean

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONFS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
VIDEO_TYPES = (".mp4", ".mov", ".avi", ".m4v")


def is_positive(path):
    name = path.stem.lower()
    return "phone" in name and not name.startswith("no")


def has_run(flags, k):
    run = 0
    for f in flags:
        run = run + 1 if f else 0
        if run >= k:
            return True
    return False


def evaluate(results, confs, k):
    """results: {video: (is_positive, [best phone score per analysed frame])}."""
    rows = []
    for conf in confs:
        pos, neg = [], []
        for positive, scores in results.values():
            flags = [s >= conf for s in scores]
            rate = sum(flags) / max(len(flags), 1)
            (pos if positive else neg).append((rate, has_run(flags, k)))
        rows.append({"conf": conf,
                     "pos_rate": mean(r for r, _ in pos) if pos else 0.0,
                     "caught": sum(1 for _, a in pos if a), "n_pos": len(pos),
                     "neg_rate": mean(r for r, _ in neg) if neg else 0.0,
                     "false_alerts": sum(1 for _, a in neg if a), "n_neg": len(neg)})
    return rows


def best_row(rows):
    """Catch every phone video with no false alert if possible, then the highest frame rate."""
    ok = [r for r in rows if r["false_alerts"] == 0 and r["caught"] == r["n_pos"]]
    pool = ok or rows
    return max(pool, key=lambda r: (r["caught"] - r["false_alerts"], r["pos_rate"] - 3 * r["neg_rate"]))


def collect(detector, path, every, max_seconds, save_dir=None, save_min=0.3):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    scores, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or i / fps > max_seconds:
            break
        if i % every == 0:
            best = max((p["score"] for p in detector.detect(frame)), default=0.0)
            scores.append(best)
            if save_dir is not None and best >= save_min:
                (save_dir / "images").mkdir(parents=True, exist_ok=True)
                (save_dir / "labels").mkdir(parents=True, exist_ok=True)
                name = f"{path.stem}_{i:06d}"
                cv2.imwrite(str(save_dir / "images" / f"{name}.jpg"), frame)
                (save_dir / "labels" / f"{name}.txt").write_text("")   # empty label = no phone here
        i += 1
    cap.release()
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=40)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--sizes", nargs="*", type=int, default=[640, 960])
    ap.add_argument("--run", type=int, default=3)
    ap.add_argument("--detail", action="store_true", help="show each video separately")
    ap.add_argument("--save-negatives", action="store_true",
                    help="save false-alarm frames from no-phone videos to data/hard_negatives for retraining")
    a = ap.parse_args()
    from cv_modules.phone_detection import PhoneDetector

    videos = sorted(p for p in Path("data").iterdir() if p.suffix.lower() in VIDEO_TYPES)
    if not videos:
        raise SystemExit("No videos found directly inside data/.")
    print("Videos:")
    for v in videos:
        print(f"  {'PHONE   ' if is_positive(v) else 'no phone'}  {v.name}")
    models = a.models or sorted(p.name for p in Path("models").glob("*.pt"))
    table, best_overall = [], None
    for model in models:
        for size in a.sizes:
            try:
                det = PhoneDetector(model, min_confidence=0.10, image_size=size)
            except Exception as e:
                print(f"\nSkipping {model}: {e}")
                break
            print(f"\n{model}  size {size}: analysing...", flush=True)
            neg_dir = Path("data/hard_negatives") if a.save_negatives else None
            results = {v.name: (is_positive(v), collect(det, v, a.every, a.max_seconds,
                                                        None if is_positive(v) else neg_dir)) for v in videos}
            skipped = [k for k, v in results.items() if not v[1]]
            if skipped:
                print("  Skipping unreadable videos:", ", ".join(skipped))
            results = {k: v for k, v in results.items() if v[1]}
            rows = evaluate(results, CONFS, a.run)
            print(f"  conf  phones-found  caught   false-frames  false-alerts")
            for r in rows:
                print(f"  {r['conf']:.2f}  {100 * r['pos_rate']:9.0f}%  {r['caught']}/{r['n_pos']}"
                      f"   {100 * r['neg_rate']:10.1f}%   {r['false_alerts']}/{r['n_neg']}")
                table.append({"model": model, "size": size, **r})
            if a.detail:
                cols = (0.60, 0.70, 0.75, 0.80, 0.85)
                print("  per video (frames flagged; * = would raise an alert):")
                print("    " + " " * 29 + "  ".join(f"{c:.2f}" + " " * 2 for c in cols))
                for name, (positive, scores) in results.items():
                    cells = []
                    for c in cols:
                        flags = [x >= c for x in scores]
                        cells.append(f"{100 * sum(flags) / max(len(flags), 1):4.0f}%" + ("*" if has_run(flags, a.run) else " "))
                    print(f"    {'PHONE' if positive else 'none ':5} {name:<22}" + "  ".join(cells))
            b = best_row(rows)
            score = (b["caught"] - b["false_alerts"], b["pos_rate"] - 3 * b["neg_rate"])
            if best_overall is None or score > best_overall[0]:
                best_overall = (score, model, size, b)
    if table:
        with open("data/phone_tuning.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        _, model, size, b = best_overall
        print(f"\nBEST: {model}  size {size}  confidence {b['conf']:.2f}  "
              f"(found {100 * b['pos_rate']:.0f}% of phone frames, caught {b['caught']}/{b['n_pos']} phone videos, "
              f"{b['false_alerts']}/{b['n_neg']} false alerts)")
        print("Saved data/phone_tuning.csv")


if __name__ == "__main__":
    main()
