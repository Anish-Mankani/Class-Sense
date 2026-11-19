"""Pick a balanced set of hard-negative frames and zip them for Google Colab.
Run from the project root:
    python3 scripts/prepare_hard_negatives.py            # at most 80 frames per video
    python3 scripts/prepare_hard_negatives.py --per-video 120
Reads data/hard_negatives/{images,labels} (made by tune_phone.py --save-negatives) and writes
data/hard_negatives_for_colab.zip with images/ and labels/ inside. Labels are empty files,
which in YOLO format means 'there is no phone in this picture'."""
import argparse
import zipfile
from collections import defaultdict
from pathlib import Path

SRC = Path("data/hard_negatives")
OUT = Path("data/hard_negatives_for_colab.zip")


def pick(items, n):
    """n items spread evenly through the list (neighbouring frames are near-duplicates)."""
    if n <= 1:
        return items[:1]
    if len(items) <= n:
        return items
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-video", type=int, default=80)
    a = ap.parse_args()
    images = sorted((SRC / "images").glob("*.jpg"))
    if not images:
        raise SystemExit("No images in data/hard_negatives/images. Run tune_phone.py with --save-negatives first.")
    by_video = defaultdict(list)
    for p in images:
        by_video[p.stem.rsplit("_", 1)[0]].append(p)
    chosen = []
    print(f"{'video':<24}{'wrong frames':>13}{'kept':>7}")
    for video, items in sorted(by_video.items()):
        keep = pick(items, a.per_video)
        chosen += keep
        print(f"{video:<24}{len(items):>13}{len(keep):>7}")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in chosen:
            z.write(p, f"images/{p.name}")
            z.writestr(f"labels/{p.stem}.txt", "")
    print(f"\nWrote {OUT}: {len(chosen)} images, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
