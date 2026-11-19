import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project folder: lets this file import cv_modules, backend, client

"""Turn a clip of NON-phone objects into training images with empty labels.
Run:  python make_negatives.py
Needs data/negatives.mp4 (record it with: python record_clip.py negatives 40)
Creates negatives.zip, ready to upload to Colab together with the other zips."""
import shutil
import zipfile
from pathlib import Path

import cv2

STEP = 2  # keep every 2nd frame
video = cv2.VideoCapture("data/negatives.mp4")
if not video.isOpened():
    raise SystemExit("data/negatives.mp4 not found. Record it first.")

root = Path("data/negatives_set")
if root.exists():
    shutil.rmtree(root)
(root / "train/images").mkdir(parents=True)
(root / "train/labels").mkdir(parents=True)

index = saved = 0
while True:
    ok, frame = video.read()
    if not ok:
        break
    if index % STEP == 0:
        name = f"neg_{saved:04d}"
        cv2.imwrite(str(root / "train/images" / f"{name}.jpg"), frame)
        (root / "train/labels" / f"{name}.txt").write_text("")  # empty = "no phone here"
        saved += 1
    index += 1
video.release()
(root / "data.yaml").write_text("names: []\nnc: 0\n")

with zipfile.ZipFile("negatives.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in root.rglob("*"):
        if f.is_file():
            z.write(f, f.relative_to(root))
print(f"Saved {saved} images into negatives.zip")
