import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project folder: lets this file import cv_modules, backend, client

"""Look inside a downloaded dataset zip before you use it.
Run:  python peek_dataset.py ~/Downloads/Calculator.v1i.yolov8.zip
Prints the class names and image counts, and saves a picture sheet (peek.jpg) of 12 random
images with their boxes, so you can check that NO PHONE appears in a negative set."""
import random
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

if len(sys.argv) < 2:
    raise SystemExit("Give the zip file, for example: python peek_dataset.py ~/Downloads/Calculator.v1i.yolov8.zip")

zip_path = Path(sys.argv[1]).expanduser()
folder = Path(tempfile.mkdtemp())
with zipfile.ZipFile(zip_path) as z:
    z.extractall(folder)

yaml_files = list(folder.rglob("data.yaml"))
if yaml_files:
    names = re.search(r"names:\s*(.*)", yaml_files[0].read_text())
    print("classes:", names.group(1).strip() if names else "?")
images = [p for p in folder.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.parent.name == "images"]
for split in ["train", "valid", "test"]:
    print(f"{split}: {sum(1 for p in images if p.parent.parent.name == split)} images")

tiles = []
for path in random.sample(images, min(12, len(images))):
    img = cv2.imread(str(path))
    if img is None:
        continue
    h, w = img.shape[:2]
    label = path.parent.parent / "labels" / (path.stem + ".txt")
    if label.exists():
        for line in label.read_text().splitlines():
            v = list(map(float, line.split()))[1:]
            if len(v) == 4:
                x, y, bw, bh = v
                cv2.rectangle(img, (int((x - bw / 2) * w), int((y - bh / 2) * h)),
                              (int((x + bw / 2) * w), int((y + bh / 2) * h)), (0, 0, 255), max(2, w // 150))
    tiles.append(cv2.resize(img, (320, 240)))
while len(tiles) % 4:
    tiles.append(np.zeros((240, 320, 3), np.uint8))
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
cv2.imwrite("peek.jpg", np.vstack(rows))
print("Saved peek.jpg")
try:
    subprocess.run(["open", "peek.jpg"], check=False)   # opens the picture on a Mac
except FileNotFoundError:
    pass
