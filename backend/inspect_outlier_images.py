
import cv2
import csv
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("dataset/debris_samples/train")
IMAGE_DIR = BASE / "images"

REPORT = Path(
    "results/quality_analysis/debris_quality_report.csv"
)

ANNOTATIONS = BASE / "annotations.csv"

OUTPUT = Path("results/outlier_inspection")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(REPORT)

# Select unusual images
selected = pd.concat([
    df.nlargest(5, "brightness"),
    df.nlargest(5, "contrast"),
    df.nlargest(5, "bbox_area_percent")
])

selected = selected.drop_duplicates("filename")

annotation_map = {}

with open(ANNOTATIONS, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        annotation_map[row["filename"]] = row

cards = []

for _, record in selected.iterrows():

    filename = record["filename"]
    image_path = IMAGE_DIR / filename

    image = cv2.imread(str(image_path))

    if image is None:
        print("Missing:", filename)
        continue

    row = annotation_map[filename]

    x1, y1, x2, y2 = map(
        int, json.loads(row["bbox"])
    )

    cv2.rectangle(
        image, (x1, y1), (x2, y2),
        (0, 255, 0), 3
    )

    cv2.putText(
        image,
        "DEBRIS",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2
    )

    image = cv2.resize(image, (300, 300))

    info = (
        f"B:{record['brightness']:.1f} "
        f"C:{record['contrast']:.1f}"
    )

    card = np.zeros((345, 320, 3), dtype=np.uint8)

    card[0:300, 10:310] = image

    cv2.putText(
        card, filename, (10, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (255, 255, 255), 1
    )

    cv2.putText(
        card, info, (10, 339),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (255, 255, 255), 1
    )

    cards.append(card)

columns = 3
rows = int(np.ceil(len(cards) / columns))

sheet = np.zeros(
    (rows * 345, columns * 320, 3),
    dtype=np.uint8
)

for i, card in enumerate(cards):
    x = (i % columns) * 320
    y = (i // columns) * 345

    sheet[y:y+345, x:x+320] = card

output_path = OUTPUT / "outlier_contact_sheet.jpg"

cv2.imwrite(str(output_path), sheet)

print("Unique images inspected:", len(cards))
print("Saved:", output_path.resolve())
