
import cv2
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("dataset/debris_samples/train")
IMAGE_DIR = BASE / "images"
CSV_FILE = BASE / "annotations.csv"

records = []

with open(CSV_FILE, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        image_path = IMAGE_DIR / row["filename"]

        image = cv2.imread(str(image_path))

        if image is None:
            print("Missing image:", row["filename"])
            continue

        height, width = image.shape[:2]

        x1, y1, x2, y2 = json.loads(row["bbox"])

        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height

        image_area = width * height

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        records.append({
            "filename": row["filename"],
            "width": width,
            "height": height,
            "bbox_width": box_width,
            "bbox_height": box_height,
            "bbox_area": box_area,
            "bbox_area_percent": box_area / image_area * 100,
            "brightness": np.mean(gray),
            "contrast": np.std(gray)
        })

df = pd.DataFrame(records)

print("\n========== DATASET QUALITY REPORT ==========")

print("\nTotal valid images:", len(df))

print("\nBounding Box Statistics:")
print(df[[
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "bbox_area_percent"
]].describe().round(3))

print("\nImage Quality Statistics:")
print(df[[
    "brightness",
    "contrast"
]].describe().round(3))

output_dir = Path("results/quality_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(
    output_dir / "debris_quality_report.csv",
    index=False
)

print("\nReport saved successfully!")
