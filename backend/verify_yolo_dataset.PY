
import cv2
import numpy as np
from pathlib import Path
import random

ROOT = Path("dataset/yolo_debris")

SPLITS = ["train", "validation", "test"]

OUTPUT = Path("results/yolo_verification")
OUTPUT.mkdir(parents=True, exist_ok=True)

total_valid = 0
total_invalid = 0

for split in SPLITS:

    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    previews = []

    print(f"\nChecking {split}...")

    for image_path in images:

        label_path = label_dir / (image_path.stem + ".txt")

        image = cv2.imread(str(image_path))

        if image is None or not label_path.exists():
            total_invalid += 1
            print("Missing image or label:", image_path.name)
            continue

        height, width = image.shape[:2]

        lines = label_path.read_text().strip().splitlines()

        valid = True

        for line in lines:

            values = line.split()

            if len(values) != 5:
                valid = False
                break

            cls, xc, yc, bw, bh = map(float, values)

            if cls != 0:
                valid = False
                break

            if not all(0 <= v <= 1 for v in [xc, yc, bw, bh]):
                valid = False
                break

            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)

            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            if not (0 <= x1 < x2 <= width and
                    0 <= y1 < y2 <= height):
                valid = False
                break

            cv2.rectangle(
                image, (x1, y1), (x2, y2),
                (0, 255, 0), 3
            )

            cv2.putText(
                image, "DEBRIS",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

        if valid:
            total_valid += 1
            previews.append((image_path.name, image))
        else:
            total_invalid += 1
            print("Invalid annotation:", image_path.name)

    # Save a contact sheet for each split
    if previews:

        random.seed(42)
        selected = random.sample(
            previews,
            min(25, len(previews))
        )

        cols = 5
        rows = (len(selected) + cols - 1) // cols

        sheet = np.zeros(
            (rows * 340, cols * 340, 3),
            dtype=np.uint8
        )

        for i, (filename, image) in enumerate(selected):

            thumb = cv2.resize(image, (320, 300))

            x = (i % cols) * 340
            y = (i // cols) * 340

            sheet[y:y+300, x:x+320] = thumb

            cv2.putText(
                sheet, filename,
                (x + 5, y + 325),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

        output_path = OUTPUT / f"{split}_contact_sheet.jpg"

        cv2.imwrite(str(output_path), sheet)

        print("Contact sheet:", output_path)

print("\n========== VERIFICATION SUMMARY ==========")
print("Valid annotations:", total_valid)
print("Invalid annotations:", total_invalid)

if total_invalid == 0 and total_valid == 150:
    print("SUCCESS: YOLO LABELS VERIFIED!")
else:
    print("WARNING: Please investigate invalid samples.")
