
import cv2
import csv
import json
from pathlib import Path
import math

BASE = Path("dataset/debris_samples/train")

IMAGE_DIR = BASE / "images"
ANNOTATION_FILE = BASE / "annotations.csv"

OUTPUT_DIR = Path("results/debris_visualization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

annotated_images = []

valid_count = 0
invalid_count = 0

with open(ANNOTATION_FILE, "r", newline="", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        filename = row["filename"]
        image_path = IMAGE_DIR / filename

        image = cv2.imread(str(image_path))

        if image is None:
            print("Image missing:", filename)
            invalid_count += 1
            continue

        height, width = image.shape[:2]

        bbox = json.loads(row["bbox"])
        x1, y1, x2, y2 = map(int, bbox)

        # Validate bounding box
        if not (
            0 <= x1 < x2 <= width
            and 0 <= y1 < y2 <= height
        ):
            print("Invalid bounding box:", filename, bbox)
            invalid_count += 1
            continue

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            image,
            "DEBRIS",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Save individual annotated image
        output_path = OUTPUT_DIR / filename
        cv2.imwrite(str(output_path), image)

        annotated_images.append((filename, image))
        valid_count += 1

print("\nAnnotation validation completed!")
print("Valid images:", valid_count)
print("Invalid images:", invalid_count)

# Generate contact sheet
if annotated_images:

    thumb_width = 320
    thumb_height = 320
    columns = 5
    rows = math.ceil(len(annotated_images) / columns)

    sheet = 30 * (rows + 1) * 0 + 255
    sheet = __import__("numpy").zeros(
        (rows * 360, columns * 340, 3),
        dtype="uint8"
    )

    for index, (filename, image) in enumerate(annotated_images):

        image = cv2.resize(
            image,
            (thumb_width, thumb_height)
        )

        x = (index % columns) * 340
        y = (index // columns) * 360

        sheet[y:y+thumb_height, x:x+thumb_width] = image

        cv2.putText(
            sheet,
            filename,
            (x + 5, y + 345),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    contact_path = OUTPUT_DIR / "debris_contact_sheet.jpg"

    cv2.imwrite(str(contact_path), sheet)

    print("Contact sheet saved:", contact_path.resolve())
