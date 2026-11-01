
import csv
import cv2
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_ROOT = PROJECT_ROOT / "dataset" / "debris_samples"
OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "yolo_debris"

SPLITS = ["train", "validation", "test"]

CLASS_ID = 0

total_images = 0
total_labels = 0
errors = []

for split in SPLITS:

    print(f"\n========== {split.upper()} ==========")

    source_images = SOURCE_ROOT / split / "images"
    annotation_file = SOURCE_ROOT / split / "annotations.csv"

    target_images = OUTPUT_ROOT / "images" / split
    target_labels = OUTPUT_ROOT / "labels" / split

    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    with open(
        annotation_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            filename = row["filename"].strip()

            image_path = source_images / filename

            if not image_path.exists():
                errors.append(f"Missing image: {filename}")
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                errors.append(f"Unreadable image: {filename}")
                continue

            height, width = image.shape[:2]

            # Read original bounding box
            bbox = [
                int(value.strip())
                for value in row["bbox"].strip("[]").split(",")
            ]

            if len(bbox) != 4:
                errors.append(f"Invalid bbox: {filename}")
                continue

            x1, y1, x2, y2 = bbox

            # Validate coordinates
            if not (
                0 <= x1 < x2 <= width
                and 0 <= y1 < y2 <= height
            ):
                errors.append(f"Invalid coordinates: {filename}")
                continue

            # Convert to YOLO format
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height

            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height

            label_content = (
                f"{CLASS_ID} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}\n"
            )

            # Copy original image
            shutil.copy2(
                image_path,
                target_images / filename
            )

            # Create YOLO label
            label_filename = Path(filename).with_suffix(".txt")

            with open(
                target_labels / label_filename,
                "w",
                encoding="utf-8"
            ) as label_file:

                label_file.write(label_content)

            total_images += 1
            total_labels += 1

    print(
        "Images:",
        len(list(target_images.glob("*")))
    )

    print(
        "Labels:",
        len(list(target_labels.glob("*.txt")))
    )

# Create YOLO data configuration
yaml_path = OUTPUT_ROOT / "data.yaml"

yaml_content = f"""path: {OUTPUT_ROOT.as_posix()}

train: images/train
val: images/validation
test: images/test

nc: 1

names:
  0: debris
"""

yaml_path.write_text(yaml_content, encoding="utf-8")

print("\n========== FINAL DATASET SUMMARY ==========")

print("Total images converted:", total_images)
print("Total labels generated:", total_labels)
print("Errors:", len(errors))

if errors:
    for error in errors:
        print("ERROR:", error)

print("\nYOLO dataset location:", OUTPUT_ROOT)
print("YOLO configuration:", yaml_path)

if not errors and total_images == 150 and total_labels == 150:
    print("\nSUCCESS: YOLO DATASET PREPARED!")
else:
    print("\nWARNING: Review errors and dataset counts.")
