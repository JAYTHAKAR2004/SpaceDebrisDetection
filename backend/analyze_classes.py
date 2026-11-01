
import csv
import requests
from pathlib import Path
from collections import Counter, defaultdict

BASE_URL = (
    "https://huggingface.co/datasets/"
    "CVI2-UniLU/SPARK-2022/resolve/main/labels/"
)

SPLITS = {
    "train": "train.csv",
    "validation": "val.csv",
    "test": "test.csv"
}

OUTPUT_DIR = Path("dataset/api_metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class_counts = Counter()
class_images = defaultdict(set)
total_annotations = 0

for split, filename in SPLITS.items():

    print(f"\nFetching {split} annotations...")

    url = BASE_URL + filename

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    csv_path = OUTPUT_DIR / filename
    csv_path.write_bytes(response.content)

    print("Saved:", csv_path)
    print("CSV size:", len(response.content), "bytes")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:

        reader = csv.DictReader(file)

        print("Columns:", reader.fieldnames)

        for row in reader:

            class_name = row["class"].strip()
            image_name = row["filename"].strip()

            class_counts[class_name] += 1
            class_images[class_name].add(image_name)
            total_annotations += 1

print("\n" + "=" * 55)
print("SPARK 2022 CLASS DISTRIBUTION")
print("=" * 55)

for class_name, count in class_counts.most_common():

    print(
        f"{class_name:25s} "
        f"Annotations: {count:8d} | "
        f"Images: {len(class_images[class_name]):8d}"
    )

print("\nTotal annotations:", total_annotations)
print("Total unique classes:", len(class_counts))

print("\nClass discovery completed!")
