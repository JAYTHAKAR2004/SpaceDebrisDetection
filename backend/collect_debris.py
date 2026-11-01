
import csv
import random
from pathlib import Path
from remotezip import RemoteZip

# Configuration
DATASET_URL = (
    "https://huggingface.co/datasets/"
    "CVI2-UniLU/SPARK-2022/resolve/main/train.zip"
)

CSV_PATH = Path("dataset/api_metadata/train.csv")

OUTPUT_DIR = Path("dataset/debris_samples/train")
IMAGE_DIR = OUTPUT_DIR / "images"

SAMPLE_SIZE = 100
RANDOM_SEED = 42

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Read debris annotations
debris_rows = []

with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["class"].strip() == "debris":
            debris_rows.append(row)

print("Total training debris annotations:", len(debris_rows))

# Select reproducible random samples
random.seed(RANDOM_SEED)

samples = random.sample(
    debris_rows,
    min(SAMPLE_SIZE, len(debris_rows))
)

print("Selected samples:", len(samples))

# Retrieve selected images remotely
successful_rows = []

print("\nConnecting to remote training archive...")

with RemoteZip(DATASET_URL) as remote_zip:

    archive_files = set(remote_zip.namelist())

    print("Remote archive connected!")
    
    for index, row in enumerate(samples, start=1):

        filename = row["filename"].strip()
        archive_path = f"train/{filename}"

        print(f"\n[{index}/{len(samples)}] {filename}")

        if archive_path not in archive_files:
            print("Image not found in archive. Skipping.")
            continue

        try:
            image_data = remote_zip.read(archive_path)

            output_path = IMAGE_DIR / filename
            output_path.write_bytes(image_data)

            successful_rows.append(row)

            print("Downloaded:", len(image_data), "bytes")

        except Exception as error:
            print("Failed:", error)

# Save matching annotations
annotation_path = OUTPUT_DIR / "annotations.csv"

with open(annotation_path, "w", encoding="utf-8", newline="") as file:

    if successful_rows:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "class", "bbox"]
        )

        writer.writeheader()
        writer.writerows(successful_rows)

print("\n" + "=" * 50)
print("DEBRIS COLLECTION SUMMARY")
print("=" * 50)

print("Selected:", len(samples))
print("Successfully collected:", len(successful_rows))
print("Failed:", len(samples) - len(successful_rows))
print("Images directory:", IMAGE_DIR.resolve())
print("Annotations:", annotation_path.resolve())
