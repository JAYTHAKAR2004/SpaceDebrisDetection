
import csv
import random
from pathlib import Path
from remotezip import RemoteZip

BASE_URL = (
    "https://huggingface.co/datasets/"
    "CVI2-UniLU/SPARK-2022/resolve/main/"
)

SAMPLE_SIZE = 25
RANDOM_SEED = 42

SPLITS = {
    "validation": {
        "csv": "dataset/api_metadata/val.csv",
        "zip": "val.zip",
        "archive_folder": "val"
    },
    "test": {
        "csv": "dataset/api_metadata/test.csv",
        "zip": "test.zip",
        "archive_folder": "test"
    }
}

for split_name, config in SPLITS.items():

    print(f"\n========== {split_name.upper()} ==========")

    csv_path = Path(config["csv"])

    with open(csv_path, "r", encoding="utf-8-sig") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["class"].strip() == "debris"
        ]

    random.seed(RANDOM_SEED)
    samples = random.sample(rows, min(SAMPLE_SIZE, len(rows)))

    output_dir = Path(
        f"dataset/debris_samples/{split_name}"
    )

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    archive_url = BASE_URL + config["zip"]

    successful = []

    print("Available debris annotations:", len(rows))
    print("Selected:", len(samples))
    print("Connecting to remote archive...")

    with RemoteZip(archive_url) as remote_zip:

        archive_files = remote_zip.namelist()

        # Match archive members by filename
        file_map = {
            Path(name).name: name
            for name in archive_files
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        }

        print("Remote archive connected!")

        for index, row in enumerate(samples, start=1):

            filename = row["filename"].strip()

            print(f"[{index}/{len(samples)}] {filename}")

            archive_name = file_map.get(filename)

            if archive_name is None:
                print("Not found in archive. Skipping.")
                continue

            try:
                image_data = remote_zip.read(archive_name)

                (image_dir / filename).write_bytes(image_data)

                successful.append(row)

                print("Downloaded:", len(image_data), "bytes")

            except Exception as error:
                print("Download failed:", error)

    annotation_path = output_dir / "annotations.csv"

    with open(
        annotation_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "class", "bbox"]
        )

        writer.writeheader()
        writer.writerows(successful)

    print("\nCollection summary:")
    print("Selected:", len(samples))
    print("Successfully collected:", len(successful))
    print("Failed:", len(samples) - len(successful))
    print("Images:", image_dir.resolve())

print("\nEvaluation dataset collection completed!")
