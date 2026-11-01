
import requests
import json
from pathlib import Path

BASE_URL = "https://datasets-server.huggingface.co"
DATASET = "CVI2-UniLU/SPARK-2022"

def get_splits():
    url = f"{BASE_URL}/splits"

    response = requests.get(
        url,
        params={"dataset": DATASET},
        timeout=60
    )
    response.raise_for_status()

    return response.json()


def get_first_rows(split="train"):
    url = f"{BASE_URL}/first-rows"

    params = {
        "dataset": DATASET,
        "config": "default",
        "split": split
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )
    response.raise_for_status()

    return response.json()


if __name__ == "__main__":

    output_dir = Path("dataset/api_metadata")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Connecting to SPARK 2022 API...")

    splits = get_splits()

    print("\nAvailable splits:")

    for item in splits["splits"]:
        print(item["split"])

    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=4)

    print("\nFetching training samples...")

    data = get_first_rows("train")

    with open(output_dir / "train_sample.json", "w") as f:
        json.dump(data, f, indent=4)

    print("\nDataset columns:")

    for feature in data["features"]:
        print(feature["name"])

    print("\nFirst five annotations:")

    for item in data["rows"][:5]:
        print(item["row"])

    print("\nAPI integration successful!")
