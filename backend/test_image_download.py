
import requests
from pathlib import Path

DATASET = "CVI2-UniLU/SPARK-2022"
FILENAME = "img086351.jpg"

# Public Hugging Face file URL
image_url = (
    f"https://huggingface.co/datasets/{DATASET}"
    f"/resolve/main/train/{FILENAME}"
)

output_dir = Path("dataset/sample_images")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / FILENAME

print("Testing remote image access...")
print("Image:", FILENAME)

try:
    response = requests.get(
        image_url,
        stream=True,
        timeout=60
    )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    
    if not content_type.startswith("image/"):
        raise ValueError(
            f"Unexpected content type: {content_type}"
        )

    with open(output_path, "wb") as file:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                file.write(chunk)

    print("\nImage retrieved successfully!")
    print("Saved at:", output_path.resolve())
    print("Image size:", output_path.stat().st_size, "bytes")

except Exception as error:
    print("\nImage retrieval failed:")
    print(error)
