
from remotezip import RemoteZip
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/"
    "CVI2-UniLU/SPARK-2022/resolve/main/train.zip"
)

TARGET = "img086351.jpg"

output_dir = Path("dataset/sample_images")
output_dir.mkdir(parents=True, exist_ok=True)

print("Connecting to remote ZIP...")

try:
    with RemoteZip(URL) as remote_zip:

        print("Remote ZIP connected!")

        # Find the image inside the archive
        matches = [
            name for name in remote_zip.namelist()
            if name.endswith(TARGET)
        ]

        if not matches:
            print("Image not found in archive.")
            print("Sample archive entries:")
            print(remote_zip.namelist()[:20])

        else:
            image_name = matches[0]

            print("Found:", image_name)
            print("Retrieving selected image...")

            image_data = remote_zip.read(image_name)

            output_path = output_dir / TARGET
            output_path.write_bytes(image_data)

            print("Image downloaded successfully!")
            print("Saved:", output_path.resolve())
            print("Size:", len(image_data), "bytes")

except Exception as error:
    print("Remote ZIP access failed:")
    print(type(error).__name__, error)
