
import cv2
import json
from pathlib import Path

image_path = Path(
    "dataset/sample_images/img086351.jpg"
)

image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("Could not load image!")

print("Image loaded successfully!")
print("Image dimensions:", image.shape)

# Annotation retrieved from SPARK API
bbox = [311, 183, 415, 657]
class_name = "smart_1"

x1, y1, x2, y2 = bbox

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
    class_name,
    (x1, max(y1 - 10, 20)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)

output_path = Path("results/annotated_sample.jpg")
output_path.parent.mkdir(exist_ok=True)

cv2.imwrite(str(output_path), image)

print("Annotated image saved:", output_path.resolve())
