
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = Path("results/quality_analysis/debris_quality_report.csv")

OUTPUT = Path("results/quality_analysis/plots")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# 1. Bounding box width distribution
plt.figure(figsize=(8, 5))
plt.hist(df["bbox_width"], bins=15)
plt.xlabel("Bounding Box Width (pixels)")
plt.ylabel("Number of Images")
plt.title("Distribution of Debris Bounding Box Width")
plt.savefig(OUTPUT / "bbox_width.png", dpi=300)
plt.close()

# 2. Bounding box height distribution
plt.figure(figsize=(8, 5))
plt.hist(df["bbox_height"], bins=15)
plt.xlabel("Bounding Box Height (pixels)")
plt.ylabel("Number of Images")
plt.title("Distribution of Debris Bounding Box Height")
plt.savefig(OUTPUT / "bbox_height.png", dpi=300)
plt.close()

# 3. Bounding box area percentage
plt.figure(figsize=(8, 5))
plt.hist(df["bbox_area_percent"], bins=15)
plt.xlabel("Bounding Box Area (% of Image)")
plt.ylabel("Number of Images")
plt.title("Debris Object Area Distribution")
plt.savefig(OUTPUT / "bbox_area_percentage.png", dpi=300)
plt.close()

# 4. Brightness distribution
plt.figure(figsize=(8, 5))
plt.hist(df["brightness"], bins=15)
plt.xlabel("Mean Grayscale Brightness")
plt.ylabel("Number of Images")
plt.title("Image Brightness Distribution")
plt.savefig(OUTPUT / "brightness.png", dpi=300)
plt.close()

# 5. Contrast distribution
plt.figure(figsize=(8, 5))
plt.hist(df["contrast"], bins=15)
plt.xlabel("Grayscale Standard Deviation")
plt.ylabel("Number of Images")
plt.title("Image Contrast Distribution")
plt.savefig(OUTPUT / "contrast.png", dpi=300)
plt.close()

# 6. Brightness vs contrast
plt.figure(figsize=(8, 5))
plt.scatter(df["brightness"], df["contrast"])
plt.xlabel("Brightness")
plt.ylabel("Contrast")
plt.title("Image Brightness vs Contrast")
plt.savefig(OUTPUT / "brightness_vs_contrast.png", dpi=300)
plt.close()

print("All visualizations generated successfully!")

for file in OUTPUT.iterdir():
    print(file)
