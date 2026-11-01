
import pandas as pd
from pathlib import Path

path = Path("results/quality_analysis/debris_quality_report.csv")

df = pd.read_csv(path)

print("\n========== BRIGHTNESS OUTLIERS ==========")

print(
    df.nlargest(5, "brightness")[
        ["filename", "brightness", "contrast", "bbox_area_percent"]
    ].to_string(index=False)
)

print("\n========== CONTRAST OUTLIERS ==========")

print(
    df.nlargest(5, "contrast")[
        ["filename", "brightness", "contrast", "bbox_area_percent"]
    ].to_string(index=False)
)

print("\n========== LARGEST OBJECTS ==========")

print(
    df.nlargest(5, "bbox_area_percent")[
        ["filename", "bbox_width", "bbox_height", "bbox_area_percent"]
    ].to_string(index=False)
)
