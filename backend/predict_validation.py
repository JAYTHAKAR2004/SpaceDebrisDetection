
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "runs" / "debris_yolo11n_cpu_test"
    / "weights" / "best.pt"
)

VAL_IMAGES = (
    ROOT / "dataset" / "yolo_debris"
    / "images" / "validation"
)

def main():
    print("Loading trained YOLO11n model...")

    model = YOLO(str(MODEL_PATH))

    print("Generating validation predictions...")

    results = model.predict(
        source=str(VAL_IMAGES),
        imgsz=640,
        conf=0.15,
        iou=0.5,
        device="cpu",
        save=True,
        project=str(ROOT / "runs"),
        name="debris_validation_predictions",
        exist_ok=True
    )

    print("\nPrediction completed!")
    print("Images processed:", len(results))
    print(
        "Saved predictions:",
        ROOT / "runs" / "debris_validation_predictions"
    )

if __name__ == "__main__":
    main()
