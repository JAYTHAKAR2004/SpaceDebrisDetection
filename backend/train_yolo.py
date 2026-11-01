
from pathlib import Path
from ultralytics import YOLO

# Project root directory
ROOT = Path(__file__).resolve().parent.parent

# Dataset configuration
DATA_YAML = ROOT / "dataset" / "yolo_debris" / "data.yaml"

# Output directory
RUNS_DIR = ROOT / "runs"

def main():
    print("=" * 55)
    print("SPACE DEBRIS DETECTION - YOLO11 TRAINING")
    print("=" * 55)

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    print("\nLoading pretrained YOLO11n model...")

    model = YOLO("yolo11n.pt")

    print("\nStarting CPU training...")
    print("Dataset:", DATA_YAML)
    print("Device: CPU")
    print("Epochs: 3 (initial pipeline test)")
    print("Image size: 640")

    results = model.train(
        data=str(DATA_YAML),
        epochs=3,
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        cache=False,
        optimizer="AdamW",
        lr0=0.001,
        patience=3,
        mosaic=0.3,
        mixup=0.0,
        plots=True,
        project=str(RUNS_DIR),
        name="debris_yolo11n_cpu_test",
        exist_ok=True
    )

    print("\nTraining completed!")
    print("Results saved at:", RUNS_DIR / "debris_yolo11n_cpu_test")
    print("Best model:", RUNS_DIR / "debris_yolo11n_cpu_test" / "weights" / "best.pt")

if __name__ == "__main__":
    main()
