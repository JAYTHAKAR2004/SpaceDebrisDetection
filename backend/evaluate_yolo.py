
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "runs" / "debris_yolo11n_cpu_test"
    / "weights" / "best.pt"
)

DATA_YAML = ROOT / "dataset" / "yolo_debris" / "data.yaml"

def main():
    print("=" * 55)
    print("SPACE DEBRIS DETECTION - TEST EVALUATION")
    print("=" * 55)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        plots=True,
        project=str(ROOT / "runs"),
        name="debris_yolo11n_test_evaluation",
        exist_ok=True
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")

    print("\nTest evaluation completed!")

if __name__ == "__main__":
    main()
