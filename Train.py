import os
from ultralytics import YOLO
import torch

def main():
    # --- Setup ---
    torch.cuda.empty_cache()

    # --- Config ---
    dataset_dir = "C:/QGIS/dataset"
    model_type = "yolov8n-seg.pt"  # You can upgrade to 's', 'm', or 'l' for more accuracy
    img_size = 640
    save_dir = "C:/QGIS/runs/segment/train"
    class_names = ["Road", "PVeg", "Water"]  # Matches YOLO class IDs: 0, 1, 2

    # --- Write YOLO data.yaml ---
    data_yaml = os.path.join(dataset_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {dataset_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    print(f"✅ data.yaml written to: {data_yaml}")

    # --- Load and train model ---
    model = YOLO(model_type)
    model.train(
        data=data_yaml,
        imgsz=img_size,
        epochs=200,
        batch=32,
        workers=2,             # Adjust based on CPU capacity
        amp=True,              # Mixed precision (faster on modern GPUs)
        patience=5,            # Early stopping
        device="cuda",         # Force GPU use
        save=True,
        save_period=-1,        # Only save final model
        project=save_dir,
        name="main_model",
        verbose=True,
        plots=True,
        cache="disk"           # Improves training speed
    )

if __name__ == "__main__":
    main()