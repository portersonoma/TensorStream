import os
from ultralytics import YOLO
import torch

def main():
    torch.cuda.empty_cache()

    dataset_dir = "C:/QGIS/dataset"
    model_type = "yolov8s-seg.pt"
    img_size = 640
    save_dir = "C:/QGIS/runs/segment/train"
    class_names = ["Water", "Road", "PVeg"]

    data_yaml = os.path.join(dataset_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {dataset_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    print(f"✅ data.yaml written to: {data_yaml}")

    model = YOLO(model_type)
    model.train(
        data=data_yaml,
        imgsz=img_size,
        epochs=50,
        batch=16,
        workers=2,
        amp=True,
        patience=5,
        device="cuda",
        save=True,
        save_period=-1,
        project=save_dir,
        name="main_model",
        verbose=True,
        plots=False,
        cache=True
    )

if __name__ == "__main__":
    main()
