import os
from ultralytics import YOLO
import torch

def main():
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"            # Hide one of the GPUs to avoid defaulting to one you don't want
    
    torch.cuda.set_device(0)

    dataset_dir = "/home/znelson/TensorStream/Labeled Data/dataset"
    data_yaml = os.path.join(dataset_dir, "data.yaml")  
    model_type = "yolov8s-seg.pt"
    img_size = 1024
    save_dir = "/home/znelson/TensorStream/Labeled Data/runs/segment/train"

    print(f"✅ Using existing data.yaml: {data_yaml}")

    model = YOLO(model_type)
    model.train(
        data=data_yaml,
        imgsz=img_size,
        epochs=200,
        batch=14,
        workers=8,
        amp=True,
        patience=20,
        device="0",
        save=True,
        save_period=-1,
        project=save_dir,
        name="yolov8s_1600to1024_model12",
        verbose=True,
        plots=True,
        cache="ram",
        optimizer="AdamW",
        lr0=3e-4, 
        weight_decay=5e-4, 
        cos_lr=True, 
        conf=0.25, 
        iou=0.55, 
        auto_augment="randaugment",
        copy_paste=0.1,
        warmup_epochs=5
    )

if __name__ == "__main__":
    main()