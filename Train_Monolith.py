import os
from ultralytics import YOLO
import torch

def main():
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
                                                        # Hide one of the GPUs to avoid defaulting to one you don't want
    torch.cuda.set_device(0)

    dataset_dir = f"/home/znelson/TensorStream/Labeled Data/dataset"
    #checkpoint_path = "/home/znelson/TensorStream/Labeled Data/runs/segment/train/yolov8m_640native_model01/weights/best.pt" 
                                                        # Save checkpoint path to load
    model_type = "yolov8s-seg.pt"
    
    img_size = 1024
    save_dir = f"/home/znelson/TensorStream/Labeled Data/runs/segment/train"
    class_names = ["Road", "PVeg", "Water"]  

    data_yaml = os.path.join(dataset_dir, "data.yaml")
    '''
    with open(data_yaml, "w") as f:
        f.write(f"path: {dataset_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
    '''
    print(f"✅ data.yaml written to: {data_yaml}")

    model = YOLO(model_type)
    #model = YOLO(checkpoint_path)                      # Load checkpoint
    model.train(
        #resume=True,                                   # Resume from last run
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
        name="yolov8s_1600to1024_model11",
        verbose=True,
        plots=True,
        cache="ram",
        optimizer="AdamW",
        lr0=3e-4, 
        weight_decay=5e-4, 
        cos_lr=True, 
        conf=0.25, 
        iou=0.55, 
        auto_augment='albumentations'
    )

if __name__ == "__main__":
    main()
