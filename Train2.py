import os
import shutil
import random
import yaml
import optuna
import pandas as pd
import torch
from sklearn.model_selection import KFold
from ultralytics import YOLO
from tqdm import tqdm


# --- Configuration ---
base_dir = "C:/QGIS"
collective_dir = os.path.join(base_dir, "collective_dataset")
folds_dir = os.path.join(base_dir, "folds")
folds_count = 3
batch_size = 16
epochs = 10  # Reduced for faster testing
img_size = 640
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = os.path.join(base_dir, "runs/segment/train")
max_dataset_size = 200  # Limit for testing

os.makedirs(folds_dir, exist_ok=True)

# --- Load Data ---
image_dir = os.path.join(collective_dir, "images", "train")
label_dir = os.path.join(collective_dir, "labels", "train")
images = sorted(os.listdir(image_dir))

# --- Limit Dataset Size for Testing ---
if max_dataset_size > 0:
    images = images[:max_dataset_size]

# --- Create Folds (Only Once) ---
if not os.path.exists(os.path.join(folds_dir, "fold_1")):
    kf = KFold(n_splits=folds_count, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(kf.split(images)):
        fold_path = os.path.join(folds_dir, f"fold_{fold+1}")
        os.makedirs(os.path.join(fold_path, "images/train"), exist_ok=True)
        os.makedirs(os.path.join(fold_path, "labels/train"), exist_ok=True)
        os.makedirs(os.path.join(fold_path, "images/val"), exist_ok=True)
        os.makedirs(os.path.join(fold_path, "labels/val"), exist_ok=True)

        # Copy training files
        for i in train_index:
            img_name = images[i]
            label_name = img_name.replace(".png", ".txt")
            shutil.copy(os.path.join(image_dir, img_name), os.path.join(fold_path, "images/train", img_name))
            shutil.copy(os.path.join(label_dir, label_name), os.path.join(fold_path, "labels/train", label_name))

        # Copy validation files
        for i in val_index:
            img_name = images[i]
            label_name = img_name.replace(".png", ".txt")
            shutil.copy(os.path.join(image_dir, img_name), os.path.join(fold_path, "images/val", img_name))
            shutil.copy(os.path.join(label_dir, label_name), os.path.join(fold_path, "labels/val", label_name))

        # Generate data.yaml
        yaml_data = {
            "path": fold_path,
            "train": "images/train",
            "val": "images/val",
            "nc": 3,
            "names": ["Road", "PVeg", "Water"]
        }

        with open(os.path.join(fold_path, "data.yaml"), "w") as f:
            yaml.dump(yaml_data, f)

        print(f"✅ Fold {fold+1} created with {len(train_index)} training samples and {len(val_index)} validation samples")

# --- Run Optuna Study ---
if __name__ == "__main__":

    def objective(trial):
        # Hyperparameter space
        lr0 = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        momentum = trial.suggest_float("momentum", 0.6, 0.98)
        cos_lr = trial.suggest_categorical("cos_lr", [True, False])

        scores = []

        # Train each fold separately
        for fold in range(1, folds_count + 1):
            fold_path = os.path.join(folds_dir, f"fold_{fold}")
            data_yaml = os.path.join(fold_path, "data.yaml")

            model = YOLO("yolov8n-seg.pt")
            results = model.train(
                data=data_yaml,
                imgsz=img_size,
                epochs=epochs,
                batch=batch_size,
                workers=4,
                device=device,
                optimizer="AdamW",
                lr0=lr0,
                weight_decay=weight_decay,
                momentum=momentum,
                cos_lr=cos_lr,
                amp=True,
                project=output_dir,
                name=f"fold_{fold}",
                verbose=False
            )

            # Extract mAP50 from segmentation metrics
            if hasattr(results, "seg"):
                scores.append(results.seg.map50)
            else:
                print(f"⚠️ No segmentation metrics found for fold {fold}")
                return 0  # Skip this trial if metrics are missing

        # Return average score across folds
        return sum(scores) / len(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    # Print best hyperparameters
    print("\n=== Best Hyperparameters ===")
    print(study.best_params)

    # Save the best parameters
    best_params_file = os.path.join(base_dir, "best_params.yaml")
    with open(best_params_file, "w") as f:
        yaml.dump(study.best_params, f)

    print(f"\n✅ Best parameters saved to: {best_params_file}")

