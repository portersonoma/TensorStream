# TensorStream Land Cover Segmentation

This project focuses on land cover classification using YOLOv8 segmentation models, in partnership with Cascade Stream Solutions, LLC. The goal is to identify and map three key land cover types in aerial imagery:

* **Road (class 0)**
* **Photosynthesizing Vegetation (PVeg, class 1)**
* **Water (class 2)**

The resulting predictions can support hydrologic modeling, habitat assessment, and surface temperature analysis.

## Repository Structure

```
├── Chip.ipynb                 # Legacy chipping workflow
├── ChipCrossVal.ipynb         # Chipping with cross-validation support
├── Chip_Monolith.ipynb        # Monolithic image chipping for full-coverage maps
├── Train.py                   # Basic YOLOv8 training script
├── TrainCrossVal.py           # Cross-validation training script
├── Train_Monolith.py          # Monolithic training configuration (GPU0)
├── Train_Monolith2.py         # Monolithic training configuration (GPU1)
├── dataset/                   # Processed dataset (YOLOv8 format)
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   ├── labels/val/
│   └── data.yaml
└── runs/                      # YOLOv8 training runs and logs
```

## Setup

1. **Install dependencies**:

   ```bash
   pip install ultralytics opencv-python geopandas rasterio
   ```

2. **Prepare the dataset**:

   * Use `Chip_Monolith.ipynb` or `ChipCrossVal.ipynb` to tile the source TIFF imagery.
   * Ensure all label masks are converted into YOLOv8 polygon format with class IDs 0 (Road), 1 (PVeg), and 2 (Water).
   * Background-only tiles can be excluded during dataset construction to focus training on meaningful examples.

3. **Train a model**:

   * For training on a monolithic dataset:

     ```bash
     python Train_Monolith.py        # GPU0
     python Train_Monolith2.py       # GPU1
     ```
   * For cross-validation:

     ```bash
     python TrainCrossVal.py
     ```

4. **Training Notes**:

   * Training scripts use the `AdamW` optimizer, mixed precision (AMP), and disk or RAM caching depending on the configuration.
   * Image resolution during training is set to 1024x1024.
   * Use class oversampling (via duplication of Road-heavy tiles) to improve minority class representation.
   * Data augmentations (flip, HSV shift, etc.) are built into the YOLO training pipeline.

## Evaluation and Validation

* **Holdout Set**: A dedicated validation set (`Flight_2_25pct`) is used for final model performance evaluation.
* **Performance Metrics**:

  * `mAP50`: Mean Average Precision at 0.5 IoU threshold
  * `mAP50-95`: Mean AP averaged across IoU thresholds (0.5 to 0.95)
  * Per-class Precision and Recall scores are reported.

## Postprocessing and Export

* A custom script reconstructs shapefiles from predicted masks:

  * Reprojects YOLOv8 polygon output back to geospatial coordinates
  * Combines tiles using original `tile_metadata.csv` and `raster_shape.txt`
  * Output is saved as a `.shp` file in the coordinate reference system (CRS) of the input TIFF

## Oversampling Road Class

To address class imbalance, a script can be run to duplicate training tiles with high Road content. This is done post-chipping, before training:

```python
copies_per_tile = 5  # Adjust to match desired Road prevalence
```

Ensure no duplicate filenames are used in the validation set, and avoid oversampling validation data to maintain evaluation integrity.

## Acknowledgments

This project is a collaboration with [Cascade Stream Solutions, LLC](https://cascadestreamsolutions.com), providing environmental modeling, river restoration, and engineering services across Oregon and California. The machine learning outputs generated from this project support:

* Rainfall runoff estimation
* Surface heat impact studies
* Climate change analysis
* Habitat availability modeling

The work highlights the power of tailored computer vision solutions for geospatial environmental data.

---

## Future Work

- Integrate into a lightweight Earth Engine alternative
- Time series change detection using past/future imagery
- Add additional classes (e.g., bare ground, buildings)

## Partner Info

**Cascade Stream Solutions, LLC** 

Mr. Joey Howard 
> Specializing in fisheries, restoration, and river engineering in Oregon and California.  
> [Learn more](https://cascadestreamsolutions.com/)

## Authors

- Andrew Porter, Anthony Tara, Sam Tyler, & Zack Nelson   
- CS470 Final Project, Spring 2025  
- Advisor: Dr. Gurman Gill

