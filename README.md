# AI-Powered Automated Underwater Marine Debris and Anomaly Detection System Using Sonar Imagery

## 1. Project Overview

Underwater marine debris poses a significant threat to marine ecosystems, biodiversity, and maritime activities. Traditional underwater monitoring and debris identification methods can be time-consuming, expensive, and dependent on manual inspection.

This project aims to develop an AI-powered computer vision system that uses Forward-Looking Sonar (FLS) imagery to automatically detect and classify underwater marine debris and other objects.

The system uses machine learning and object detection techniques to identify objects in sonar images, locate them using bounding boxes, and classify them into predefined categories.

The initial implementation focuses on supervised object detection using the Watertank subset of the Marine Debris FLS Dataset.

## 2. Project Objectives

The main objectives are to:

- Explore and understand the Marine Debris FLS dataset.
- Analyse the distribution of object categories.
- Inspect and validate the existing image annotations.
- Prepare the dataset for object detection.
- Convert the original bounding-box annotations into YOLO format.
- Train an object detection model to identify underwater objects.
- Evaluate model performance using appropriate object detection metrics.
- Develop a prototype that visualises detected objects, predicted classes, and confidence scores.

## 3. Dataset Description

### Dataset Source

The project uses the Marine Debris FLS Datasets by Valdenegro-Toro et al. (2025).

- **Dataset:** Marine Debris FLS Datasets
- **DOI:** https://doi.org/10.5281/zenodo.15101686
- **GitHub Repository:** https://github.com/mvaldenegro/marine-debris-fls-datasets
- **Research Paper:** https://arxiv.org/html/2503.22880v1
- **Sonar Equipment:** ARIS Explorer 3000 Forward-Looking Sonar.

### Dataset Subsets

The dataset contains three main subsets:

| Subset | Description |
|---|---|
| Watertank | Annotated sonar images for object detection and segmentation |
| Turntable | Image patches for object classification |
| Quarry | Unlabelled sonar frames for potential anomaly exploration |

**Current project scope:** The initial object detection implementation uses the Watertank subset.

### Watertank Dataset

The Watertank subset contains:

- 1,868 annotated sonar images.
- 2,364 annotated objects.
- 10 object categories.
- Bounding-box annotations.
- Polygon segmentation annotations.

The initial implementation focuses on the bounding-box annotations for object detection.

## 4. Technology Stack

The project uses the following technologies:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Dataset exploration and experimentation |
| Pandas | Data analysis and manipulation |
| Matplotlib | Data visualisation |
| Pillow (PIL) | Image loading and inspection |
| JSON | Reading the original annotations |
| PyTorch / Ultralytics YOLO | Planned object detection implementation |

The object detection framework will be finalised during model development.

## 5. Project Structure

The project follows the following directory structure:

```text
Sonar-Based-Marine-Debris-Detection/
│
├── data/
│   ├── raw/
│   │   └── watertank/
│   │       └── marine-debris-watertank-release/
│   │           └── fls-images/
│   │               ├── annotations.json
│   │               └── sonar image files
│   │
│   └── processed/
│       ├── images/
│       └── labels/
│
├── notebooks/
│   └── dataset_exploration.ipynb
│
├── models/
│
├── src/
│
├── results/
│
├── README.md
└── requirements.txt
```

*Note: The processed, models, src, and results directories represent the planned project structure. They will be populated as development progresses.*

## 6. Dataset Exploration and Initial Analysis

Dataset exploration was performed using the Jupyter Notebook:

`notebooks/dataset_exploration.ipynb`

### 6.1 Dataset Path Configuration

The project directory and dataset directory were defined using Python's `pathlib` module.

```python
from pathlib import Path

# Project directory
PROJECT_DIR = Path.cwd().parent

# Dataset directory
DATA_DIR = PROJECT_DIR / "data" / "raw" / "watertank"

print("Project directory:", PROJECT_DIR)
print("Dataset directory:", DATA_DIR)

# Check whether the dataset exists
print("Dataset exists:", DATA_DIR.exists())

# List folders and files
for item in DATA_DIR.iterdir():
    print(item.name)
```

This approach provides a consistent way to access project files without hardcoding absolute file paths.

### 6.2 Image and Annotation Discovery

The dataset was explored to locate the sonar images and annotation files.

The image directory was identified using:

```python
image_dirs = list(DATA_DIR.rglob("fls-images"))

IMAGE_DIR = image_dirs[0]

print("Image directory:", IMAGE_DIR)
```

The annotation file was located using:

```python
annotation_files = list(DATA_DIR.rglob("annotations.json"))

ANNOTATION_FILE = annotation_files[0]

print("Annotation file:", ANNOTATION_FILE)
```

### 6.3 Image Inspection

The images were loaded using Pillow and displayed using Matplotlib.

Initial visual inspection confirmed that the dataset contains Forward-Looking Sonar imagery.

Sample images were displayed to understand the visual characteristics of underwater sonar data before beginning model development.

### 6.4 Annotation Inspection

The original annotation file was loaded using Python's JSON module.

```python
import json

with open(ANNOTATION_FILE, "r") as file:
    annotations = json.load(file)
```

The annotation structure was inspected to understand how objects are represented.

Each image is referenced by its filename, with its corresponding annotations stored under the `bounding-boxes` key.

An individual bounding box contains:

| Field | Description |
|---|---|
| class | Object category name |
| class-id | Numeric identifier of the category |
| width | Bounding-box width in pixels |
| height | Bounding-box height in pixels |
| top-left-x | Horizontal coordinate of the top-left corner |
| top-left-y | Vertical coordinate of the top-left corner |

Example annotation:

```python
{
    'class': 'can',
    'width': 61,
    'height': 51,
    'top-left-x': 123,
    'top-left-y': 399,
    'class-id': 0
}
```

### 6.5 Bounding-Box Visualisation

The original annotations were visualised over the corresponding sonar images using Matplotlib.

Rectangular bounding boxes were drawn using `matplotlib.patches.Rectangle`.

Each bounding box was displayed with:

- A red rectangle indicating the object location.
- A yellow text label identifying the object category.

This visual inspection confirmed that the annotation coordinates were being interpreted correctly for the inspected sample image.

## 7. Dataset Statistics

A dataset-level analysis was conducted to determine the number of annotated images, total objects, and distribution of object categories.

### 7.1 Overall Dataset Statistics

| Metric | Result |
|---|---:|
| Total annotated images | 1,868 |
| Images with bounding boxes | 1,868 |
| Images without bounding boxes | 0 |
| Total annotated objects | 2,364 |
| Number of object categories | 10 |
| Average objects per image | 1.27 |

Every image referenced in the annotation dictionary contains at least one bounding box.

### 7.2 Object Class Distribution

The following table presents the distribution of annotated objects across the ten categories.

| Class ID | Object Category | Number of Objects |
|---:|---|---:|
| 0 | can | 367 |
| 1 | bottle | 449 |
| 2 | drink-carton | 349 |
| 3 | chain | 226 |
| 4 | propeller | 137 |
| 5 | tire | 331 |
| 6 | hook | 133 |
| 7 | valve | 208 |
| 8 | shampoo-bottle | 99 |
| 9 | standing-bottle | 65 |
| | **Total** | **2,364** |

A bar chart was generated using Matplotlib to visualise the class distribution.

The analysis shows that `bottle` is the most represented category, with 449 objects, while `standing-bottle` is the least represented, with 65 objects.

This indicates class imbalance within the dataset, which will be considered during model development and evaluation.

## 8. Data Quality and Annotation Validation

Before preparing the dataset for model training, several validation procedures were performed.

### 8.1 Class ID Consistency

A validation procedure was implemented to confirm that each object category consistently uses the same numeric class ID throughout the annotation file.

The validation produced the following mapping:

| Class ID | Object Category |
|---:|---|
| 0 | can |
| 1 | bottle |
| 2 | drink-carton |
| 3 | chain |
| 4 | propeller |
| 5 | tire |
| 6 | hook |
| 7 | valve |
| 8 | shampoo-bottle |
| 9 | standing-bottle |

**Result:**
- Total categories: 10.
- Inconsistent class IDs: 0.
- All categories have consistent numeric identifiers.

The original class IDs will be preserved during YOLO annotation conversion.

### 8.2 Bounding-Box Coordinate Validation

All bounding-box coordinates were checked against their corresponding image dimensions.

The validation checked that:

- The top-left coordinates are non-negative.
- Bounding-box width and height are positive.
- The right boundary does not exceed the image width.
- The bottom boundary does not exceed the image height.

The following results were obtained:

| Metric | Result |
|---|---:|
| Total bounding boxes | 2,364 |
| Valid bounding boxes | 2,364 |
| Invalid bounding boxes | 0 |
| Validation success rate | 100% |

All bounding boxes were found to be within their respective image boundaries.

### 8.3 Image File Verification

The image directory was checked to confirm that all image filenames referenced in the annotation file correspond to actual image files.

The validation filtered files using supported image extensions to avoid treating metadata files such as `annotations.json` as images.

The results were:

| Metric | Result |
|---|---:|
| Images referenced in annotations | 1,868 |
| Actual image files | 1,868 |
| Missing image files | 0 |
| Images without annotations | 0 |

All annotated images were successfully matched with their corresponding image files.

## 9. Summary of Initial Dataset Validation

The initial dataset exploration and validation produced the following results:

| Validation Check | Outcome |
|---|---|
| Dataset loading | Successful |
| Image discovery | Successful |
| Annotation loading | Successful |
| Sample image visualisation | Successful |
| Bounding-box visualisation | Successful for inspected sample |
| Class distribution analysis | Completed |
| Class ID consistency | Passed |
| Bounding-box coordinate validation | Passed |
| Image file verification | Passed |

The Watertank dataset is now ready for the next stage of preparation, subject to further annotation review and conversion checks.

## 10. Next Development Stages

The following stages will be implemented next.

### Stage 1: YOLO Annotation Conversion

The original bounding-box annotations use pixel coordinates and top-left corner positions.

YOLO requires the following format:

```text
class_id x_center y_center width height
```

The bounding-box coordinates must be normalised using the original image dimensions.

The conversion equations are:

\[
x_{center} = \frac{x + w/2}{W}
\]

\[
y_{center} = \frac{y + h/2}{H}
\]

\[
w_{normalized} = \frac{w}{W}
\]

\[
h_{normalized} = \frac{h}{H}
\]

Where:

- \(x,y\): Top-left bounding-box coordinates.
- \(w,h\): Bounding-box width and height.
- \(W,H\): Original image width and height.

The conversion will initially be tested on a single image before processing the complete dataset.

### Stage 2: Dataset Splitting

The processed dataset will be divided into:

- Training set.
- Validation set.
- Testing set.

The splitting strategy will be selected to ensure appropriate class representation and prevent data leakage.

### Stage 3: Model Development

A YOLO-based object detection model will be implemented as the initial detection baseline.

The model will learn to identify and localise the ten object categories in sonar images.

### Stage 4: Model Evaluation

Model performance will be assessed using appropriate object detection metrics, including:

- Precision.
- Recall.
- Mean Average Precision (mAP).
- Intersection over Union (IoU).

### Stage 5: Detection Visualisation and Prototype

The final prototype will display sonar images with predicted bounding boxes, object categories, and confidence scores.

Further exploration of anomaly detection using unlabelled Quarry sonar frames may be considered as an extension.

## 11. Team Members

- Raudhat Ramadhan
- Given Kangwa
- Kelvin Kimani

## 12. Project Status

**Current Status: Initial Dataset Exploration and Validation Completed**

The project has successfully completed the initial exploration and integrity checks of the Watertank sonar dataset.

The next immediate task is to convert the original bounding-box annotations into YOLO format, validate the conversion, and prepare the dataset for model training.

---

**Project:** AI-Powered Automated Underwater Marine Debris and Anomaly Detection System Using Sonar Imagery

**Dataset:** Marine Debris FLS Datasets (2025)

**Current Focus:** Supervised Object Detection Using Forward-Looking Sonar Imagery
