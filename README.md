# Land Cover Classification for Central Asia Using Multi-Sensor Satellite Imagery

## Project Overview

This project implements a deep learning pipeline for **semantic segmentation of Land Use/Land Cover (LULC)** in Central Asia. It uses a multi-sensor fusion approach, combining:

* **Sentinel-2** (Optical)
* **Sentinel-1** (Radar)

to classify **8 land cover categories** across diverse terrains in:

* Kazakhstan
* Kyrgyzstan
* Tajikistan
* Turkmenistan
* Uzbekistan

The model architecture is a **U-Net** with a **ResNet-34 encoder**, designed to handle **9-channel geospatial inputs**:

* 6 Optical (Sentinel-2)
* 2 Radar (Sentinel-1)
* 1 Label (LULC)

---

## Project Structure

```bash
central-asia-lulc/
│
├── src/
│   └── data/                   # CSV coordinate files (Samples_Country.csv)
│
├── scripts/                    # Python source code
│   ├── download_chips.py       # Data extraction script (GEE -> Local)
│   ├── dataset_loader.py       # PyTorch Dataset class & preprocessing
│   ├── model.py                # U-Net architecture definition
│   ├── train.py                # Main training loop (Train/Val/Test split)
│   ├── evaluate.py             # Quantitative evaluation (IoU calculation)
│   └── visualize.py            # Qualitative visualization (Input vs. Prediction)
│
└── README.md                   # Project documentation
```


## Dependencies & Installation

This project is implemented in **Python 3.10+**.

### Required Libraries

* **PyTorch** – Deep learning framework
* **segmentation-models-pytorch** – U-Net & encoder implementations
* **rasterio** – Handling geospatial TIFF files
* **earthengine-api** – Connecting to Google Earth Engine
* **pandas** – DataFrame manipulation for CSVs
* **tqdm** – Progress bars
* **matplotlib** – Visualization
* **requests** – HTTP requests (for downloads / APIs)

### Installation Instructions

```bash
# 1. Create a virtual environment (optional but recommended)
conda create -n ca-lulc python=3.10 -y
conda activate ca-lulc

# 2. Install PyTorch (example: CPU-only; adjust as needed)
pip install torch torchvision

# 3. Install project dependencies
pip install rasterio segmentation-models-pytorch earthengine-api pandas tqdm requests matplotlib
```

## Data Acquisition

The dataset is created using **Google Earth Engine (GEE)**.

Step 1: Export Image Assets (GEE)
We downloaded assets in Google Earth Engine Code Editor. For large countries (Kazakhstan, Turkmenistan), used 30m resolution. For others, used 10m. 

Step 2: Generate Sampling Points (GEE)
Runned the script to export CSV files containing random coordinates (Samples_Kazakhstan.csv, etc.) to the Google Drive. They are provided in ```bash src/data/``` to avoid the long downloading time.

### Step 3: Download Image Chips (Python)

From the project root:

```bash
python scripts/download_chips.py
```

This script:

* Reads the CSVs in `src/data/`
* Connects to the GEE assets
* Downloads **224 × 224** pixel GeoTIFF chips
* Saves them into `dataset_central_asia/`

## Usage

### 1. Training the Model

To train the U-Net model on the downloaded dataset:

```bash
python scripts/train.py
```

* **Configuration:**

  * 80% Train
  * 10% Validation
  * 10% Test

* **Output:**

  * Saves the best model weights to:

    ```text
    scripts/best_model.pth
    ```

### 2. Evaluation (quantitative)

To calculate **Intersection over Union (IoU)** scores for each class on the test set:

```bash
python scripts/evaluate.py
```

* Outputs per-class and mean IoU metrics to the console (and/or logs, depending on implementation).

### 3. Visualization (qualitative)

To generate side-by-side comparisons of:

* Satellite Input
* Ground Truth
* Model Prediction

run:

```bash
python scripts/visualize.py
```

* **Output:**

  * Visualization images are saved to:

    ```text
    results_visualization/
    ```
