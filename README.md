Land Cover Classification for Central Asia Using Multi-Sensor Satellite Imagery
Project Overview
This project implements a Deep Learning pipeline for semantic segmentation of Land Use/Land Cover (LULC) in Central Asia. It uses a multi-sensor fusion approach, combining Sentinel-2 (Optical) and Sentinel-1 (Radar) satellite imagery to classify 8 land cover categories across diverse terrains in Kazakhstan, Kyrgyzstan, Tajikistan, Turkmenistan, and Uzbekistan.
The model architecture is a U-Net with a ResNet-34 encoder, designed to handle 9-channel geospatial inputs (6 Optical + 2 Radar + 1 Label).

Project Structure
central-asia-lulc/
│
├── dataset_central_asia/       # (Generated) Directory storing downloaded image chips
│   ├── Kazakhstan/
│   ├── Kyrgyzstan/
│   └── ...
│
├── src/
│   └── data/                   # Directory for CSV coordinate files (Samples_Country.csv)
│
├── scripts/                    # Python source code
│   ├── download_chips.py       # Data extraction script (GEE -> Local)
│   ├── dataset_loader.py       # PyTorch Dataset class & Preprocessing
│   ├── model.py                # U-Net architecture definition
│   ├── train.py                # Main training loop (Train/Val/Test split)
│   ├── evaluate.py             # Quantitative evaluation (IoU calculation)
│   └── visualize.py            # Qualitative visualization (Input vs. Prediction)
│
└── README.md                   # Project documentation


Dependencies & Installation
This project is implemented in Python 3.10+.
Required Libraries
PyTorch: Deep learning framework.
Segmentation Models PyTorch: U-Net architecture implementations.
Rasterio: Handling geospatial TIFF files.
Earth Engine API: Connecting to GEE for data download.
Pandas: Dataframe manipulation for CSVs.
TQDM: Progress bars.
Matplotlib: Visualization.

Installation Instructions
Run the following commands to set up the environment:
# 1. Create a virtual environment (Optional but recommended)
conda create -n ca-lulc python=3.10 -y
conda activate ca-lulc

# 2. Install PyTorch (Mac users with M1/M2/M3 chips)
pip install torch torchvision

# 3. Install Project Dependencies
pip install rasterio segmentation-models-pytorch earthengine-api pandas tqdm requests matplotlib


Data Acquisition Instructions
The dataset is created using Google Earth Engine (GEE). 

Step 1: Export Image Assets (GEE)
We downloaded assets in Google Earth Engine Code Editor. For large countries (Kazakhstan, Turkmenistan), used 30m resolution. For others, used 10m. 

Step 2: Generate Sampling Points (GEE)
Runned the script to export CSV files containing random coordinates (Samples_Kazakhstan.csv, etc.) to your Google Drive.

Step 3: Download Image Chips (Python)
Use the downloaded CSV files from src/data/.
Run the python downloader:
python download_chips.py

This script reads the CSVs, connects to the GEE Assets, and downloads $224 \times 224$ pixel GeoTIFF chips to dataset_central_asia/.

Usage
1. Training the Model
To train the U-Net model on the downloaded dataset:
python scripts/train.py


Configuration: 80% Train, 10% Validation, 10% Test.
Output: Saves the best weights to scripts/best_model.pth.

2. Evaluation (quantitative)
To calculate the Intersection over Union (IoU) scores for each class on the Test set:
python scripts/evaluate.py


3. Visualization (qualitative)
To generate side-by-side comparisons (Satellite Input vs. Ground Truth vs. Prediction):
python scripts/visualize.py
Output images are saved to results_visualization/.

Credits & References
U-Net Implementation: This project uses the segmentation_models_pytorch library by Pavel Yakubovskiy.
Source: https://github.com/qubvel/segmentation_models.pytorch
ResNet Backbone: The encoder uses weights pre-trained on ImageNet, provided by torchvision.
Datasets Used:
Sentinel-1 & Sentinel-2: Copernicus Programme (ESA).
ESA WorldCover v200: European Space Agency.
LGRIP30: Global Food Security-support Analysis Data (GFSAD).
