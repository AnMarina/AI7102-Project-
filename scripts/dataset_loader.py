import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import os

class CentralAsiaDataset(Dataset):
    """
    Custom PyTorch Dataset for Central Asia LULC Segmentation.
    Handles 9-channel TIF files (Sentinel-2 + Sentinel-1 + Label).
    """

    def __init__(self, root_dir, country_filter=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g. '../dataset_central_asia')
            country_filter (list, optional): List of countries to include. 
                                             e.g. ['Kazakhstan', 'Tajikistan'] for training.
                                             If None, loads all countries.
            transform (callable, optional): Albumentations or Torchvision transforms for augmentation.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Walk through the directory to find all TIF files
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Data directory not found at {root_dir}. Please run download_chips.py first.")

        print(f"Scanning for data in {root_dir}...")
        
        for country in os.listdir(root_dir):

            if country_filter is not None and country not in country_filter:
                continue
            
            country_path = os.path.join(root_dir, country)
            if not os.path.isdir(country_path): 
                continue
            
            for class_folder in os.listdir(country_path):
                class_path = os.path.join(country_path, class_folder)
                if not os.path.isdir(class_path):
                    continue
                
                for filename in os.listdir(class_path):
                    if filename.endswith(".tif"):
                        self.image_paths.append(os.path.join(class_path, filename))

        print(f"-> Found {len(self.image_paths)} images for {country_filter if country_filter else 'All Countries'}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Open the 9-Channel TIF using Rasterio
        try:
            with rasterio.open(img_path) as src:
                # Read shape is (Channels, Height, Width) -> (9, 224, 224)
                data = src.read() 
        except Exception as e:
            print(f"Error reading file {img_path}: {e}")
            return torch.zeros(8, 224, 224), torch.zeros(224, 224)

        
        # Extract Inputs X and Targets Y
        image = data[0:8, :, :] 
        mask = data[8, :, :]  

        # Preprocessing and Normalization
        image = image.astype(np.float32)

        image[0:6] = np.clip(image[0:6] * 3.3, 0, 1) 

        image[6:8] = np.clip((image[6:8] + 30) / 30, 0, 1)

        # Tensor Conversion
        image_tensor = torch.from_numpy(image)

        mask_tensor = torch.from_numpy(mask).long()

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask_tensor

# test the dataset loader
if __name__ == "__main__":
    import sys
    
    try:
        dataset = CentralAsiaDataset(root_dir='../dataset_central_asia')
        
        if len(dataset) > 0:
            x, y = dataset[0]
            print("\n--- Data Loader Test Success ---")
            print(f"Input Tensor Shape: {x.shape} (Should be [8, 224, 224])")
            print(f"Target Mask Shape: {y.shape} (Should be [224, 224])")
            print(f"Input Data Type: {x.dtype}")
            print(f"Target Data Type: {y.dtype}")
            print(f"Min/Max Value in Input: {x.min():.2f} / {x.max():.2f}")
        else:
            print("Dataset initialized but found 0 images. (Did you run the downloader?)")
            
    except Exception as e:
        print(f"\nTest Setup Note: {e}")