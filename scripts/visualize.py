import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from torch.utils.data import random_split
from dataset_loader import CentralAsiaDataset
from model_draft import build_model

# Cpnfigurations
MODEL_PATH = 'best_model.pth'
DATA_PATH = '../dataset_central_asia'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available(): DEVICE = "mps"

SAVE_DIR = '../results_visualization'
NUM_SAMPLES = 5 

# WorldCover Colors 
COLORS = [
    [0, 100, 0],    # Dark Green
    [255, 187, 34], # Orange
    [255, 255, 76], # Yellow
    [240, 150, 255],# Pink
    [250, 0, 0],    # Red
    [180, 180, 180],# Grey
    [240, 240, 240],# White
    [0, 100, 200]   # Blue
]

def colorize_mask(mask):
    """Converts a 2D mask (H, W) into a 3D RGB image (H, W, 3)."""
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for id in range(8):
        img[mask == id] = COLORS[id]
    return img

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Generating visualizations in {SAVE_DIR}...")
    
    # 1. Recreate the Dataset Split
    full_dataset = CentralAsiaDataset(
        root_dir=DATA_PATH,
        country_filter=['Kazakhstan', 'Kyrgyzstan', 'Tajikistan', 'Turkmenistan', 'Uzbekistan']
    )
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Sampling from Test Set ({len(test_dataset)} images)...")

    # 2. Load Model
    model = build_model(num_classes=8, encoder="resnet34").to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 3. Pick Random Samples
    indices = random.sample(range(len(test_dataset)), NUM_SAMPLES)
    
    for i, idx in enumerate(indices):
        try:
            image, mask = test_dataset[idx]
            
            # Predict
            input_tensor = image.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # Plotting
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            # A. Input Image (Sentinel-2 RGB)
            # Channels 0,1,2 = B4,B3,B2 (Red, Green, Blue)
            rgb = image[0:3].permute(1, 2, 0).numpy()
            rgb = np.clip(rgb, 0, 1) 
            
            ax[0].imshow(rgb)
            ax[0].set_title("Sentinel-2 Input (RGB)")
            ax[0].axis('off')
            
            # B. Ground Truth
            ax[1].imshow(colorize_mask(mask.numpy()))
            ax[1].set_title("Ground Truth (WorldCover)")
            ax[1].axis('off')
            
            # C. Prediction
            ax[2].imshow(colorize_mask(pred))
            ax[2].set_title("Model Prediction")
            ax[2].axis('off')
            
            # Save
            save_path = os.path.join(SAVE_DIR, f"viz_sample_{i}.png")
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved: {save_path}")
            plt.close()
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")

    print("Done! Check the folder.")

if __name__ == "__main__":
    main()