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
    [0, 100, 0],    # 0: Tree (dark green)
    [255, 187, 34], # 1: Shrub (orange)
    [255, 255, 76], # 2: Grass (yellow)
    [240, 150, 255],# 3: Crop (pink)
    [250, 0, 0],    # 4: Urban (red)
    [180, 180, 180],# 5: Bare (grey)
    [240, 240, 240],# 6: Snow (white)
    [0, 100, 200]   # 7: Water (blue)
]

CLASS_NAMES = ['Tree', 'Shrub', 'Grass', 'Crop', 'Urban', 'Bare', 'Snow', 'Water']

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
    
    # 1. Recreate the dataset split 
    try:
        full_dataset = CentralAsiaDataset(
            root_dir=DATA_PATH,
            country_filter=['Kazakhstan', 'Kyrgyzstan', 'Tajikistan', 'Turkmenistan', 'Uzbekistan']
        )
    except FileNotFoundError:
        print("Error: Dataset directory not found. Check DATA_PATH.")
        return

    if len(full_dataset) == 0:
        print("Error: No images found in the dataset folder.")
        return
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 2. Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = build_model(num_classes=8, encoder="resnet34").to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    dataset_indices = list(range(len(test_dataset)))
    random.shuffle(dataset_indices)
    
    saved_count = 0
        
    for idx in dataset_indices:
        if saved_count >= NUM_SAMPLES:
            break
            
        try:
            image, mask = test_dataset[idx]
            
            # filtering logic
            unique_classes = torch.unique(mask).tolist()
            
            if len(unique_classes) < 2:
                continue
                
            # Predict
            input_tensor = image.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # plotting
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            # Input Image (Sentinel-2 RGB)
            rgb = image[0:3].permute(1, 2, 0).numpy()
            rgb = np.clip(rgb, 0, 1) 
            
            ax[0].imshow(rgb)
            ax[0].set_title("Sentinel-2 Input (RGB)")
            ax[0].axis('off')
            
            #Ground truth
            ax[1].imshow(colorize_mask(mask.numpy()))
            ax[1].set_title("Ground Truth (WorldCover)")
            ax[1].axis('off')
            
            #Prediction
            ax[2].imshow(colorize_mask(pred))
            ax[2].set_title("Model Prediction")
            ax[2].axis('off')
            
            save_path = os.path.join(SAVE_DIR, f"viz_sample_{saved_count}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            saved_count += 1
            
        except Exception as e:
            print(f"Error processing sample index {idx}: {e}")

    print("Done! Check the results_visualization folder.")

if __name__ == "__main__":
    main()
