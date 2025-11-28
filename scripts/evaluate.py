import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from dataset_loader import CentralAsiaDataset
from model_draft import build_model

# Configurations
MODEL_PATH = 'best_model.pth'
DATA_PATH = '../dataset_central_asia'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available(): DEVICE = "mps"

# Classes
CLASS_NAMES = ['Tree', 'Shrub', 'Grass', 'Crop', 'Urban', 'Bare', 'Snow', 'Water']

def calculate_iou(pred, label, num_classes):
    """Calculates Intersection over Union per class."""
    iou_list = []
    pred = pred.view(-1)
    label = label.view(-1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (label == cls)
        
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        if union == 0:
            iou_list.append(float('nan')) 
        else:
            iou_list.append(intersection / union)
            
    return np.array(iou_list)

def main():
    print(f"Evaluating model on {DEVICE}...")
    
    # 1. Load the Full Dataset to replicate the split
    full_dataset = CentralAsiaDataset(
        root_dir=DATA_PATH,
        country_filter=['Kazakhstan', 'Kyrgyzstan', 'Tajikistan', 'Turkmenistan', 'Uzbekistan']
    )
    
    # 2. Recreate the 80/10/10 Split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Total Images: {total_size}")
    print(f"Recreating Split -> Test Set Size: {test_size}")

    _, _, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Load Model
    model = build_model(num_classes=8, encoder="resnet34").to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    total_iou = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1) # Convert probs to Class ID
            
            # Calculate IoU for this batch
            batch_iou = calculate_iou(preds, masks, num_classes=8)
            total_iou.append(batch_iou)
            
    # Average across all batches
    mean_iou_per_class = np.nanmean(np.array(total_iou), axis=0)
    
    print("\n--- FINAL RESULTS (IoU Scores) ---")
    print(f"{'Class':<10} | {'IoU Score':<10}")
    print("-" * 25)
    
    for i, name in enumerate(CLASS_NAMES):
        score = mean_iou_per_class[i]
        print(f"{name:<10} | {score:.4f}")
        
    print("-" * 25)
    print(f"Mean IoU   | {np.nanmean(mean_iou_per_class):.4f}")

if __name__ == "__main__":
    main()