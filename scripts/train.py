import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset_loader import CentralAsiaDataset
from model_draft import build_model 

#Configurations
BATCH_SIZE = 16          
LEARNING_RATE = 1e-4
EPOCHS = 15
NUM_CLASSES = 8          
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if torch.backends.mps.is_available():
    DEVICE = "mps"
    print(f"Using MPS (Metal Performance Shaders) acceleration.")
else:
    print(f"Using device: {DEVICE}")

DATA_PATH = '../dataset_central_asia'  
SAVE_PATH = 'best_model.pth'

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs one epoch of training."""
    model.train()
    running_loss = 0.0
    
    loop = tqdm(loader, desc="Training", leave=True)
    
    for batch_idx, (images, masks) in enumerate(loop):
        # 1. Move data to GPU/CPU
        images = images.to(device)
        masks = masks.to(device)
        
        # 2. Forward Pass
        predictions = model(images)
        
        # 3. Calculate Loss
        loss = criterion(predictions, masks)
        
        # 4. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 5. Logging
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad(): 
        loop = tqdm(loader, desc="Validation", leave=True)
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            
            predictions = model(images)
            loss = criterion(predictions, masks)
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
    return running_loss / len(loader)

def main():
    print(f"--- Starting Training on {DEVICE} ---")
    
    print("Initializing Datasets...")
    
    # STRATEGY: 80% Train, 10% Val, 10% Test from ALL Countries
    
    # 1. Load the Entire Dataset (All 5 Countries)
    full_dataset = CentralAsiaDataset(
        root_dir=DATA_PATH,
        country_filter=['Kazakhstan', 'Kyrgyzstan', 'Tajikistan', 'Turkmenistan', 'Uzbekistan']
    )
    
    # 2. Calculate Split Sizes
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size # The remaining 10%
    
    print(f"Total Images: {total_size}")
    print(f"Splitting into: Train ({train_size}), Val ({val_size}), Test ({test_size})")
    
    # 3. Random Split
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )
    
    # 4. Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 5. Setup Model, Loss, Optimizer
    model = build_model(num_classes=NUM_CLASSES, encoder="resnet34").to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 6. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Avg Train Loss: {train_loss:.4f} | Avg Val Loss: {val_loss:.4f}")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved new best model to {SAVE_PATH}")

    print("\n--- Training Complete ---")

if __name__ == "__main__":
    main()