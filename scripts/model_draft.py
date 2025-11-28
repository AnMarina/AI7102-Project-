import segmentation_models_pytorch as smp
import torch

def build_model(num_classes=8, encoder="resnet34"):
    """
    Creates a U-Net model with a ResNet backbone.
    Adapted for 9-channel satellite input (Sentinel-2 + Sentinel-1).
    
    Args:
        num_classes (int): Number of LULC classes to predict.
        encoder (str): Backbone name (e.g. 'resnet34', 'resnet50').
    """
    print(f"Building U-Net with {encoder} encoder for 9-channel input...")
    
    model = smp.Unet(
        encoder_name=encoder,         
        encoder_weights="imagenet",   
        in_channels=8,                 
        classes=num_classes            
    )
    
    return model

if __name__ == "__main__":
    dummy_input = torch.randn(2, 9, 224, 224)
    model = build_model()
    
    output = model(dummy_input)
    print(f"\nModel Built Successfully!")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape} (Expected: [2, 8, 224, 224])")