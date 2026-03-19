import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from nets.yolo import YoloBody

# Generate feature visualizations
def generate_feature_visualization(model, input_shape, save_path):
    print("Generate feature visualization...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Create a random input
    input_tensor = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    
    # Define hook function
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output.cpu().detach())
    
    # Register hooks to specific layers of the backbone network
    if hasattr(model.backbone, 'waveformer'):
        # Register to WaveFormer's first stage output
        layer = model.backbone.waveformer.layers[0][0]
        hook = layer.register_forward_hook(hook_fn)
    
    # forward propagation
    with torch.no_grad():
        model(input_tensor)
    
    # remove hook
    hook.remove()
    
    # Visualize the first feature map
    if feature_maps:
        # Select the first feature map and take the first 16 channels
        features = feature_maps[0][0]
        num_channels = min(16, features.shape[0])
        
        plt.figure(figsize=(16, 16))
        for i in range(num_channels):
            plt.subplot(4, 4, i+1)
            plt.imshow(features[i], cmap='viridis')
            plt.title(f'Channel {i+1}', fontsize=25)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'feature_visualization.png'))
        plt.close()
        print(f"Feature visualization map has been saved to {os.path.join(save_path, 'feature_visualization.png')}")

# main function
def main():
    # Configuration parameters
    input_shape = [640, 640]
    phi = 's'
    save_dir = '.'
    num_classes = 26  # 26 English letters
    
    # Load model
    model = YoloBody(input_shape, num_classes, phi, pretrained=False)
    
    # Load optimal weights
    best_weights_path = os.path.join('logs', 'best_epoch_weights.pth')
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path))
        print(f"The best weights have been loaded: {best_weights_path}")
    
    # Transfer to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Generate feature visualizations
    generate_feature_visualization(model, input_shape, save_dir)

if __name__ == "__main__":
    main()