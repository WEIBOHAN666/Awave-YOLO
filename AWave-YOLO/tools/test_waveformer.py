import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import sys
import os

# Add WaveFormer path to Python path
sys.path.append('d:\\WaveFormer-main')

from nets.yolo import YoloBody

# Testing WaveFormer integration into YOLOv8
if __name__ == '__main__':
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model
    input_shape = [3, 640, 640]
    num_classes = 20  # Number of categories in the VOC dataset
    phi = 's'  # Use small version
    
    print('Initializing YOLOv8 with WaveFormer backbone...')
    model = YoloBody(input_shape, num_classes, phi=phi, pretrained=False)
    model.to(device)
    
    # Create test input
    input_tensor = torch.randn(1, 3, 640, 640).to(device)
    
    # Forward propagation test
    print('Testing forward pass...')
    with torch.no_grad():
        outputs = model(input_tensor)
    
    print('Forward pass successful!')
    print(f'Outputs type: {type(outputs)}')
    print(f'Number of outputs: {len(outputs)}')
    
    # Print output shape
    if isinstance(outputs, tuple):
        for i, output in enumerate(outputs):
            if isinstance(output, torch.Tensor):
                print(f'Output {i} shape: {output.shape}')
            elif isinstance(output, list):
                print(f'Output {i} is a list with {len(output)} tensors')
                for j, tensor in enumerate(output):
                    if isinstance(tensor, torch.Tensor):
                        print(f'  Tensor {j} shape: {tensor.shape}')
    
    print('WaveFormer integration test completed successfully!')
