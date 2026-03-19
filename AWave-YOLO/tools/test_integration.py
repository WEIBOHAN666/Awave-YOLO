import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nets.waveformer_dct import Wave2D, WaveBlock, WaveFormer

# Test Wave2D
print("Testing Wave2D...")
wave2d = Wave2D(res=14, dim=96, hidden_dim=96)
input_tensor = torch.randn(1, 96, 14, 14)
freq_embed = torch.randn(14, 14, 96)
output = wave2d(input_tensor, freq_embed)
print(f"Wave2D output shape: {output.shape}")

# Test WaveBlock
print("\nTesting WaveBlock...")
wave_block = WaveBlock(res=14, hidden_dim=96)
output = wave_block(input_tensor, freq_embed)
print(f"WaveBlock output shape: {output.shape}")

# TestWaveFormer
print("\nTesting WaveFormer...")
waveformer = WaveFormer(patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], drop_path_rate=0.2, patch_norm=True, post_norm=True,
                 layer_scale=None, use_checkpoint=False, mlp_ratio=4.0, img_size=224,
                 act_layer='GELU', infer_mode=False)
input_tensor = torch.randn(1, 3, 224, 224)
output = waveformer(input_tensor)
print(f"WaveFormer output shape: {output.shape}")

print("\nAll tests passed successfully!")
