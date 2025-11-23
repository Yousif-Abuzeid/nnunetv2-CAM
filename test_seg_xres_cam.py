import torch
import torch.nn as nn
import numpy as np
from nnunetv2_cam.custom_cams.seg_xres_cam import SegXResCAM
from nnunetv2_cam.cam_core import CAM_METHODS

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def test_seg_xres_cam():
    print("Testing SegXResCAM...")
    
    # 1. Check registration
    if "segxrescam" not in CAM_METHODS:
        print("FAILED: segxrescam not registered in CAM_METHODS")
        return
    print("✓ segxrescam registered")

    # 2. Setup
    model = DummyModel().eval()
    target_layers = [model.conv]
    input_tensor = torch.randn(1, 3, 32, 32)
    
    # 3. Test without pooling
    cam = SegXResCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)
    
    if grayscale_cam.shape != (1, 32, 32):
        print(f"FAILED: Output shape mismatch. Expected (1, 32, 32), got {grayscale_cam.shape}")
        return
    print("✓ Basic forward pass successful")
    
    # 4. Test with pooling
    pool_size = 2
    cam_pooled = SegXResCAM(model=model, target_layers=target_layers, pool_size=pool_size)
    grayscale_cam_pooled = cam_pooled(input_tensor=input_tensor)
    
    if grayscale_cam_pooled.shape != (1, 32, 32):
        print(f"FAILED: Pooled output shape mismatch. Expected (1, 32, 32), got {grayscale_cam_pooled.shape}")
        return
    print("✓ Pooled forward pass successful")

    print("All tests passed!")

if __name__ == "__main__":
    test_seg_xres_cam()
