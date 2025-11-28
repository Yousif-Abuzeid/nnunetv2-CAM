
import torch
import torch.nn as nn
from nnunetv2_cam.custom_cams.score_cam_3d import ScoreCAM3D

class DummyModel3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 1, 3, padding=1)
        self.conv2 = nn.Conv3d(1, 1, 3, padding=1)
        self.fc = nn.Linear(10*10*10, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def test_scorecam3d_custom():
    print("Testing ScoreCAM3D custom implementation...")
    
    device = torch.device("cpu")
    model = DummyModel3D().to(device)
    model.eval()
    
    target_layers = [model.conv2]
    
    # Instantiate ScoreCAM3D
    cam = ScoreCAM3D(model=model, target_layers=target_layers)
    
    # Create dummy 3D input (Batch, Channel, Depth, Height, Width)
    input_tensor = torch.randn(1, 1, 10, 10, 10).to(device)
    
    # Run CAM
    try:
        # We need a target. For binary classification, let's target class 0
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(0)]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
        print(f"✓ ScoreCAM3D ran successfully. Output shape: {grayscale_cam.shape}")
        
        assert grayscale_cam.shape == (1, 10, 10, 10) or grayscale_cam.shape == (10, 10, 10), \
            f"Unexpected output shape: {grayscale_cam.shape}"
            
    except Exception as e:
        print(f"❌ ScoreCAM3D failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_scorecam3d_custom()
