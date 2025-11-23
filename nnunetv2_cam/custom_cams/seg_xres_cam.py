import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection


class SegXResCAM(BaseCAM):
    """
    Seg-XRes-CAM implementation (HiResCAM with optional gradient pooling).
    
    Based on: https://github.com/Nouman97/Seg_XRes_CAM
    """
    def __init__(self, model, target_layers, reshape_transform=None, 
                 pool_size=None, pool_mode='max'):
        super(SegXResCAM, self).__init__(model, target_layers, reshape_transform)
        self.pool_size = pool_size
        self.pool_mode = pool_mode

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        
        # Apply pooling to gradients if requested
        if self.pool_size is not None and self.pool_size > 1:
            # Convert to torch for easier pooling
            is_numpy = isinstance(grads, np.ndarray)
            if is_numpy:
                grads_tensor = torch.from_numpy(grads)
            else:
                grads_tensor = grads


            
            orig_shape = grads_tensor.shape
            
            # We need to handle 2D and 3D cases
            if len(orig_shape) == 4: # (B, C, H, W)
                # Pool
                if self.pool_mode == 'max':
                    pooled = F.max_pool2d(grads_tensor, kernel_size=self.pool_size, stride=self.pool_size)
                elif self.pool_mode == 'mean':
                    pooled = F.avg_pool2d(grads_tensor, kernel_size=self.pool_size, stride=self.pool_size)
                else:
                    raise ValueError(f"Unsupported pool_mode: {self.pool_mode}")
                
                # Upsample back (Nearest neighbor as per original implementation's order=0)
                upsampled = F.interpolate(pooled, size=orig_shape[2:], mode='nearest')
                
            elif len(orig_shape) == 5: # (B, C, D, H, W)
                # Pool
                if self.pool_mode == 'max':
                    pooled = F.max_pool3d(grads_tensor, kernel_size=self.pool_size, stride=self.pool_size)
                elif self.pool_mode == 'mean':
                    pooled = F.avg_pool3d(grads_tensor, kernel_size=self.pool_size, stride=self.pool_size)
                else:
                    raise ValueError(f"Unsupported pool_mode: {self.pool_mode}")
                
                # Upsample back
                upsampled = F.interpolate(pooled, size=orig_shape[2:], mode='nearest')
            else:
                # Fallback or error? Let's assume it matches
                upsampled = grads_tensor

            if is_numpy:
                grads = upsampled.cpu().numpy()
            else:
                grads = upsampled

        elementwise_activations = grads * activations

        if eigen_smooth:
            print("Warning: HiResCAM's faithfulness guarantees do not hold if smoothing is applied")
            cam = get_2d_projection(elementwise_activations)
        else:
            cam = elementwise_activations.sum(axis=1)
            
        return cam
