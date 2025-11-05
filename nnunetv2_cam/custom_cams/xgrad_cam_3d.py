"""
3D-compatible XGradCAM implementation.

This module provides a fixed version of XGradCAM that works with both
2D and 3D medical imaging data. The original pytorch-grad-cam implementation
hardcodes axis=(2, 3) which only works for 4D tensors (batch, channels, H, W).

This version dynamically determines spatial dimensions to support 5D tensors
(batch, channels, D, H, W) used in 3D medical imaging.
"""

import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


class XGradCAM3D(BaseCAM):
    """
    3D-compatible XGradCAM implementation.
    
    This version fixes the hardcoded axis=(2, 3) in the original implementation
    to support both 2D (4D tensors) and 3D (5D tensors) data.
    """
    
    def __init__(self, model, target_layers, reshape_transform=None):
        super(XGradCAM3D, self).__init__(
            model, target_layers, reshape_transform
        )

    def get_cam_weights(
        self,
        input_tensor,
        target_layer,
        target_category,
        activations,
        grads
    ):
        """
        Compute XGradCAM weights with dynamic spatial dimension handling.
        
        Args:
            input_tensor: Input tensor
            target_layer: Target layer for CAM computation
            target_category: Target class
            activations: Layer activations (numpy array)
            grads: Layer gradients (numpy array)
            
        Returns:
            Weights for each channel (shape: batch x channels)
        """
        # Determine spatial dimensions dynamically
        # For 2D: shape is (batch, channels, H, W) -> spatial_axes = (2, 3)
        # For 3D: shape is (batch, channels, D, H, W) -> spatial_axes = (2, 3, 4)
        if len(grads.shape) == 4:
            # 2D image
            spatial_axes = (2, 3)
            expand_shape = (None, None)
        elif len(grads.shape) == 5:
            # 3D volume
            spatial_axes = (2, 3, 4)
            expand_shape = (None, None, None)
        else:
            raise ValueError(
                f"Invalid grads shape: {grads.shape}. "
                f"Shape should be 4 (2D image) or 5 (3D volume)."
            )
        
        # Sum activations over spatial dimensions
        sum_activations = np.sum(activations, axis=spatial_axes)
        eps = 1e-7
        
        # Expand dimensions for broadcasting
        sum_activations_expanded = sum_activations[(slice(None), slice(None)) + expand_shape]
        
        # Compute weights: grads * activations / (sum_activations + eps)
        weights = grads * activations / (sum_activations_expanded + eps)
        weights = weights.sum(axis=spatial_axes)
        
        return weights
