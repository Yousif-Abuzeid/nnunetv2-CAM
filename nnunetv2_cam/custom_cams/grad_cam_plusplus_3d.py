"""
3D-compatible GradCAM++ implementation.

This module provides a fixed version of GradCAM++ that works with both
2D and 3D medical imaging data. The original pytorch-grad-cam implementation
hardcodes axis=(2, 3) which only works for 4D tensors (batch, channels, H, W).

This version dynamically determines spatial dimensions to support 5D tensors
(batch, channels, D, H, W) used in 3D medical imaging.
"""

import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


class GradCAMPlusPlus3D(BaseCAM):
    """
    3D-compatible GradCAM++ implementation.
    
    Based on: https://arxiv.org/abs/1710.11063
    
    This version fixes the hardcoded axis=(2, 3) in the original implementation
    to support both 2D (4D tensors) and 3D (5D tensors) data.
    """
    
    def __init__(self, model, target_layers, reshape_transform=None):
        super(GradCAMPlusPlus3D, self).__init__(
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
        Compute GradCAM++ weights with dynamic spatial dimension handling.
        
        Implements Equation 19 from https://arxiv.org/abs/1710.11063
        
        Args:
            input_tensor: Input tensor
            target_layer: Target layer for CAM computation
            target_category: Target class
            activations: Layer activations (numpy array)
            grads: Layer gradients (numpy array)
            
        Returns:
            Weights for each channel (shape: batch x channels)
        """
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        
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
        
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=spatial_axes)
        eps = 0.000001
        
        # Expand dimensions for broadcasting: (batch, channels) -> (batch, channels, 1, 1) or (batch, channels, 1, 1, 1)
        sum_activations_expanded = sum_activations[(slice(None), slice(None)) + expand_shape]
        
        # Compute alpha_ij (equation 19)
        aij = grads_power_2 / (
            2 * grads_power_2 +
            sum_activations_expanded * grads_power_3 +
            eps
        )
        
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)
        
        # Compute final weights
        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=spatial_axes)
        
        return weights
