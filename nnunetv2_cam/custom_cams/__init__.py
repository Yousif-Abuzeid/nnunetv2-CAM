"""
Custom CAM implementations with 3D support.

This module contains modified versions of pytorch-grad-cam methods
that have been adapted to work with 3D medical imaging volumes.
"""

from nnunetv2_cam.custom_cams.grad_cam_plusplus_3d import GradCAMPlusPlus3D
from nnunetv2_cam.custom_cams.xgrad_cam_3d import XGradCAM3D

__all__ = ["GradCAMPlusPlus3D", "XGradCAM3D"]
