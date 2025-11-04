"""
Utility functions for nnunetv2_cam.

Includes functions for saving heatmaps, visualizations, and other helpers.
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm


def save_cam_slices(
    predicted_cam: torch.Tensor,
    original_data: torch.Tensor,
    output_folder: str,
    case_name: str,
    method: str = "gradcam",
    properties: dict = None,
    configuration_manager=None,
    verbose: bool = False,
) -> None:
    """
    Save CAM heatmap overlays as PNG slices.

    Creates a directory structure: output_folder/cam_{method}/case_name/slice_*.png

    Args:
        predicted_cam: CAM heatmap tensor (1, D, H, W) or (1, H, W)
        original_data: Original preprocessed data (C, D, H, W) or (C, H, W)
        output_folder: Base output folder path
        case_name: Name of the case (e.g., 'patient_001')
        method: CAM method name (e.g., 'gradcam', 'gradcam++')
        properties: nnUNet properties dict containing original shape info (optional)
        configuration_manager: nnUNet ConfigurationManager for resampling (optional)
        verbose: Whether to print debug information
    """

    # Resample CAM to original shape using nnUNet's resampling function
    if (
        properties is not None
        and configuration_manager is not None
        and "shape_after_cropping_and_before_resampling" in properties
    ):
        original_shape = properties["shape_after_cropping_and_before_resampling"]

        # Get spacing information
        current_spacing = (
            configuration_manager.spacing
            if len(configuration_manager.spacing) == len(original_shape)
            else [properties["spacing"][0], *configuration_manager.spacing]
        )

        # Use nnUNet's resampling function (same as used for predictions)
        predicted_cam = configuration_manager.resampling_fn_probabilities(
            predicted_cam, original_shape, current_spacing, properties["spacing"]
        )

        # Also resample original_data to match
        original_data = configuration_manager.resampling_fn_probabilities(
            original_data, original_shape, current_spacing, properties["spacing"]
        )

        # Convert back to torch if numpy
        if isinstance(predicted_cam, np.ndarray):
            predicted_cam = torch.from_numpy(predicted_cam)
        if isinstance(original_data, np.ndarray):
            original_data = torch.from_numpy(original_data)

    # Split into slices (dim=1 corresponds to slice dimension after nnUNet preprocessing)
    slices_cam = torch.split(predicted_cam, 1, dim=1)
    slices_ori = torch.split(original_data, 1, dim=1)

    # Create output directory: output_folder/cam_{method}/case_name/
    cam_folder = Path(output_folder) / f"cam_{method}" / case_name
    cam_folder.mkdir(parents=True, exist_ok=True)

    # Save each slice with progress bar
    slice_iterator = tqdm(
        zip(slices_cam, slices_ori),
        total=len(slices_cam),
        desc="Saving slices",
        disable=not verbose,
        leave=False,
        position=2,
    )

    for index, (slice_cam, slice_ori) in enumerate(slice_iterator):
        try:
            # Get slice data - handle multi-channel inputs
            slice_data = slice_ori.squeeze().cpu().numpy()  # Shape: (C, H, W) or (H, W)

            # Handle multi-channel images
            if slice_data.ndim == 3:
                # Multi-channel: average across channels for visualization
                # Shape: (C, H, W) -> (H, W)
                img_array_gray = slice_data.mean(axis=0)
            else:
                # Single channel: use as-is
                img_array_gray = slice_data

            # Normalize to [0, 1]
            img_array_gray_normalized = (img_array_gray - img_array_gray.min()) / (
                img_array_gray.max() - img_array_gray.min() + 1e-8
            )

            # Convert grayscale to RGB for CAM overlay
            img_array_rgb = (
                cv2.cvtColor(
                    (img_array_gray_normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB
                ).astype(np.float32)
                / 255.0
            )

            # Get CAM
            cam = slice_cam.cpu().squeeze().numpy()

            # Create overlay
            visualization = show_cam_on_image(img_array_rgb, cam, use_rgb=True)

            # Save
            image = Image.fromarray(visualization)
            image.save(cam_folder / f"slice_{index:04d}.png")

        except Exception as e:
            slice_iterator.write(f"✗ ERROR saving slice {index}: {e}")


def save_heatmap_nifti(
    heatmap: np.ndarray,
    output_path: str,
    reference_image_path: Optional[str] = None,
    affine: Optional[np.ndarray] = None,
) -> None:
    """
    Save heatmap as NIfTI file.

    Args:
        heatmap: Heatmap array
        output_path: Output file path
        reference_image_path: Path to reference image (for getting affine)
        affine: Affine transformation matrix (alternative to reference_image_path)
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError(
            "SimpleITK is required for saving NIfTI files. " "Install with: pip install SimpleITK"
        )

    # Convert to SimpleITK image
    if heatmap.ndim == 2:
        # 2D image
        sitk_image = sitk.GetImageFromArray(heatmap)
    elif heatmap.ndim == 3:
        # 3D image
        sitk_image = sitk.GetImageFromArray(heatmap.transpose(2, 1, 0))
    else:
        raise ValueError(f"Unsupported heatmap dimensionality: {heatmap.ndim}")

    # Set spacing and origin from reference if available
    if reference_image_path and os.path.exists(reference_image_path):
        ref_image = sitk.ReadImage(reference_image_path)
        sitk_image.SetSpacing(ref_image.GetSpacing())
        sitk_image.SetOrigin(ref_image.GetOrigin())
        sitk_image.SetDirection(ref_image.GetDirection())

    # Write to file
    sitk.WriteImage(sitk_image, output_path)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Normalize heatmap to [0, 1] range.

    Args:
        heatmap: Raw heatmap array

    Returns:
        Normalized heatmap
    """
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap.astype(np.float32)


def get_available_layers(model: torch.nn.Module, max_display: int = 20) -> list:
    """
    Get a list of available layer names from a model.

    Args:
        model: PyTorch model
        max_display: Maximum number of layers to display

    Returns:
        List of layer names
    """
    layer_names = [name for name, _ in model.named_modules() if name]
    return layer_names[:max_display]


def get_available_cam_methods() -> list:
    """
    Get a list of available CAM methods from pytorch-grad-cam.

    Returns:
        List of CAM method names
    """
    from nnunetv2_cam.cam_core import CAM_METHODS

    return list(CAM_METHODS.keys())
