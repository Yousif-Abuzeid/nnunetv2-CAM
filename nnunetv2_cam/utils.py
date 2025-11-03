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


def save_cam_slices(
    predicted_cam: torch.Tensor,
    original_data: torch.Tensor,
    output_file: str,
    properties: dict = None,
    configuration_manager=None,
    verbose: bool = False,
) -> None:
    """
    Save CAM heatmap overlays as PNG slices.

    This replicates the slice-by-slice saving logic from the reference implementation.
    If properties and configuration_manager are provided, CAM will be resampled to
    original shape using nnUNet's resampling functions.

    Args:
        predicted_cam: CAM heatmap tensor (1, D, H, W) or (1, H, W)
        original_data: Original preprocessed data (C, D, H, W) or (C, H, W)
        output_file: Base output file path (will be modified to create cam folder)
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

    # Find where to insert 'cam' in the path
    last_infer_index = output_file.rfind("infer")

    if verbose:
        print(f"DEBUG: output_file = '{output_file}'")
        print(f"DEBUG: last_infer_index = {last_infer_index}")

    if last_infer_index == -1:
        # Fallback: use the output file path directly and add _cam
        output_path = Path(output_file)
        cam_folder = output_path.parent / f"{output_path.stem}_cam"
        cam_folder.mkdir(parents=True, exist_ok=True)
        file_prefix = output_path.stem
    else:
        # Replace 'infer' with 'cam'
        tmp_path = (
            output_file[:last_infer_index] + "cam" + output_file[last_infer_index + len("infer") :]
        )

        if verbose:
            print(f"DEBUG: tmp_path = '{tmp_path}'")

        # Create directory structure
        path_parts = tmp_path.rsplit("/", 1)
        if len(path_parts) == 2:
            new_path = Path(path_parts[0])
            file_name = path_parts[1]
        else:
            new_path = Path(".")
            file_name = path_parts[0]

        cam_folder = new_path / file_name
        cam_folder.mkdir(parents=True, exist_ok=True)
        file_prefix = file_name

        if verbose:
            print(f"DEBUG: cam_folder = '{cam_folder}'")

    if verbose:
        print(f"DEBUG: Saving {len(slices_cam)} slices to {cam_folder}")

    # Save each slice
    index = 0
    for slice_cam, slice_ori in zip(slices_cam, slices_ori):
        try:
            # Normalize original image to [0, 1]
            img_array_gray = slice_ori.squeeze().cpu().numpy()
            img_array_gray_normalized = (img_array_gray - img_array_gray.min()) / (
                img_array_gray.max() - img_array_gray.min() + 1e-8
            )

            # Convert grayscale to RGB
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
            image.save(f"{cam_folder}/{file_prefix}_{index}.png")

            if verbose:
                print(f"✓ Saved slice {index}")
        except Exception as e:
            print(f"✗ ERROR saving slice {index}: {e}")
        index += 1


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
