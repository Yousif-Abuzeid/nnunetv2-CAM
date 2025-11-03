"""
Main API for nnunetv2_cam.

This module provides the primary programmatic interface for generating
Class Activation Maps (CAMs) using pre-trained nnUNetv2 models.
"""

import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from tqdm import tqdm

from nnunetv2_cam.cam_core import compute_cam_with_sliding_window
from nnunetv2_cam.utils import save_cam_slices


def run_cam_for_prediction(
    predictor: nnUNetPredictor,
    input_files: Union[str, List[str]],
    output_folder: str,
    target_layer: str,
    target_class: int = 1,
    method: str = "gradcam",
    cam_type: str = "2d",
    device: torch.device = torch.device("cuda"),
    save_slices: bool = True,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Generate CAM heatmaps for nnUNetv2 predictions.

    This is the main entry point for programmatic use. It leverages the
    nnUNetv2 predictor's preprocessing and inference pipeline while
    computing CAMs using pytorch-grad-cam.

    Args:
        predictor: Initialized nnUNetPredictor instance
        input_files: Single file path or list of file paths to process
        output_folder: Directory to save CAM outputs
        target_layer: Name of the target layer for CAM computation
                      (e.g., 'encoder.stages.4.0')
        target_class: Target class index for CAM (default: 1, foreground)
        method: CAM method - 'gradcam' or 'gradcam++' (default: 'gradcam')
        cam_type: '2d' or '3d' (default: '2d')
        device: Torch device to use (default: cuda)
        save_slices: Whether to save individual slice visualizations (default: True)
        verbose: Print detailed progress information (default: False)

    Returns:
        List of CAM heatmap arrays, one per input file

    Example:
        >>> from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        >>> from nnunetv2_cam.api import run_cam_for_prediction
        >>>
        >>> predictor = nnUNetPredictor()
        >>> predictor.initialize_from_trained_model_folder(
        ...     model_folder='/path/to/model',
        ...     use_folds=(0, 1, 2, 3, 4),
        ...     checkpoint_name='checkpoint_final.pth'
        ... )
        >>>
        >>> heatmaps = run_cam_for_prediction(
        ...     predictor=predictor,
        ...     input_files='/path/to/input/image_0000.nii.gz',
        ...     output_folder='/path/to/output',
        ...     target_layer='encoder.stages.4.0',
        ...     target_class=1
        ... )
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert single file to list
    if isinstance(input_files, str):
        if os.path.isfile(input_files):
            input_files = [input_files]
        elif os.path.isdir(input_files):
            # If directory, find all image files
            input_files = _find_input_files(input_files)
        else:
            raise ValueError(f"Input path does not exist: {input_files}")

    # Set model to evaluation mode
    model = predictor.network.to(device).eval()

    if verbose:
        print(f"Processing {len(input_files)} files...")
        print(f"Target layer: {target_layer}")
        print(f"Target class: {target_class}")
        print(f"Method: {method}")
        print(f"CAM type: {cam_type}")

    # Process each input file
    heatmaps = []
    iterator = tqdm(input_files, desc="Processing files", disable=False)

    for input_file in iterator:
        if verbose:
            print(f"\nProcessing: {input_file}")

        # Get base filename for output
        base_name = Path(input_file).stem
        if base_name.endswith("_0000"):
            base_name = base_name[:-5]
        output_file = os.path.join(output_folder, f"{base_name}")

        # Preprocess input using predictor's preprocessing
        # This ensures identical preprocessing to normal nnUNetv2 prediction
        data, seg_prev_stage, properties, output_truncated = _preprocess_file(
            predictor, input_file, output_file
        )

        # Compute CAM using sliding window inference
        predicted_cam = compute_cam_with_sliding_window(
            model=model,
            data=data,
            target_layer_name=target_layer,
            target_class=target_class,
            method=method,
            device=device,
            configuration_manager=predictor.configuration_manager,
            label_manager=predictor.label_manager,
            list_of_parameters=predictor.list_of_parameters,
            tile_step_size=predictor.tile_step_size,
            use_mirroring=predictor.use_mirroring,
            allowed_mirroring_axes=predictor.allowed_mirroring_axes,
            cam_type=cam_type,
            verbose=verbose,
        )

        # Save slice visualizations
        if save_slices:
            save_cam_slices(
                predicted_cam=predicted_cam,
                original_data=data,
                output_file=output_truncated,
                properties=properties,
                configuration_manager=predictor.configuration_manager,
                verbose=verbose,
            )

        # Convert to numpy and store
        heatmap = predicted_cam.cpu().numpy()
        heatmaps.append(heatmap)

        if verbose:
            print(f"✓ Completed: {input_file}")

    return heatmaps


def _preprocess_file(predictor: nnUNetPredictor, input_file: str, output_file: str) -> tuple:
    """
    Preprocess a single input file using the predictor's preprocessing.

    This function leverages nnUNetv2's internal preprocessing to ensure
    identical preprocessing between CAM generation and regular prediction.

    Args:
        predictor: nnUNetPredictor instance
        input_file: Path to input image file
        output_file: Base output file path

    Returns:
        Tuple of (data, seg_prev_stage, properties, output_truncated)
    """
    from batchgenerators.utilities.file_and_folder_operations import subfiles
    from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles

    # Find all channel files for this case
    # nnUNetv2 expects files like: case_0000.nii.gz, case_0001.nii.gz, etc.
    base_path = Path(input_file)
    input_folder = base_path.parent
    case_id = base_path.stem

    # Remove channel suffix if present (e.g., _0000)
    if case_id.endswith("_0000"):
        case_id = case_id[:-5]

    # Find all channel files for this case
    all_files = subfiles(str(input_folder), suffix=base_path.suffix, join=True)
    case_files = [f for f in all_files if Path(f).stem.startswith(case_id)]
    case_files.sort()

    if len(case_files) == 0:
        # Single file, no channels
        case_files = [input_file]

    # Create data iterator
    list_of_lists = [case_files]
    output_filenames_truncated = [output_file]
    list_of_segs_from_prev_stage_files = [None]

    data_iterator = preprocessing_iterator_fromfiles(
        list_of_lists,
        list_of_segs_from_prev_stage_files,
        output_filenames_truncated,
        predictor.plans_manager,
        predictor.dataset_json,
        predictor.configuration_manager,
        num_processes=1,
        pin_memory=False,
        verbose=False,
    )

    # Get preprocessed data
    preprocessed = next(data_iterator)

    # Load data if it's saved to disk
    data = preprocessed["data"]
    if isinstance(data, str):
        data = torch.from_numpy(np.load(data))
        # Note: we don't delete the temp file here in case it's needed later
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    return (
        data,
        preprocessed.get("seg_from_prev_stage", None),
        preprocessed["data_properties"],
        preprocessed.get("ofile", output_file),
    )


def _find_input_files(input_folder: str) -> List[str]:
    """
    Find all valid input image files in a folder.

    Args:
        input_folder: Path to folder containing input images

    Returns:
        List of input file paths
    """
    from batchgenerators.utilities.file_and_folder_operations import subfiles

    valid_extensions = [".nii.gz", ".nii", ".mha", ".nrrd"]
    files = []

    for ext in valid_extensions:
        files.extend(subfiles(input_folder, suffix=ext, join=True))

    # Filter to only _0000 files (first channel) to avoid duplicates
    files = [f for f in files if "_0000" in Path(f).stem]
    files.sort()

    return files
