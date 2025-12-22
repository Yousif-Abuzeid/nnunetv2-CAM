"""
Main API for nnunetv2_cam.

This module provides the primary programmatic interface for generating
Class Activation Maps (CAMs) using pre-trained nnUNetv2 models.
"""

import os
from pathlib import Path
from typing import List, Union,Optional

import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from tqdm import tqdm

from nnunetv2_cam.cam_core import compute_cam_with_sliding_window
from nnunetv2_cam.utils import save_cam_slices
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice


def _resample_cam_to_original_shape(
    predicted_cam: torch.Tensor,
    properties: dict,
    configuration_manager,
    plans_manager=None,
    label_manager=None,
) -> torch.Tensor:
    """
    Resample CAM heatmap back to original image shape.

    Uses nnUNet's resampling function to match the output shape with
    the original prediction output (same dimensions as saved predictions).
    Also performs revert cropping and revert transpose if necessary.

    Args:
        predicted_cam: CAM tensor in resampled/preprocessed space
        properties: nnUNet properties dict containing shape/spacing info
        configuration_manager: nnUNet ConfigurationManager for resampling
        plans_manager: nnUNet PlansManager for transpose information
        label_manager: nnUNet LabelManager for reverting cropping

    Returns:
        Resampled CAM tensor in original image space
    """
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
        if isinstance(predicted_cam, torch.Tensor):
            predicted_cam = predicted_cam.cpu().numpy()
        predicted_cam = configuration_manager.resampling_fn_probabilities(
            predicted_cam, original_shape, current_spacing, properties["spacing"]
        )

        # Convert back to torch if numpy
        if isinstance(predicted_cam, np.ndarray):
            predicted_cam = torch.from_numpy(predicted_cam)

        # Revert cropping
        # Revert cropping
        if "bbox_used_for_cropping" in properties:
             # Manually revert cropping to ensure background is 0 (and not 1 as LabelManager does for probs)
             shape_original = properties["shape_before_cropping"]
             bbox = properties["bbox_used_for_cropping"]
             
             new_shape = (predicted_cam.shape[0], *shape_original)
             probs_reverted = torch.zeros(
                 new_shape, dtype=predicted_cam.dtype, device=predicted_cam.device
             )
             
             slicer = bounding_box_to_slice(bbox)
             # correct slicing for channel + spatial
             full_slicer = tuple([slice(None)] + list(slicer))
             
             probs_reverted[full_slicer] = predicted_cam
             predicted_cam = probs_reverted

        # Revert transpose
        if plans_manager is not None and "transpose_backward" in plans_manager.plans.keys():
             # Check if transpose_backward is needed (it usually is for 3D)
             # The existing nnunet code accesses it via plans_manager.transpose_backward which might be a property or in plans
             # Let's check how it's accessed in export_prediction.py. It uses plans_manager.transpose_backward.
             
             # Note: predicted_cam is (C, ...) or (1, ...)
             # transpose_backward is for spatial dimensions. We need to offset by 1 for the channel dim.
             transpose_order = [0] + [i + 1 for i in plans_manager.transpose_backward]
             
             if isinstance(predicted_cam, torch.Tensor):
                 predicted_cam = predicted_cam.permute(*transpose_order)
             else:
                 predicted_cam = predicted_cam.transpose(transpose_order)

    return predicted_cam


def run_cam_for_prediction(
    predictor: nnUNetPredictor,
    input_files: Union[str, List[str]],
    output_folder: str,
    target_layer: Union[str, List[str]],
    target_class: int = 1,
    method: str = "gradcam",
    cam_type: str = "2d",
    device: torch.device = torch.device("cuda"),
    save_slices: bool = True,
    verbose: bool = False,
    pool_size: Optional[int] = None,
    pool_mode: str = "max",
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
        target_layer: Name(s) of the target layer(s) for CAM computation
                      (e.g., 'encoder.stages.4.0' or ['encoder.stages.4.0', 'decoder.stages.0.0'])
        target_class: Target class index for CAM (default: 1, foreground)
        method: CAM method - any method from pytorch-grad-cam (e.g., 'gradcam',
                'gradcam++', 'eigencam', 'layercam', etc.) (default: 'gradcam')
        cam_type: '2d' or '3d' (default: '2d')
        device: Torch device to use (default: cuda)
        save_slices: Whether to save individual slice visualizations (default: True)
        verbose: Print detailed progress information (default: False)
        pool_size: Optional pooling size for Seg-XRes-CAM (default: None)
        pool_mode: Pooling mode for Seg-XRes-CAM ('max' or 'mean') (default: 'max')

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
        ...     target_class=1,
        ...     method='segxrescam',
        ...     pool_size=2,
        ...     pool_mode='mean'
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

    print(
        """
If you find this tool useful, please consider citing:

@misc{abuzeid2025xaidrivendiagnosisgeneralizationfailure,
      title={XAI-Driven Diagnosis of Generalization Failure in State-Space Cerebrovascular Segmentation Models: A Case Study on Domain Shift Between RSNA and TopCoW Datasets}, 
      author={Youssef Abuzeid and Shimaa El-Bana and Ahmad Al-Kabbany},
      year={2025},
      eprint={2512.13977},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.13977}, 
}
"""
    )

    if verbose:
        print(f"Processing {len(input_files)} files...")
        print(f"Target layer: {target_layer}")
        print(f"Target class: {target_class}")
        print(f"Method: {method}")
        print(f"CAM type: {cam_type}")

    # Process each input file
    heatmaps = []
    iterator = tqdm(
        input_files, desc="Processing files", disable=not verbose, leave=True,position=0
    )

    for input_file in iterator:
        if verbose:
            iterator.set_postfix_str(f"Processing: {Path(input_file).name}")

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
            pool_size=pool_size,
            pool_mode=pool_mode,
        )

        # Resample CAM back to original shape (same as nnUNet does for predictions)
        resampled_cam = _resample_cam_to_original_shape(
            predicted_cam=predicted_cam,
            properties=properties,
            configuration_manager=predictor.configuration_manager,
            plans_manager=predictor.plans_manager,
            label_manager=predictor.label_manager,
        )
        # Also resample original data to match
        resampled_data = _resample_cam_to_original_shape(
            predicted_cam=data,
            properties=properties,
            configuration_manager=predictor.configuration_manager,
            plans_manager=predictor.plans_manager,
            label_manager=predictor.label_manager,
        )

        # Save slice visualizations (now both CAM and data are already resampled)
        if save_slices:
            save_cam_slices(
                predicted_cam=resampled_cam,
                original_data=resampled_data,
                output_folder=output_folder,
                case_name=base_name,
                method=method,
                properties=None,  # Pass None to skip resampling in save function
                configuration_manager=None,
                verbose=verbose,
            )

        # Convert to numpy and store (now properly resampled)
        heatmap = resampled_cam.cpu().numpy()
        heatmaps.append(heatmap)

        if verbose:
            iterator.write(f"✓ Completed: {Path(input_file).name}")

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
