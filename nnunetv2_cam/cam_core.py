"""
Core CAM computation logic using pytorch-grad-cam.

This module wraps pytorch-grad-cam to compute Class Activation Maps
with support for nnUNetv2's sliding window inference.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from tqdm import tqdm

# Import all available CAM methods from pytorch-grad-cam
try:
    from pytorch_grad_cam import (
        AblationCAM,
        EigenCAM,
        EigenGradCAM,
        FullGrad,
        GradCAM,
        GradCAMElementWise,
        HiResCAM,
        LayerCAM,
        ScoreCAM,
    )

    # Import our 3D-compatible implementations
    from nnunetv2_cam.custom_cams import GradCAMPlusPlus3D, XGradCAM3D
    from nnunetv2_cam.custom_cams.ablation_cam_3d import AblationCAM3D
    from nnunetv2_cam.custom_cams.seg_xres_cam import SegXResCAM

    # Map method names to classes (based on pytorch-grad-cam documentation)
    CAM_METHODS = {
        # Basic methods
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "gradcamelementwise": GradCAMElementWise,
        "gradcam++": GradCAMPlusPlus3D,  # Use our 3D-compatible version
        "xgradcam": XGradCAM3D,  # Use our 3D-compatible version
        "segxrescam": SegXResCAM,  # Seg-XRes-CAM
        # Perturbation-based methods
        "ablationcam": AblationCAM3D,  # Use our 3D-compatible version
        # Eigen-based methods
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        # Advanced methods
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
    }
except ImportError as e:
    # Fallback to basic methods if some are not available
    from pytorch_grad_cam import GradCAM

    from nnunetv2_cam.custom_cams import GradCAMPlusPlus3D, XGradCAM3D

    CAM_METHODS = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus3D,
        "xgradcam": XGradCAM3D,
    }
    print(f"Warning: Some CAM methods not available: {e}")


def get_available_cam_methods() -> list:
    """
    Get list of available CAM methods.

    Returns:
        List of available method names
    """
    return list(CAM_METHODS.keys())


class SemanticSegmentationTarget:
    """
    Target class for semantic segmentation CAM.

    This matches the implementation from the reference repository.
    It masks the model output and sums the values for the target category.
    """

    def __init__(self, category: int, mask: np.ndarray, scale_factor: float = 1.0):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        self.scale_factor = scale_factor

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return (model_output * self.mask).sum() * self.scale_factor


def get_target_layer(
    model: torch.nn.Module, target_layer_name: Union[str, List[str]]
) -> List[torch.nn.Module]:
    """
    Get the target layer(s) from the model by name.

    Args:
        model: The neural network model
        target_layer_name: Name(s) of the layer(s) (e.g., 'encoder.stages.4.0')

    Returns:
        List containing the target layer module(s)

    Raises:
        ValueError: If a target layer is not found
    """
    # Build a dictionary of all named modules
    module_dict = dict(model.named_modules())

    if isinstance(target_layer_name, str):
        target_layer_names = [target_layer_name]
    else:
        target_layer_names = target_layer_name

    target_layers = []
    for name in target_layer_names:
        if name not in module_dict:
            available_layers = [n for n, _ in model.named_modules() if n]
            raise ValueError(
                f"Target layer '{name}' not found in model.\n"
                f"Available layers: {available_layers[:]}"
            )
        target_layers.append(module_dict[name])

    return target_layers


def compute_cam_with_sliding_window(
    model: torch.nn.Module,
    data: torch.Tensor,
    target_layer_name: Union[str, List[str]],
    target_class: int,
    method: str,
    device: torch.device,
    configuration_manager,
    label_manager,
    list_of_parameters: List[dict],
    tile_step_size: float,
    use_mirroring: bool,
    allowed_mirroring_axes: Optional[Tuple[int, ...]],
    cam_type: str = "2d",
    verbose: bool = False,
    pool_size: Optional[int] = None,
    pool_mode: str = "max",
) -> torch.Tensor:
    """
    Compute CAM using sliding window inference, matching the reference implementation.

    This function replicates the exact logic from the MoriiHuang repository's
    predict_from_data_iterator method for CAM computation.

    Args:
        model: The neural network model
        data: Preprocessed input data (C, D, H, W) or (C, H, W)
        target_layer_name: Name of the target layer
        target_class: Target class index for CAM
        method: CAM method ('gradcam' or 'gradcam++')
        device: Torch device
        configuration_manager: nnUNet configuration manager
        label_manager: nnUNet label manager
        list_of_parameters: List of model parameters (for ensemble)
        tile_step_size: Step size for sliding window
        use_mirroring: Whether to use test-time augmentation
        allowed_mirroring_axes: Axes for mirroring
        cam_type: '2d' or '3d'
        verbose: Whether to print debug information
        pool_size: Optional pooling size for Seg-XRes-CAM
        pool_mode: Pooling mode for Seg-XRes-CAM ('max' or 'mean')

    Returns:
        CAM heatmap tensor
    """

    # Get target layers
    target_layers = get_target_layer(model, target_layer_name)

    # Pad data to match patch size
    cam_data, slicer_revert_padding = pad_nd_image(
        data,
        configuration_manager.patch_size,
        "constant",
        {"value": 0},
        True,
        None,
    )

    # Get sliding window slicers
    if cam_type == "2d":
        image_size = cam_data.shape[1:]
        slicers = _get_sliding_window_slicers(
            image_size, configuration_manager.patch_size, tile_step_size, verbose
        )
    else:
        image_size = data.shape[1:]

        slicers = _get_sliding_window_slicers(
            image_size, configuration_manager.patch_size, tile_step_size, verbose
        )

    # Initialize predicted CAM
    predicted_cam = torch.zeros((1, *cam_data.shape[1:]), dtype=torch.half, device=device)

    # Select CAM class from available methods
    method_name = method.lower()
    if method_name not in CAM_METHODS:
        available = ", ".join(CAM_METHODS.keys())
        raise ValueError(f"Unknown CAM method '{method}'. Available methods: {available}")
    cam_class = CAM_METHODS[method_name]

    # Prepare kwargs for CAM class
    cam_kwargs = {}
    if pool_size is not None:
        cam_kwargs["pool_size"] = pool_size
        cam_kwargs["pool_mode"] = pool_mode

    # Iterate over model parameters (for ensemble prediction)
    for params in list_of_parameters:
        # Load model parameters
        if hasattr(model, "_orig_mod"):
            model._orig_mod.load_state_dict(params)
        else:
            model.load_state_dict(params)

        predicted_cam_file = torch.zeros(
            (1, *cam_data.shape[1:]),
            dtype=torch.half,
            device=device,
        )
        n_predictions_cam_file = torch.zeros(cam_data.shape[1:], dtype=torch.half, device=device)

        # Create GradCAM object
        with cam_class(model, target_layers=target_layers, **cam_kwargs) as cam:
            # Process each sliding window patch
            for sl in tqdm(
                slicers, desc="Processing patches", disable=not verbose, leave=False,position=1
            ):
                workon = cam_data[sl][None]
                workon = workon.to(device, non_blocking=False)

                # Generate prediction mask for this patch
                with torch.no_grad():
                    prediction = _maybe_mirror_and_predict(
                        model, workon, use_mirroring, allowed_mirroring_axes
                    )[0]
                    predicted_probabilities = label_manager.apply_inference_nonlin(prediction)
                    if isinstance(predicted_probabilities, torch.Tensor):
                        predicted_probabilities = predicted_probabilities.cpu()
                    car_mask = (
                        label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
                        .unsqueeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                # Create mask for target class
                car_mask_float = np.float32(car_mask == target_class)
                targets = [SemanticSegmentationTarget(target_class, car_mask_float)]

                # Compute CAM for this patch
                cam_output = cam(input_tensor=workon, targets=targets)[0, :]
                grayscale_cam = torch.from_numpy(cam_output).to(device)

                # Accumulate CAM
                predicted_cam_file[sl] += grayscale_cam
                n_predictions_cam_file[sl[1:]] += 1

            # Normalize CAM
            if cam_type == "2d":
                predicted_cam_file /= n_predictions_cam_file
                predicted_cam_file *= torch.max(n_predictions_cam_file)
            else:
                predicted_cam_file /= n_predictions_cam_file
                predicted_cam_file *= 1.3

        predicted_cam += predicted_cam_file

    # Revert padding
    predicted_cam = predicted_cam[tuple([slice(None), *slicer_revert_padding[1:]])]

    # Average over ensemble
    predicted_cam /= len(list_of_parameters)

    return predicted_cam


def _get_sliding_window_slicers(
    image_size: Tuple[int, ...],
    patch_size: Tuple[int, ...],
    tile_step_size: float,
    verbose: bool = False,
) -> List[Tuple[slice, ...]]:
    """
    Generate sliding window slicers for the given image size.

    This replicates the _internal_get_sliding_window_slicers logic.
    """
    from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window

    slicers = []

    if len(patch_size) < len(image_size):
        # 2D patches on 3D volume
        assert len(patch_size) == len(image_size) - 1, (
            "if tile_size has less entries than image_size, "
            "len(tile_size) must be one shorter than len(image_size)"
        )
        steps = compute_steps_for_sliding_window(
            image_size[1:],
            patch_size,
            tile_step_size,
        )
        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicers.append(
                        tuple(
                            [
                                slice(None),
                                d,
                                *[slice(si, si + ti) for si, ti in zip((sx, sy), patch_size)],
                            ]
                        )
                    )
    else:
        # 3D patches
        steps = compute_steps_for_sliding_window(image_size, patch_size, tile_step_size)
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(
                        tuple(
                            [
                                slice(None),
                                *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), patch_size)],
                            ]
                        )
                    )

    return slicers


def _maybe_mirror_and_predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    use_mirroring: bool,
    allowed_mirroring_axes: Optional[Tuple[int, ...]],
) -> torch.Tensor:
    """
    Predict with optional test-time augmentation (mirroring).

    Replicates _internal_maybe_mirror_and_predict logic.
    """
    mirror_axes = allowed_mirroring_axes if use_mirroring else None
    prediction = model(x)

    if mirror_axes is not None:
        assert (
            max(mirror_axes) <= len(x.shape) - 3
        ), "mirror_axes does not match the dimension of the input!"

        num_predictions = 2 ** len(mirror_axes)
        if 0 in mirror_axes:
            prediction += torch.flip(model(torch.flip(x, (2,))), (2,))
        if 1 in mirror_axes:
            prediction += torch.flip(model(torch.flip(x, (3,))), (3,))
        if 2 in mirror_axes:
            prediction += torch.flip(model(torch.flip(x, (4,))), (4,))
        if 0 in mirror_axes and 1 in mirror_axes:
            prediction += torch.flip(model(torch.flip(x, (2, 3))), (2, 3))
        if 0 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(model(torch.flip(x, (2, 4))), (2, 4))
        if 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(model(torch.flip(x, (3, 4))), (3, 4))
        if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
            prediction += torch.flip(model(torch.flip(x, (2, 3, 4))), (2, 3, 4))
        prediction /= num_predictions

    return prediction
