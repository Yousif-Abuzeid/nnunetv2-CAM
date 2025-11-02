"""
Command-line interface for nnunetv2_cam.

This module provides a CLI that mimics nnUNetv2_predict arguments
while adding CAM-specific options.
"""

import argparse
import sys

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from nnunetv2_cam.api import run_cam_for_prediction
from nnunetv2_cam.utils import get_available_layers


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Class Activation Maps (CAMs) for nnUNet v2 predictions. "
        "This tool extends nnUNetv2_predict with CAM generation capabilities."
    )

    # Input/Output arguments
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="Input folder containing images. Files must follow nnUNetv2 naming conventions "
        "(e.g., case_0000.nii.gz, case_0001.nii.gz for multi-channel inputs).",
    )
    parser.add_argument(
        "-o",
        type=str,
        required=True,
        help="Output folder for CAM visualizations. Will be created if it doesn't exist.",
    )

    # Model arguments
    parser.add_argument(
        "-m",
        type=str,
        required=True,
        help="Path to trained nnUNet model folder. Should contain fold_X subfolders.",
    )
    parser.add_argument(
        "-f",
        nargs="+",
        type=str,
        required=False,
        default=["0", "1", "2", "3", "4"],
        help="Folds to use for ensemble prediction. Default: 0 1 2 3 4",
    )
    parser.add_argument(
        "-chk",
        type=str,
        required=False,
        default="checkpoint_final.pth",
        help="Checkpoint name. Default: checkpoint_final.pth",
    )

    # CAM-specific arguments
    parser.add_argument(
        "--target-layer",
        type=str,
        required=True,
        help="Name of target layer for CAM computation (e.g., 'encoder.stages.4.0'). "
        "Use --list-layers to see available layers.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=1,
        help="Target class index for CAM. Default: 1 (foreground)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
        choices=["gradcam", "gradcam++"],
        help="CAM method to use. Default: gradcam",
    )
    parser.add_argument(
        "--cam-type",
        type=str,
        default="2d",
        choices=["2d", "3d"],
        help="CAM type: '2d' for 2D patches on 3D volumes, '3d' for 3D patches. Default: 2d",
    )

    # Inference arguments
    parser.add_argument(
        "--disable-tta",
        action="store_true",
        default=False,
        help="Disable test-time augmentation (mirroring). Faster but less accurate.",
    )
    parser.add_argument(
        "-step_size",
        type=float,
        default=0.5,
        help="Step size for sliding window prediction. Default: 0.5",
    )

    # Device arguments
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use for inference. Default: cuda",
    )

    # Utility arguments
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="List available layer names in the model and exit. Useful for finding target layers.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress information."
    )
    parser.add_argument(
        "--no-save-slices",
        action="store_true",
        help="Don't save individual slice visualizations (only useful for debugging).",
    )

    args = parser.parse_args()

    # Print citation
    print(
        "\n" + "=" * 70 + "\nnnunetv2_cam: CAM Generation for nnU-Net v2\n"
        "=" * 70 + "\n\nWhen using this tool, please cite:\n"
        "  - nnU-Net: Isensee et al., Nature Methods 18(2), 203-211 (2021)\n"
        "  - Grad-CAM: Selvaraju et al., ICCV 2017\n" + "=" * 70 + "\n"
    )

    # Process folds argument
    args.f = [i if i == "all" else int(i) for i in args.f]

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    # Initialize predictor
    if args.verbose:
        print(f"\n1. Loading model from: {args.m}")
        print(f"   Using folds: {args.f}")
        print(f"   Checkpoint: {args.chk}")

    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_gpu=(args.device == "cuda"),
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    try:
        predictor.initialize_from_trained_model_folder(
            args.m, use_folds=args.f, checkpoint_name=args.chk
        )
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    # List layers if requested
    if args.list_layers:
        print("\n" + "=" * 70)
        print("Available layers in model:")
        print("=" * 70)
        layers = get_available_layers(predictor.network, max_display=50)
        for i, layer in enumerate(layers, 1):
            print(f"{i:3d}. {layer}")
        print("=" * 70)
        print(f"\nShowing first 50 layers. Total: {len(list(predictor.network.named_modules()))}")
        print("\nCommon target layers for nnU-Net:")
        print("  - encoder.stages.4.0  (deepest encoder layer)")
        print("  - encoder.stages.3.0  (4th encoder stage)")
        print("  - decoder.stages.0.0  (1st decoder stage)")
        print("=" * 70 + "\n")
        sys.exit(0)

    # Validate target layer
    try:
        dict(predictor.network.named_modules())[args.target_layer]
    except KeyError:
        print(f"\nERROR: Target layer '{args.target_layer}' not found in model!")
        print("Use --list-layers to see available layers.\n")
        sys.exit(1)

    # Run CAM generation
    if args.verbose:
        print("\n2. Generating CAMs")
        print(f"   Input: {args.i}")
        print(f"   Output: {args.o}")
        print(f"   Target layer: {args.target_layer}")
        print(f"   Target class: {args.target_class}")
        print(f"   Method: {args.method}")
        print(f"   CAM type: {args.cam_type}")
        print()

    try:
        heatmaps = run_cam_for_prediction(
            predictor=predictor,
            input_files=args.i,
            output_folder=args.o,
            target_layer=args.target_layer,
            target_class=args.target_class,
            method=args.method,
            cam_type=args.cam_type,
            device=device,
            save_slices=not args.no_save_slices,
            verbose=args.verbose,
        )

        print(f"\n✓ Successfully generated CAMs for {len(heatmaps)} cases")
        print(f"  Output saved to: {args.o}\n")

    except Exception as e:
        print(f"\nERROR: CAM generation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
