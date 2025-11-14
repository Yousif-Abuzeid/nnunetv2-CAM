"""
nnunetv2_cam: Class Activation Map Generation for nnUNet v2 Models

A standalone external module for computing Class Activation Maps (CAMs)
on models trained with nnUNetv2. This module does not modify nnUNetv2
source code and uses it as a dependency.

Usage:
    from nnunetv2_cam.api import run_cam_for_prediction
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor()
    predictor.initialize_from_trained_model_folder(model_folder, folds)

    heatmap = run_cam_for_prediction(
        predictor=predictor,
        input_files=input_files,
        output_folder=output_folder,
        target_layer="encoder.stages.4.0",
        target_class=1,
        method='gradcam'
    )
"""

__version__ = "0.1.4.1"
__author__ = "Youssef Abuzeid"

from nnunetv2_cam.api import run_cam_for_prediction

__all__ = ["run_cam_for_prediction"]
