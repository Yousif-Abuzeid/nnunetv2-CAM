# nnunetv2_cam

**Class Activation Map (CAM) Generation for nnUNet v2 Models**

A standalone, external Python module for computing Class Activation Maps (CAMs) on models trained with [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet). This module **does not modify** nnUNetv2 source code and uses it as a dependency.

---

## 📑 Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Local Installation](#local-installation)
  - [Google Colab Installation](#google-colab-installation)
- [Quick Start](#quick-start)
  - [Python API](#python-api)
  - [Command Line](#command-line)
- [Usage Examples](#usage-examples)
- [Finding Target Layers](#finding-target-layers)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [CLI Reference](#cli-reference)
- [Citation](#citation)

---

## Features

- ✅ **Zero nnUNetv2 Modifications**: Works as an external library
- ✅ **Leverages Official Pipeline**: Uses nnUNetv2's preprocessing, inference, and postprocessing
- ✅ **Sliding Window Support**: Full support for nnUNet's patch-based inference
- ✅ **Multiple CAM Methods**: GradCAM and GradCAM++ (extensible to more)
- ✅ **2D and 3D Support**: Works with both 2D and 3D medical images
- ✅ **Ensemble Predictions**: Supports multi-fold ensemble inference
- ✅ **CLI and Python API**: Use from command line or integrate into your code

---

## Installation

### Prerequisites

- Python >= 3.9
- PyTorch >= 2.0.0
- nnUNetv2 >= 2.0
- pytorch-grad-cam >= 1.4.0

### Local Installation

```bash
cd nnunetv2_cam
pip install -e .
```

### Google Colab Installation

⚠️ **IMPORTANT**: You MUST restart the runtime after installation!

```python
# Cell 1: Install
!cd /content/nnunetv2_cam && pip install -e .

# Cell 2: RESTART RUNTIME
# Go to: Runtime → Restart runtime

# Cell 3: Test (after restart)
from nnunetv2_cam import run_cam_for_prediction
print("✅ Installation successful!")
```

**Why restart?** Google Colab and Jupyter notebooks require a runtime restart after installing packages for them to become importable.

---

## Quick Start

### Python API

```python
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2_cam import run_cam_for_prediction
import torch

# Initialize nnUNet predictor
predictor = nnUNetPredictor(device=torch.device('cuda'))
predictor.initialize_from_trained_model_folder(
    '/path/to/trained/model',
    use_folds=(0,),  # Use single fold for faster processing
    checkpoint_name='checkpoint_final.pth'
)

# Generate CAMs
heatmaps = run_cam_for_prediction(
    predictor=predictor,
    input_files='/path/to/input/image_0000.nii.gz',
    output_folder='/path/to/output',
    target_layer='encoder.stages.4.0',  # MUST specify!
    target_class=1,
    method='gradcam',
    cam_type='2d',
    verbose=True
)

print(f"Generated {len(heatmaps)} heatmaps")
```

### Command Line

```bash
nnunetv2_cam \
    -i /path/to/input/images \
    -o /path/to/output \
    -m /path/to/trained/model \
    -f 0 \
    --target-layer encoder.stages.4.0 \
    --target-class 1 \
    --verbose
```

---

## Usage Examples

### Example 1: Complete Google Colab Workflow

```python
# After installation and restart!

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2_cam import run_cam_for_prediction
import torch
import os

# Setup paths
MODEL = "/content/data/nnUNet_results/Dataset997/nnUNetTrainer__nnUNetPlans__3d_fullres"
INPUT = "/content/data/nnUNet_raw/Dataset997/imagesTs/"
OUTPUT = "/content/output_cams"
os.makedirs(OUTPUT, exist_ok=True)

# Initialize predictor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictor = nnUNetPredictor(device=device, verbose=True)
predictor.initialize_from_trained_model_folder(MODEL, use_folds=(0,))

# Generate CAMs
heatmaps = run_cam_for_prediction(
    predictor=predictor,
    input_files=INPUT,
    output_folder=OUTPUT,
    target_layer='encoder.stages.4.0',
    target_class=1,
    verbose=True
)

print(f"✅ Generated {len(heatmaps)} CAMs")
```

### Example 2: Using GradCAM++

```python
heatmaps = run_cam_for_prediction(
    predictor=predictor,
    input_files='/path/to/images',
    output_folder='/path/to/output',
    target_layer='encoder.stages.4.0',
    target_class=1,
    method='gradcam++',  # Use GradCAM++ instead
    cam_type='2d',
    verbose=True
)
```

### Example 3: 3D CAM with Custom Layer

```python
heatmaps = run_cam_for_prediction(
    predictor=predictor,
    input_files='/path/to/images',
    output_folder='/path/to/output',
    target_layer='decoder.stages.0.0',  # Decoder layer
    target_class=2,  # Different class
    method='gradcam',
    cam_type='3d',  # 3D CAM
    verbose=True
)
```

### Example 4: Processing Multiple Files

```python
# Process specific files
file_list = [
    '/data/case001_0000.nii.gz',
    '/data/case002_0000.nii.gz',
    '/data/case003_0000.nii.gz',
]

heatmaps = run_cam_for_prediction(
    predictor=predictor,
    input_files=file_list,
    output_folder='/output',
    target_layer='encoder.stages.4.0',
    target_class=1,
    verbose=True
)

# Analyze results
for i, (file, heatmap) in enumerate(zip(file_list, heatmaps)):
    print(f"File: {file}")
    print(f"  Shape: {heatmap.shape}")
    print(f"  Min: {heatmap.min():.3f}, Max: {heatmap.max():.3f}")
    print(f"  Mean: {heatmap.mean():.3f}")
```

---

## Finding Target Layers

### Method 1: List Layers in Python

```python
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

predictor = nnUNetPredictor(device=torch.device('cuda'))
predictor.initialize_from_trained_model_folder('/path/to/model', use_folds=(0,))

# Print first 30 layers
print("Available layers:")
for i, (name, _) in enumerate(predictor.network.named_modules(), 1):
    if name:
        print(f"{i:3d}. {name}")
        if i >= 30:
            break
```

### Method 2: Use CLI

```bash
nnunetv2_cam --list-layers \
    -m /path/to/model \
    -i /dummy -o /dummy --target-layer dummy
```

### Common Target Layers

For standard nnU-Net architectures (including U-Mamba):

| Layer | Description | Recommended Use |
|-------|-------------|-----------------|
| `encoder.stages.4.0` | Deepest encoder | ⭐ **Most semantic features** |
| `encoder.stages.3.0` | 4th encoder stage | Mid-level features |
| `encoder.stages.2.0` | 3rd encoder stage | Low-level features |
| `encoder.stages.1.0` | 2nd encoder stage | Very low-level features |
| `decoder.stages.0.0` | First decoder | After upsampling |
| `decoder.stages.1.0` | Second decoder | Mid-resolution |

💡 **Tip**: Start with `encoder.stages.4.0` - it usually gives the best results!

---

## Output Format

The tool generates two types of outputs:

### 1. Slice Visualizations (PNG)

- **Location**: `{output_folder}/cam/{case_name}/{case_name}_{slice_idx}.png`
- **Format**: Jet colormap overlaid on grayscale image
- **Example**: `output/cam/case001/case001_050.png`

### 2. Heatmap Arrays (NumPy)

- Returned by `run_cam_for_prediction()` as a list
- Each element is a NumPy array with shape matching preprocessed input
- Values normalized to [0, 1] range
- Can be saved for further analysis

```python
# Save heatmap to file
import numpy as np
np.save('/output/case001_cam.npy', heatmaps[0])

# Load later
loaded_cam = np.load('/output/case001_cam.npy')
```

---

## Troubleshooting

### ❌ Error: "ModuleNotFoundError: No module named 'nnunetv2_cam'"

**Cause**: Runtime needs restart after installation (Colab/Jupyter only)

**Solution**:
1. Install: `!cd /content/nnunetv2_cam && pip install -e .`
2. **Restart runtime**: `Runtime → Restart runtime`
3. Import: `from nnunetv2_cam import run_cam_for_prediction`

### ❌ Error: "Target layer not found"

**Cause**: Invalid layer name

**Solution**: Use `--list-layers` to see available layers:
```bash
nnunetv2_cam --list-layers -m /path/to/model -i /dummy -o /dummy --target-layer dummy
```

### ❌ Error: Missing --target-layer value

**Wrong**:
```bash
--target-layer --target-class 1  # Missing layer name!
```

**Correct**:
```bash
--target-layer encoder.stages.4.0 --target-class 1
```

### ❌ Out of Memory

**Solutions**:

1. Use fewer folds:
   ```python
   predictor.initialize_from_trained_model_folder(model, use_folds=(0,))
   ```

2. Increase step size (faster, uses less memory):
   ```python
   predictor = nnUNetPredictor(tile_step_size=0.75)
   ```

3. Use CPU:
   ```python
   predictor = nnUNetPredictor(device=torch.device('cpu'))
   ```

### ❌ CAM values are all zero

**Possible causes**:
- Target class doesn't exist in the segmentation
- Wrong target layer
- Model not properly trained

**Solutions**:
- Verify target class exists in your data
- Try different layers (start with `encoder.stages.4.0`)
- Check model predictions are working correctly

### 🔧 Quick Diagnostic

Run the diagnostic script:
```python
!python /content/nnunetv2_cam/diagnose_installation.py
```

---

## Advanced Usage

### Custom Preprocessing

```python
from nnunetv2_cam.api import run_cam_for_prediction

# Use custom predictor settings
predictor = nnUNetPredictor(
    tile_step_size=0.5,        # Overlap between patches
    use_gaussian=True,         # Gaussian importance weighting
    use_mirroring=True,        # Test-time augmentation
    perform_everything_on_gpu=True,
    device=torch.device('cuda'),
    verbose=True
)

predictor.initialize_from_trained_model_folder(
    '/path/to/model',
    use_folds=(0, 1, 2),       # Multi-fold ensemble
    checkpoint_name='checkpoint_best.pth'
)

heatmaps = run_cam_for_prediction(
    predictor=predictor,
    input_files='/path/to/images',
    output_folder='/path/to/output',
    target_layer='encoder.stages.4.0',
    target_class=1,
    method='gradcam',
    cam_type='2d',
    device=torch.device('cuda'),
    save_slices=True,
    verbose=True
)
```

### Analyzing Multiple Classes

```python
# Generate CAMs for multiple classes
classes_to_analyze = [1, 2, 3]
all_heatmaps = {}

for target_class in classes_to_analyze:
    print(f"\nGenerating CAMs for class {target_class}")
    heatmaps = run_cam_for_prediction(
        predictor=predictor,
        input_files='/path/to/images',
        output_folder=f'/output/class_{target_class}',
        target_layer='encoder.stages.4.0',
        target_class=target_class,
        verbose=True
    )
    all_heatmaps[target_class] = heatmaps

# Compare attention across classes
import numpy as np
for cls in classes_to_analyze:
    mean_attention = np.mean([h.mean() for h in all_heatmaps[cls]])
    print(f"Class {cls}: Mean attention = {mean_attention:.4f}")
```

### Saving Results to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

OUTPUT = '/content/drive/MyDrive/cam_outputs'
os.makedirs(OUTPUT, exist_ok=True)

heatmaps = run_cam_for_prediction(
    predictor=predictor,
    input_files='/content/data/images',
    output_folder=OUTPUT,  # Saves to Drive
    target_layer='encoder.stages.4.0',
    target_class=1,
    verbose=True
)
```

---

## CLI Reference

### Required Arguments

- `-i, --input`: Input folder or file path
- `-o, --output`: Output folder for CAM visualizations
- `-m, --model`: Path to trained nnUNet model folder
- `--target-layer`: Name of layer to compute CAM for

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-f, --folds` | 0 1 2 3 4 | Folds to use for ensemble |
| `-chk, --checkpoint` | checkpoint_final.pth | Checkpoint filename |
| `--target-class` | 1 | Target class index |
| `--method` | gradcam | CAM method (gradcam/gradcam++) |
| `--cam-type` | 2d | CAM type (2d/3d) |
| `--disable-tta` | False | Disable test-time augmentation |
| `-step_size` | 0.5 | Sliding window step size |
| `-device` | cuda | Device (cuda/cpu/mps) |
| `--verbose` | False | Print detailed progress |
| `--list-layers` | False | List available layers and exit |
| `--no-save-slices` | False | Don't save PNG slices |

### Examples

**Basic usage**:
```bash
nnunetv2_cam -i /data/images -o /output -m /model --target-layer encoder.stages.4.0
```

**Single fold, verbose**:
```bash
nnunetv2_cam -i /data/images -o /output -m /model -f 0 --target-layer encoder.stages.4.0 --verbose
```

**GradCAM++ with 3D**:
```bash
nnunetv2_cam -i /data/images -o /output -m /model --target-layer encoder.stages.4.0 --method gradcam++ --cam-type 3d
```

**List layers**:
```bash
nnunetv2_cam -m /model --list-layers -i /dummy -o /dummy --target-layer dummy
```

---

## Architecture

```
nnunetv2_cam/
├── __init__.py          # Package initialization
├── api.py               # Main programmatic interface
├── cam_core.py          # CAM computation logic
├── cli.py               # Command-line interface
├── utils.py             # Helper functions
├── example.py           # Usage examples
└── test_integration.py  # Integration tests
```

### How It Works

1. **Initialization**: Receives initialized `nnUNetPredictor` instance
2. **Preprocessing**: Uses nnUNet's `preprocessing_iterator_fromfiles` for identical preprocessing
3. **Sliding Window**: Replicates nnUNet's sliding window logic
4. **CAM Computation**: For each patch:
   - Generates prediction using nnUNet inference
   - Computes CAM using pytorch-grad-cam
   - Accumulates across overlapping patches
5. **Postprocessing**: Normalizes and saves visualizations

---

## Citation

If you use this tool, please cite:

**nnU-Net:**
```bibtex
@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

**Grad-CAM:**
```bibtex
@inproceedings{selvaraju2017grad,
  title={Grad-cam: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={618--626},
  year={2017}
}
```

---

## License

Apache License 2.0

---

## Contributing

Contributions are welcome! Please open an issue or pull request.

---

## Acknowledgments

- **nnUNet Team**: For the excellent nnUNet framework
- **pytorch-grad-cam**: For the CAM implementation library
- **Reference**: Based on insights from MoriiHuang's nnUNet-UAMT-DA-GRADCAM

---

## Support

If you encounter any issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Run the diagnostic script: `python diagnose_installation.py`
3. Check that all dependencies are installed: `pip list | grep -E "torch|nnunet|grad-cam"`
4. For Google Colab: Make sure you restarted the runtime after installation
5. Open an issue with details about your error

---

**Happy CAM generation! 🔥**
