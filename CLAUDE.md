# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

### Initial Setup
```bash
# Create conda environment and install dependencies
bash scripts/setup.sh

# After code changes, reinstall package in Blender
bash scripts/reinstall.sh
```

### Manual Setup (alternative)
```bash
# Create conda environment
conda env create -f environment.yml
conda activate uav_data_generation

# Install package in Blender environment
blenderproc pip install .
```

### Environment Management
- **Environment**: Uses conda environment named `uav_data_generation`
- **Python version**: 3.10 required
- **CUDA**: Requires CUDA 12.1 compatible GPU
- **Blender**: Uses Blender 3.5.1 via BlenderProc ~2.7.2
- **PyTorch**: 2.4.1 with CUDA support

### Activate Environment
```bash
conda activate uav_data_generation
```

## Core Generation Commands

### UAV Data Generation
```bash
# Generate UAV samples using specific config
blenderproc run -- "scripts/blender/generate_foregrounds.py" configs/robust_anti_uav_low.py --devices 0

# Demo with material randomization
blenderproc run -- "scripts/blender/demo_randomization.py" configs/robust_anti_uav_demo.py --out-dir <output_path> --devices 0 --samples 10

# Demo multi-view with UAV model and random material
blenderproc run -- "scripts/blender/demo_uav_models.py" "configs/robust_anti_uav_demo.py" --out-dir <output_path> --devices 0 --samples 30

# Batch generation scripts
bash scripts/blender/generate_uav_samples_low.sh
bash scripts/blender/generate_uav_samples_s_low.sh
```

### Dataset Preparation Pipeline
```bash
# Full pipeline for Robust Anti-UAV Low dataset
bash scripts/prepare_robust_anti_uav_low.sh

# Full pipeline for Robust Anti-UAV S Low dataset
bash scripts/prepare_robust_anti_uav_s_low.sh

# Prepare Image2Weather dataset
bash scripts/datasets/prepare_image2weather.sh <root_path>

# Prepare backgrounds from weather datasets
python scripts/pre_processings/prepare_backgrounds.py <root_paths> <output_path> --dataset-names <root_path_names> --max-samples <max_samples_per_label>
```

### Validation and Post-Processing
```bash
# Validate COCO dataset format
python scripts/post_processings/validate_coco_dataset.py <dataset_root>

# Post-process generated data
python scripts/post_processings/post_process.py <root_path> --val-ratio <val_subset_ratio, e.g. 0.3333> 
```

## Architecture Overview

### Configuration-Driven Generation
- **Config files**: Located in `configs/` directory, each inherits from `BaseConfig`
- **Key configs**: 
  - `robust_anti_uav_low.py` - Low resolution generation
  - `robust_anti_uav_demo.py` - Demo/testing configuration
  - `robust_anti_uav_s.py` - Standard resolution generation
- **Pattern**: Each config defines scene paths, background paths, UAV properties, and generation parameters

### 3D Pipeline Architecture
- **Blender Integration**: Uses BlenderProc for 3D rendering and scene management
- **UAV Models**: Stored in `.blend` files with randomizable properties (geometry, materials, motion)
- **Background System**: Supports various background types (studio lights, HDRI, weather datasets)
- **Checkpoint System**: Built-in checkpointing for resuming long-running generation tasks

### Data Flow
1. **Background Preparation**: Extract backgrounds from real-world datasets (nuScenes, Image2Weather, etc.)
2. **UAV Generation**: Generate 3D UAV samples with randomized properties using Blender
3. **Weather Transfer**: Deprecated. Apply weather stylization using diffusion models and ControlNet
4. **Image Harmonization**: Apply image harmonization using Adobe PIH (not in current repo)
5. **Post-Processing**: Convert to COCO format, validate annotations, combine datasets

### Key Modules
- **`uav_data_generation.blender`**: Core 3D generation logic and Blender integration
  - `config.py` - Configuration system and base classes
  - `setup.py` - Blender scene setup and GPU configuration  
  - `drone.py` - UAV property randomization
  - `camera.py` - Camera positioning and alignment
  - `checkpoint.py` - Progress saving/resuming
  - `coco.py` - COCO format annotation writer
  - `utils/` - Geometry, material, and mesh utilities
- **`uav_data_generation.datasets`**: Deprecated
- **`uav_data_generation.utils`**: Common utilities (e.g. dataset, io, etc.)

### Key Scripts
- **`scripts/blender/`**: Generation entry points and demos
- **`scripts/datasets/`**: Dataset preparation and conversion utilities
- **`scripts/pre_processings/`**: Background and data preprocessing
- **`scripts/post_processings/`**: Validation and format conversion

### GPU and Device Management
- All generation scripts support `--devices` parameter for GPU selection
- Uses CUDA for PyTorch operations and Cycles rendering in Blender
- Automatic GPU detection available via `scripts/blender/list_gpu_devices_for_cycles.py`

## Documentation and Development

### Documentation (out-of-date)
```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

### Research Notebooks
Located in `notebooks/` for experimentation:
- Stable Diffusion experiments (`sd15_controlnet.ipynb`, `sdxl10.ipynb`)
- Dataset tutorials (`nuscenes_tutorial.ipynb`)
- Image processing workflows (`mask_generation.ipynb`, `image_composition.ipynb`)

### No Formal Testing
The project relies on validation scripts and demo outputs rather than unit tests. Use validation commands after generation to verify output quality.

## Important Architectural Notes

### Configuration Inheritance Pattern
All config files inherit from `BaseConfig` class in `uav_data_generation.blender.config`. When creating new configs:
- Extend `BaseConfig` and override specific parameters
- All randomization ranges, paths, and generation settings are defined in config
- Config validation happens automatically during instantiation

### Dual Python Environment Architecture
The project operates in two Python environments:
1. **Conda environment** (`uav_data_generation`) - For preprocessing and dataset preparation
2. **Blender environment** - Managed by BlenderProc, where the core generation package is installed

### Checkpoint and Resumability
Generation supports checkpointing via `checkpoint.py`:
- Automatically saves progress during long-running generations
- Checkpoint files contain generation state and completed sample indices
- `scripts/blender/generate_foregrounds.py` supports resuming from latest checkpoint in `--out-dir` using `--resume` or specific checkpoint using `--resume --checkpoint <checkpoint_path>`
