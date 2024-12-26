# Applying weather stylization (Experimental)
!!! warning
    It is still in experimental. The method may change in the future.

Transfer weather style to images.

## Installation

```bash
# clone the forked repo
git clone https://github.com/guyleaf/contrastive-unpaired-translation

cd contrastive-unpaired-translation

# setup the conda environment
conda env create -f environment.yml
```

## Pre-trained weight
Download the compression file and place it in `checkpoints/` folder.

* Foggy, Rainy, Snowy: [Google Drive](https://drive.google.com/file/d/1aC13wPYu-2aykKXnMIAMpHhtkKLpvNDy/view?usp=drive_link)

```bash
mkdir checkpoints
tar -xf CUT_checkpoints.tar.xz
```

## Inference
Please fill the arguments `<...>` first.

```bash
conda activate cut

# Usage: $0 ROOT_FOLDER [DEVICE, e.g. 4] [EXP_NAME, e.g. weather_anti_uav_CUT_load_845_crop_768]
# EXP_NAME follows the name of the weight folder

# transfer style to fake weather images only in the folder with a suffix '_fake_style'
bash inference_scripts/inference_weather_anti_uav_f.sh \
    "<the folder contains UAV datasets>/Weather_Anti_UAV_T"

# OR

# transfer style to all weather images
bash inference_scripts/inference_weather_anti_uav.sh \
    "<the folder contains UAV datasets>/Weather_Anti_UAV_T"

# saved in Weather_Anti_UAV_T/stylized_cut_weather_anti_uav_CUT_load_845_crop_768
```

For more details about how to use it, see [the offical repository](https://github.com/taesungp/contrastive-unpaired-translation).
