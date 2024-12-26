# Applying image harmonization
Adjust the light and color of UAV samples to match the environment of the background image.

## Installation

```bash
# clone the forked repo
git clone https://github.com/guyleaf/PIH

cd PIH

# setup the conda environment
conda env create -f environment.yml

# download the pretrained weight
bash scripts/download_checkpoint.sh
```

## Inference
Please fill the arguments `<...>` first.

```bash
conda activate pytorch_pih

# inference
bash inference_scripts/Inference_Weather_Anti_UAV.sh \
    "<the folder contains UAV datasets>/Weather_Anti_UAV_T"

# OR

# inference with the specific GPU
bash inference_scripts/Inference_Weather_Anti_UAV.sh \
    "<the folder contains UAV datasets>/Weather_Anti_UAV_T" \
    "cuda:0"

# saved in Weather_Anti_UAV_T/harmonized
```
