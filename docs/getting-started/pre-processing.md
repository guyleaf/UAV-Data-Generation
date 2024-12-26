# Pre-processing
## Background dataset for demo
Please fill the arguments `<...>` first. The command below follows these settings:

* Maximum samples per weather: 300
* Only use clear/cloudy to fill other weathers (ignore other existed weathers)

!!! note
    Please remember the output folder `$out` which will be your final UAV dataset.

```bash
root="<the folder contains weather datasets>/FWID_Image2Weather"
out="<the folder contains UAV datasets>/Weather_Anti_UAV_T"

conda activate uav_data_generation

python scripts/pre_processings/prepare_backgrounds.py \
    "$root/Image2Weather" "$root/FWID" "$out/backgrounds" \
    --max-samples 300 --fake-only
```

For more details about how to use it, run with `-h` option or see the arguments in the script, `scripts/pre_processings/prepare_backgrounds.py`.

## Blender scene and assets
By default, all rotary-wing UAVs are enabled and the textures are used in randomization.

For more details about blender scene:

* The system design? see [Blender in Design](../design/blender.md).
* How to customize it? see [Blender customization in User guides](../user-guides/blender.md).
