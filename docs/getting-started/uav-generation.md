# Generating UAV samples
## Configuring the parameters

Please fill the parameters `<...>` (highlight) in the config file before running.

For more details, see [Parameters in User guides](../user-guides/parameters.md).

```python title="configs/weather_anti_uav_t.py" linenums="1" hl_lines="14 20 36"
--8<-- "configs/weather_anti_uav_t.py"
```

## Running with the demo config

!!! note "Note: If you modify the source code of the `uav_data_generation` project..."
    Please reinstall it by executing `bash scripts/reinstall.sh` before generating UAV samples.

    Blenderproc doesn't accept `-e` option.

```bash
conda activate uav_data_generation

# generating the UAV samples with GPU 0
blenderproc run -- "scripts/blender/generate_foregrounds.py" \
    configs/weather_anti_uav_t.py

# OR

# generating the UAV samples with GPU 1
blenderproc run -- "scripts/blender/generate_foregrounds.py" \
    configs/weather_anti_uav_t.py \
    --devices 1

# saved in Weather_Anti_UAV_T/foregrounds and Weather_Anti_UAV_T/annotations/foreground.json
```

For more details about how to use it, run with `-h` option or see the arguments in the script, `scripts/blender/generate_foregrounds.py`.

!!! tip "Tip: If you want to resume from the interruption..."
    Please see [Checkpoint in User guides](../user-guides/checkpoint.md).
