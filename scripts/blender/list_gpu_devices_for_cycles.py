import blenderproc as bproc  # noqa:F401, isort:skip, this should be at the top due to the check of blenderproc

import bpy
from uav_data_generation.logging import get_logger

if __name__ == "__main__":
    desired_gpu_device_type = ["OPTIX", "CUDA", "HIP"]

    logger = get_logger()
    preferences = bpy.context.preferences.addons["cycles"].preferences

    for device_type in desired_gpu_device_type:
        devices = preferences.get_devices_for_type(device_type)
        if len(devices) == 0:
            continue

        logger.info(f"\nFound device type: {device_type}")
        for i, device in enumerate(devices):
            logger.info(f"{i}. {device.name} of type {device.type}.")
