import blenderproc as bproc  # noqa:F401, isort:skip, this should be at the top due to the check of blenderproc

import bpy

if __name__ == "__main__":
    desired_gpu_device_type = ["OPTIX", "CUDA", "HIP"]

    preferences = bpy.context.preferences.addons["cycles"].preferences

    for device_type in desired_gpu_device_type:
        devices = preferences.get_devices_for_type(device_type)
        if len(devices) == 0:
            continue

        print()
        print("Found device type:", device_type)
        for i, device in enumerate(devices):
            print(f"{i}. {device.name} of type {device.type}.")
