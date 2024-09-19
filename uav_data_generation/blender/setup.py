import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc

import os
from operator import methodcaller

import bpy
from blenderproc.api.types import Entity, MeshObject


def setup(
    scene_path: str,
    background_path: str,
    device_type: str,
    devices: list[int],
    motion_blur: bool = False,
    resolution: tuple[int, int] = (1920, 1920),
    max_samples: int = 1024,
    tile_size: int = 1024,
):
    bproc.init()

    # set render device
    use_only_cpu = device_type == "CPU"
    device_type = device_type if not use_only_cpu else None
    bproc.renderer.set_render_devices(
        use_only_cpu=use_only_cpu,
        desired_gpu_device_type=device_type if not use_only_cpu else None,
        desired_gpu_ids=devices,
    )

    # WORKAROUND: currently, the blenderproc api doesn't support enabling OpenImageDenoise
    bproc.renderer.set_denoiser(None)
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.view_layer.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = "OPENIMAGEDENOISE"

    print("\nLoading the background:", os.path.basename(background_path))
    bproc.world.set_world_background_hdr_img(background_path)

    # note: only objects in the obj_types can be loaded
    # otherwise, such as scene settings, they aren't loaded by blenderprc
    objs = bproc.loader.load_blend(
        scene_path, obj_types=["mesh", "light"], data_blocks=["objects", "materials"]
    )
    objs: list[Entity] = bproc.filter.all_with_type(objs, filtered_data_type=Entity)

    # Setup scene settingss
    bpy.context.scene.render.fps = 60
    bproc.camera.set_resolution(*resolution)
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.set_max_amount_of_samples(max_samples)
    bproc.renderer.set_noise_threshold(0.01)
    bpy.context.scene.cycles.tile_size = tile_size

    if motion_blur:
        print("Enabling motion blur")
        bproc.renderer.enable_motion_blur(motion_blur_length=0.5)

    # collect UAV models by custom property
    uav_objs: list[MeshObject] = bproc.filter.by_cp(
        objs, "UAV_model", True, filtered_data_type=MeshObject
    )
    uav_objs.sort(key=methodcaller("get_name"))

    # organize components for each uav model as dict
    uav_models: dict[str, list[MeshObject]] = {}
    for obj in uav_objs:
        uav_name = obj.get_name()
        uav_models[uav_name] = [obj] + sorted(
            obj.get_children(return_all_offspring=True), key=methodcaller("get_name")
        )

    assert len(uav_models) > 0, "UAV model is not found."
    print("\nFind", len(uav_models), "UAV models")

    # hide all objects by default
    for obj in objs:
        obj.hide()

    return objs, uav_models
