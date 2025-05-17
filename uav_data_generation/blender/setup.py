from typing import Optional

import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc

import os
from operator import methodcaller

import bpy
from blenderproc.api.types import Entity

from .config import SceneConfig


def setup(
    scene_path: str,
    background_path: str,
    device_type: str,
    devices: list[int],
    motion_blur: bool = False,
    resolution: tuple[int, int] = (1920, 1920),
    max_samples: int = 1024,
    tile_size: int = 1024,
    models: Optional[list[str]] = None,
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

    if SceneConfig.denoiser == "OPENIMAGEDENOISE":
        # WORKAROUND: currently, the blenderproc api doesn't support enabling OpenImageDenoise
        bproc.renderer.set_denoiser(None)
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.view_layer.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = "OPENIMAGEDENOISE"
        bpy.context.scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        bpy.context.scene.cycles.denoising_prefilter = "ACCURATE"
    else:
        bproc.renderer.set_denoiser(SceneConfig.denoiser)

    # Setup scene settingss
    bpy.context.scene.render.fps = SceneConfig.fps
    bproc.camera.set_resolution(*resolution)
    bproc.renderer.set_output_format(
        SceneConfig.file_format,
        SceneConfig.color_depth,
        SceneConfig.enable_transparency,
        SceneConfig.jpg_quality,
    )
    bproc.renderer.set_max_amount_of_samples(max_samples)
    bproc.renderer.set_noise_threshold(SceneConfig.sampling_noise_threshold)
    bpy.context.scene.cycles.tile_size = tile_size

    bpy.context.scene.cycles.use_fast_gi = SceneConfig.use_fast_gi
    bproc.renderer.set_light_bounces(
        SceneConfig.diffuse_bounces,
        SceneConfig.glossy_bounces,
        SceneConfig.ao_bounces_render,
        SceneConfig.max_bounces,
        SceneConfig.transmission_bounces,
        SceneConfig.transparency_bounces,
        SceneConfig.volume_bounces,
    )

    if motion_blur:
        print("Enabling motion blur")
        bproc.renderer.enable_motion_blur(motion_blur_length=0.5)

    print("\nLoading the background:", os.path.basename(background_path))
    bproc.world.set_world_background_hdr_img(background_path)

    # note: only objects in the obj_types can be loaded
    # otherwise, such as scene settings, they aren't loaded by blenderprc
    objs = bproc.loader.load_blend(
        scene_path,
        obj_types=["mesh", "light", "empty", "camera"],
        data_blocks=["objects", "materials"],
    )
    entities: list[Entity] = bproc.filter.all_with_type(objs, filtered_data_type=Entity)

    # collect UAV models by custom property
    uav_models: list[Entity] = bproc.filter.by_cp(entities, "UAV_model", True)
    uav_models.sort(key=methodcaller("get_name"))

    # organize entities for each uav model as dict
    uav_model_dict: dict[str, list[Entity]] = {}
    for model in uav_models:
        uav_name = model.get_name()
        uav_model_dict[uav_name] = [model] + sorted(
            model.get_children(return_all_offspring=True), key=methodcaller("get_name")
        )

    if models is not None:
        uav_model_dict = {name: uav_model_dict[name] for name in models}

    assert len(uav_model_dict) > 0, "UAV model is not found."
    print("\nFind", len(uav_model_dict), "UAV models")

    # hide all objects by default
    for model in entities:
        model.hide()
        model.blender_obj.hide_viewport = True

    return entities, uav_model_dict
