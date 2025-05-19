import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc

import os
from operator import methodcaller

import bpy
from blenderproc.api.types import Entity

from .config import BaseConfig


def setup(config: BaseConfig, device_type: str, devices: list[int]):
    bproc.init()

    # set render device
    use_only_cpu = device_type == "CPU"
    device_type = device_type if not use_only_cpu else None
    bproc.renderer.set_render_devices(
        use_only_cpu=use_only_cpu,
        desired_gpu_device_type=device_type if not use_only_cpu else None,
        desired_gpu_ids=devices,
    )

    if config.render_denoiser == "OPENIMAGEDENOISE":
        # WORKAROUND: currently, the blenderproc api doesn't support enabling OpenImageDenoise
        bproc.renderer.set_denoiser(None)
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.view_layer.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = "OPENIMAGEDENOISE"
        bpy.context.scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        bpy.context.scene.cycles.denoising_prefilter = "ACCURATE"
    else:
        bproc.renderer.set_denoiser(config.render_denoiser)

    # setup scene settings
    bpy.context.scene.render.fps = config.fps
    bproc.camera.set_resolution(*config.render_resolution)
    bproc.renderer.set_output_format(
        config.file_format,
        config.color_depth,
        config.enable_transparency,
        config.jpg_quality,
    )
    bproc.renderer.set_max_amount_of_samples(config.render_max_samples)
    bproc.renderer.set_noise_threshold(config.render_sampling_noise_threshold)
    bpy.context.scene.cycles.tile_size = config.render_tile_size
    bpy.context.scene.render.use_simplify = True
    bpy.context.scene.render.simplify_subdivision_render = (
        config.simplify_subdivision_render
    )

    # light paths settings
    if config.use_light_preset:
        print("Loading the light preset:", config.light_preset_path)
        bpy.utils.execfile(config.light_preset_path)
    else:
        bpy.context.scene.cycles.caustics_reflective = config.caustics_reflective
        bpy.context.scene.cycles.caustics_refractive = config.caustics_refractive
        bpy.context.scene.cycles.use_fast_gi = config.use_fast_gi
        bpy.context.scene.cycles.ao_bounces = config.ao_bounces
        bproc.renderer.set_light_bounces(
            config.diffuse_bounces,
            config.glossy_bounces,
            config.ao_bounces_render,
            config.max_bounces,
            config.transmission_bounces,
            config.transparent_max_bounces,
            config.volume_bounces,
        )

    if config.motion_blur:
        print("Enabling motion blur")
        bproc.renderer.enable_motion_blur(motion_blur_length=0.5)

    print("\nLoading the background:", os.path.basename(config.background_path))
    bproc.world.set_world_background_hdr_img(config.background_path)

    # note: only objects in the obj_types can be loaded
    # otherwise, such as scene settings, they aren't loaded by blenderprc
    objs = bproc.loader.load_blend(
        config.scene_path,
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

    if config.models is not None:
        uav_model_dict = {name: uav_model_dict[name] for name in config.models}

    assert len(uav_model_dict) > 0, "UAV model is not found."
    print("\nFind", len(uav_model_dict), "UAV models")

    # hide all objects by default
    for model in entities:
        model.hide()
        model.blender_obj.hide_viewport = True

    return entities, uav_model_dict
