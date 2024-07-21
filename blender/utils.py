import os
import random
from collections import defaultdict
from math import radians

import blenderproc as bproc
import bpy
import numpy as np
from blenderproc.api.types import MeshObject
from mathutils import Euler, Matrix, Vector

AXIS = {
    "X": (1, 0, 0),
    "Y": (0, 1, 0),
    "Z": (0, 0, 1),
    "-X": (-1, 0, 0),
    "-Y": (0, -1, 0),
    "-Z": (0, 0, -1),
}


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

    # Setup scene settingss
    bpy.context.scene.render.fps = 60
    bproc.camera.set_resolution(*resolution)
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.set_max_amount_of_samples(max_samples)
    bpy.context.scene.cycles.tile_size = tile_size

    if motion_blur:
        bproc.renderer.enable_motion_blur(motion_blur_length=0.5)

    # collect UAV models by custom property
    uav_objs: list[MeshObject] = bproc.filter.by_cp(
        objs, "UAV_model", True, filtered_data_type=MeshObject
    )

    # organize components for each uav model as dict
    uav_models: dict[str, list[MeshObject]] = defaultdict(list)
    for obj in uav_objs:
        uav_name = obj.get_name()
        uav_models[uav_name] += [obj] + obj.get_children(return_all_offspring=True)

    assert len(uav_models) > 0, "UAV model is not found."
    print("\nFind", len(uav_models), "UAV models")

    # hide all uav components and set categorid_id to 0 as drone category
    for uav_components in uav_models.values():
        for uav_component in uav_components:
            uav_component.hide()

    return objs, uav_models


def collect_materials_by_cp(
    cp_name: str = "random_material",
) -> list[bproc.types.Material]:
    materials = bproc.material.collect_all()
    materials = bproc.filter.by_cp(
        materials, cp_name, True, filtered_data_type=bproc.types.Material
    )
    print(f"Find {len(materials)} materials")
    return materials


def get_cp(object: MeshObject, key: str, default=None):
    if object.has_cp(key):
        return object.get_cp(key)
    else:
        return default


def reset_keyframes(original_action_keys: list[str] = []) -> None:
    """Removes registered keyframes from all objects which are not in original_action_keys and resets frame_start and frame_end"""
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0

    # clear all camera poses among keyframes
    tbd_action_keys = set(bpy.data.actions.keys()) - set(original_action_keys)
    for action_key in tbd_action_keys:
        action = bpy.data.actions[action_key]
        bpy.data.actions.remove(action)


def select_object(obj: MeshObject):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj.blender_obj
    obj.select()
    bpy.ops.object.select_hierarchy(direction="CHILD", extend=True)


def translate_axis(obj: bpy.types.Object, axis: str, amount: float):
    axis_vector = Vector(AXIS[axis.upper()])
    axis_vector *= amount

    # transform local vector to global vector by rotation matrix (ignore scale)
    translation_vector = obj.rotation_euler.to_matrix() @ axis_vector

    # translate location
    obj.location += translation_vector

    # update the location immediately
    bpy.context.view_layer.update()


def rand_rotation_euler(
    x_range: tuple[int, int] = (0, 0),
    y_range: tuple[int, int] = (0, 0),
    z_range: tuple[int, int] = (0, 0),
) -> Euler:
    def rand_angle(range: tuple[int, int]):
        angle = random.uniform(*range)
        return radians(angle)

    x_angle = rand_angle(x_range)
    y_angle = rand_angle(y_range)
    z_angle = rand_angle(z_range)

    # rotate each axis separately to avoid potentioal gimbal lock
    rot_matrix = Matrix.Identity(3)
    rot_matrix.rotate(Euler((x_angle, 0, 0)))
    rot_matrix.rotate(Euler((0, y_angle, 0)))
    rot_matrix.rotate(Euler((0, 0, z_angle)))
    return rot_matrix.to_euler()


def find_bbox_by_alpha(image: np.ndarray):
    assert image.shape[-1] == 4, "The color format should be in RGBA."
    y_indices, x_indices = image[..., -1].nonzero()
    min_x = np.amin(x_indices)
    max_x = np.amax(x_indices)
    min_y = np.amin(y_indices)
    max_y = np.amax(y_indices)
    return min_x, min_y, max_x, max_y
