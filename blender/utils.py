import glob
import os
import random
from collections import defaultdict
from math import radians
from mimetypes import MimeTypes
from typing import Union

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
    cp_name: str = "random_material", cp_value: bool = True
) -> list[bproc.types.Material]:
    materials = bproc.material.collect_all()
    materials = bproc.filter.by_cp(
        materials, cp_name, cp_value, filtered_data_type=bproc.types.Material
    )
    print(f"Find {len(materials)} materials")
    return materials


def collect_images(root_path: str):
    mime_checker = MimeTypes()

    def validate_file_type(path: str):
        mime_type = mime_checker.guess_type(path)[0]
        return mime_type is not None and "image" in mime_type

    return map(
        lambda path: os.path.join(root_path, path),
        filter(
            validate_file_type,
            glob.iglob("**/*.*", root_dir=root_path, recursive=True),
        ),
    )


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


def find_bbox_xyxy_by_alpha(image: np.ndarray):
    assert image.shape[-1] == 4, "The color format should be in RGBA."
    y_indices, x_indices = image[..., -1].nonzero()
    min_x = np.amin(x_indices)
    max_x = np.amax(x_indices) + 1
    min_y = np.amin(y_indices)
    max_y = np.amax(y_indices) + 1
    return min_x, min_y, max_x, max_y


def bboxes_xywh_to_xyxy(bboxes: np.ndarray):
    bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
    return bboxes


# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/evaluation/functional/bbox_overlaps.py
def bbox_overlaps(
    bboxes1: Union[np.ndarray, list[tuple[int, int, int, int]]],
    bboxes2: Union[np.ndarray, list[tuple[int, int, int, int]]],
    mode: str = "iou",
    eps: float = 1e-6,
):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (Union[np.ndarray, list[tuple[int, int, int, int]]]): Shape (n, 4), Format (x, y, w, h)
        bboxes2 (Union[np.ndarray, list[tuple[int, int, int, int]]]): Shape (k, 4), Format (x, y, w, h)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)

    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ["iou", "iof"]
    if not isinstance(bboxes1, np.ndarray):
        bboxes1 = np.array(bboxes1)
    if not isinstance(bboxes2, np.ndarray):
        bboxes2 = np.array(bboxes2)

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious

    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True

    area1 = bboxes1[:, 2] * bboxes1[:, 3]
    area2 = bboxes2[:, 2] * bboxes2[:, 3]
    bboxes1 = bboxes_xywh_to_xyxy(bboxes1)
    bboxes2 = bboxes_xywh_to_xyxy(bboxes2)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(y_end - y_start, 0)
        if mode == "iou":
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union

    if exchange:
        ious = ious.T
    return ious
