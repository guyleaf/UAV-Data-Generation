import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc


import bpy  # noqa: F401 # isort:skip
import os
import random
import sys
from collections import defaultdict

import idprop
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.utility.Utility import KeyFrame

sys.path.append(os.path.dirname(__file__))

from utils import (
    are_all_meshes_in_camera_view,
    get_cp,
    rand_rotation_euler,
    select_objects,
    translate_axis,
)


def group_and_filter_material_slots_by_cp(
    meshes: list[MeshObject],
    group_cp_name: str = "group_name",
    filter_cp_name: str = "material_randomization",
) -> dict[str, list[tuple[MeshObject, int]]]:
    def get_cp_(obj: MeshObject, cp_name: str, default):
        cp_values = get_cp(obj, cp_name, default=default)
        if isinstance(cp_values, idprop.types.IDPropertyArray):
            cp_values = cp_values.to_list()

        # check the length of cp_values == num_slots
        num_slots = max(len(obj.blender_obj.material_slots), 1)
        if isinstance(cp_values, list):
            assert len(cp_values) == num_slots
        else:
            cp_values = [cp_values] * num_slots
        return cp_values

    groups = defaultdict(list)
    for mesh in meshes:
        name = mesh.get_name()
        num_slots = max(len(mesh.blender_obj.material_slots), 1)

        # grouping by group_cp_name
        # 1. if the cp is not defined, make every slot as an individual group
        # 2. if the cp_value is a single value, broadcast to every slot
        # 3. if the cp_value is a list, use it directly.
        default_value = [f"{name}_{i}" for i in range(num_slots)]
        group_cp_values: list[str] = get_cp_(mesh, group_cp_name, default=default_value)

        # filtering by filter_cp_name
        # 1. if the cp is not defined, make every slot require randomization
        # 2. if the cp_value is a single value, broadcast to every slot
        # 3. if the cp_value is a list, use it directly.
        default_value = [True] * num_slots
        filter_cp_values: list[bool] = get_cp_(
            mesh, filter_cp_name, default=default_value
        )

        for i, group_cp in filter(
            lambda item: filter_cp_values[item[0]], enumerate(group_cp_values)
        ):
            groups[group_cp].append((mesh, i))
    return groups


def randomize_drone_properties(
    uav_model: MeshObject,
    material_slots_groups: dict[str, list[tuple[MeshObject, int]]],
    materials: list[Material],
    x_range: tuple[int, int] = (-45, 45),
    y_range: tuple[int, int] = (-45, 45),
    z_range: tuple[int, int] = (0, 360),
) -> int:
    # randomly sample a frame for animation
    frame = random.randint(0, 249)

    # randomly sample an euler angle
    euler = rand_rotation_euler(x_range, y_range, z_range)
    uav_model.set_rotation_euler(euler, frame=frame)

    # randomly apply a material for each group
    for material_slots in material_slots_groups.values():
        material = random.choice(materials)
        for mesh, i in material_slots:
            if mesh.has_materials():
                mesh.set_material(i, material)
            else:
                assert (
                    i == 0
                ), "The index of material slot must be 0 because there is no material slots in object."
                mesh.add_material(material)

    return frame


def align_camera_pose(
    frame: int,
    uav_components: list[MeshObject],
    adaptive_alignment: bool = True,
    alignment_z_offset: float = 0,
    alignment_z_step: float = 0.1,
    motion_blur: bool = True,
):
    # select current UAV model
    select_objects(uav_components)

    # including the animation
    with KeyFrame(frame):
        # align the camera view to fit the UAV model
        # note: only the location will be modified.
        bpy.ops.view3d.camera_to_view_selected()

    # move the camera backward along local Z-axis (0, 0, 1)
    # by default, the camera view direction is local -Z axis in blender.
    # so, we just take the local Z-axis. (move backward)
    camera = bpy.context.scene.camera
    z_offset = alignment_z_offset
    translate_axis(camera, "Z", z_offset)

    # align the camera view adaptively to fit the vertices in frame-1, frame, frame+1
    if adaptive_alignment:
        # if motion_blur is enabled, we also need to check the previous and next frame
        frames = [frame]
        if motion_blur:
            frames += [frame - 1, frame + 1]

        # check if all vertices of UAV are in the camera
        while not are_all_meshes_in_camera_view(uav_components, frames):
            z_offset += alignment_z_step
            print(f"[Adaptive alignment] Retrying to move backward... {z_offset:.3f}m")
            translate_axis(camera, "Z", alignment_z_step)

    # set the camera pose
    bproc.camera.add_camera_pose(camera.matrix_world, frame=frame)
    return z_offset
