import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc

import bpy  # noqa: F401 # isort:skip

from blenderproc.python.types.EntityUtility import Entity
from blenderproc.python.types.MeshObjectUtility import MeshObject

from .frame import Frame
from .utils.camera import are_all_meshes_in_camera_view
from .utils.geometry import translate_axis
from .utils.utils import select_objects


def align_camera_pose(
    frame: int,
    entities: list[Entity],
    adaptive_alignment: bool = True,
    alignment_z_offset: float = 0,
    alignment_z_step: float = 0.1,
):
    camera = bpy.context.scene.camera

    # select the current model
    select_objects(entities)

    # including the animation
    with Frame(frame):
        # align the camera view to fit the UAV model
        # note: only the location will be modified.
        bpy.ops.view3d.camera_to_view_selected()

    # move the camera backward along local Z-axis (0, 0, 1)
    # by default, the camera view direction is local -Z axis in blender.
    # so, we just take the local Z-axis. (move backward)
    z_offset = alignment_z_offset
    translate_axis(camera, "Z", z_offset)

    # align the camera view adaptively to fit the vertices in frame-1, frame, frame+1
    if adaptive_alignment:
        # if motion_blur is enabled, we also need to check the previous and next frame
        frames = [frame]
        if bpy.context.scene.render.use_motion_blur:
            half_shutter = bpy.context.scene.render.motion_blur_shutter / 2
            frames += [(frame - 1, 1 - half_shutter), (frame, half_shutter)]

        # check if all vertices of UAV are in the camera
        uav_meshes: list[MeshObject] = bproc.filter.all_with_type(
            entities, filtered_data_type=MeshObject
        )
        while not are_all_meshes_in_camera_view(uav_meshes, frames):
            z_offset += alignment_z_step
            translate_axis(camera, "Z", alignment_z_step)
        print(f"[Adaptive alignment] Retrying to move backward... {z_offset:.3f}m")

    # set the camera pose
    matrix_world = bproc.camera.get_camera_pose()
    bproc.camera.add_camera_pose(matrix_world, frame=frame)
    return z_offset
