from typing import Union

import bpy
import numpy as np
from blenderproc.api.types import MeshObject
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from ..frame import Frame


def is_vertex_in_camera_view(vertex: Union[np.ndarray, Vector]) -> bool:
    if isinstance(vertex, np.ndarray):
        vertex = Vector(vertex)

    camera = bpy.context.scene.camera
    vertex_in_camera_view = world_to_camera_view(bpy.context.scene, camera, vertex)
    return (
        0 <= vertex_in_camera_view.x <= 1
        and 0 <= vertex_in_camera_view.y <= 1
        and camera.data.clip_start <= vertex_in_camera_view.z <= camera.data.clip_end
    )


def is_vertices_in_camera_view(vertices: Union[np.ndarray, list[Vector]]) -> bool:
    return all(map(is_vertex_in_camera_view, vertices))


def are_all_meshes_in_camera_view(
    meshes: list[MeshObject], frames: list[Union[int, tuple[int, float]]]
):
    for frame in frames:
        if isinstance(frame, tuple):
            frame, subframe = frame
        else:
            frame, subframe = frame, 0

        with Frame(frame, subframe=subframe):
            for mesh in meshes:
                if not is_vertices_in_camera_view(mesh.get_bound_box()):
                    return False
    return True
