import random
from math import radians
from typing import Union

import bpy
from blenderproc.api.types import Entity
from mathutils import Euler, Matrix, Vector

AXIS = {
    "X": (1, 0, 0),
    "Y": (0, 1, 0),
    "Z": (0, 0, 1),
    "-X": (-1, 0, 0),
    "-Y": (0, -1, 0),
    "-Z": (0, 0, -1),
}


def translate_axis(obj: Union[Entity, bpy.types.Object], axis: str, amount: float):
    if isinstance(obj, Entity):
        obj = obj.blender_obj

    # calculate local vector for translation
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
