import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc

import bpy
from blenderproc.api.types import Entity, Struct


def get_cp(obj: Struct, key: str, default=None):
    if obj.has_cp(key):
        return obj.get_cp(key)
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


def select_objects(objs: list[Entity]):
    assert len(objs) > 0

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = objs[0].blender_obj
    for obj in objs:
        obj.select()
