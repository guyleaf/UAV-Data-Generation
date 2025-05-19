import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc


import bpy
from blenderproc.api.types import Entity, Struct


def get_cp(obj: Struct, key: str, default=None):
    if obj.has_cp(key):
        return obj.get_cp(key)
    else:
        return default


def get_blender_preset(preset_subdir: str, name: str) -> str:
    """Get the preset in the Blender

    Args:
        preset_subdir (str): the subdir relative to the blender/blender-x.x.x-linux-x64/x.x/scripts/presets.
        name (str): The name of the preset

    Returns:
        str: the path of the preset
    """
    # blender_install_dir = os.path.dirname(bpy.app.binary_path)
    # blender_version = ".".join(bpy.app.version[:2])
    # assert len(blender_install_dir) > 0

    # presets_dir = os.path.join(
    #     blender_install_dir, blender_version, "scripts", "presets"
    # )
    # preset_path = os.path.join(presets_dir, path)
    # assert os.path.isfile(preset_path)
    preset_path = bpy.utils.preset_find(name, preset_subdir)
    if not preset_path:
        preset_path = bpy.utils.preset_find(name, preset_subdir, display_name=True)
    assert preset_path is not None, f"The preset {preset_subdir}, {name} is not found."
    return preset_path


def reset_keyframes(original_action_keys: list[str] = []) -> None:
    """
    Removes registered keyframes from all objects which are not in original_action_keys and resets frame_start and frame_end.

    Refers to `bproc.api.utility.reset_keyframes()`.
    By default, the bproc version will remove all animations (including those in .blend file).

    So, I reimplement it to exclude the original actions.

    Args:
        original_action_keys (list[str], optional): Original actions. You can get it from `bpy.data.actions.keys()`. Defaults to [].
    """
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0

    # clear all camera poses and animations among keyframes
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
