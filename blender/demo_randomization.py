import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc
import bpy  # noqa: F401 # isort:skip
import argparse
import os
import random
import sys
from collections import defaultdict
from math import radians
from typing import Optional

import idprop
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.types.MeshObjectUtility import MeshObject
from mathutils import Euler, Vector
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(__file__))

from utils import (
    collect_materials_by_cp,
    find_bbox_by_alpha,
    get_cp,
    rand_rotation_euler,
    reset_keyframes,
    select_object,
    setup,
    translate_axis,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("scene_path", type=str, help="Path to the .blend scene file")
    parser.add_argument(
        "background_path",
        type=str,
        help="Path to the background HDRI file",
    )
    parser.add_argument(
        "--x-range",
        type=int,
        nargs=2,
        default=(-45, 45),
        help="The angle range of x-axis in degree.",
    )
    parser.add_argument(
        "--y-range",
        type=int,
        nargs=2,
        default=(-45, 45),
        help="The angle range of y-axis in degree.",
    )
    parser.add_argument(
        "--z-range",
        type=int,
        nargs=2,
        default=(0, 360),
        help="The angle range of y-axis in degree.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Path to where the final files, will be saved",
    )
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument(
        "--samples",
        default=60,
        type=int,
        help="The number of times the objects should be animated and rendered.",
    )
    parser.add_argument(
        "--seed",
        default=2024,
        type=int,
        help="The seed for random sampling.",
    )
    parser.add_argument(
        "--device-type",
        default="OPTIX",
        type=str,
        help="The GPU device type for rendering. Possible choices are [CPU, OPTIX, CUDA, METAL, HIP]",
    )
    parser.add_argument(
        "--devices",
        default=[0],
        type=int,
        nargs="+",
        help="The GPU device ids for rendering. You can check the id by executing list_gpu_devices.py",
    )

    args = parser.parse_args()
    assert args.scene_path.endswith(".blend") and os.path.isfile(args.scene_path)
    return args


def group_and_filter_material_slots_by_cp(
    objs: list[MeshObject],
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
    for obj in objs:
        name = obj.get_name()
        num_slots = max(len(obj.blender_obj.material_slots), 1)

        # grouping by group_cp_name
        # 1. if the cp is not defined, make every slot as an individual group
        # 2. if the cp_value is a single value, broadcast to every slot
        # 3. if the cp_value is a list, use it directly.
        default_value = [f"{name}_{i}" for i in range(num_slots)]
        group_cp_values: list[str] = get_cp_(obj, group_cp_name, default=default_value)

        # filtering by filter_cp_name
        # 1. if the cp is not defined, make every slot require randomization
        # 2. if the cp_value is a single value, broadcast to every slot
        # 3. if the cp_value is a list, use it directly.
        default_value = [True] * num_slots
        filter_cp_values: list[bool] = get_cp_(
            obj, filter_cp_name, default=default_value
        )

        for i, group_cp in filter(
            lambda item: filter_cp_values[item[0]], enumerate(group_cp_values)
        ):
            groups[group_cp].append((obj, i))
    return groups


def randomize_drone_properties(
    uav_model: MeshObject,
    material_slots_groups: dict[str, list[tuple[MeshObject, int]]],
    materials: list[Material],
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    z_range: tuple[int, int],
):
    # randomly sample a frame for animation
    frame = random.randint(0, 249)

    # set the current frame in order to get the correct animtation to let the camera fit the object
    bpy.context.scene.frame_set(frame)

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


def set_camera_pose(frame: int, backward_amount: float = 0.3):
    camera = bpy.context.scene.camera
    # align the camera view to fit the UAV model
    bpy.ops.view3d.camera_to_view_selected()

    # move the camera backward about 0.3m along local Z-axis (0, 0, 1)
    # by default, the camera view direction is local -Z axis in blender.
    # so, we just take the local Z-axis. (move backward)
    translate_axis(camera, "Z", backward_amount)

    # set the camera pose
    bproc.camera.add_camera_pose(camera.matrix_world, frame=frame)


def draw_bounding_box(image: Image.Image, coord: tuple[int, int, int, int]):
    size = (coord[2] - coord[0]) * (coord[3] - coord[1])

    draw = ImageDraw.Draw(image)
    draw.rectangle(coord, outline="red")

    text = f"Object size: {size:,}"
    font_file = font_manager.findfont("arial")
    font = ImageFont.truetype(font_file, 36)
    text_xy = list(coord[:2])
    text_coord = text_xy + list(draw.textbbox(text_xy, text=text, font=font)[2:])

    draw.rectangle(text_coord, fill="red")
    draw.text(text_coord[:2], text=text, font=font, fill="white")


def main(
    scene_path: str,
    background_path: str,
    models: Optional[list[str]] = None,
    out_dir: str = "outputs",
    samples: int = 3,
    device_type: str = "OPTIX",
    devices: list[int] = [0],
    x_range: tuple[int, int] = (-45, 45),
    y_range: tuple[int, int] = (-45, 45),
    z_range: tuple[int, int] = (0, 360),
):
    objs, uav_models = setup(
        scene_path, background_path, device_type, devices, motion_blur=True
    )
    materials = collect_materials_by_cp()

    if models is not None:
        uav_models = {name: uav_models[name] for name in models}

    # place the camera in front of the UAV model
    camera = bpy.context.scene.camera
    camera.location = Vector((0, -1, 0))
    camera.rotation_euler = Euler((radians(90), 0, 0))
    bpy.context.view_layer.update()

    original_action_keys = bpy.data.actions.keys()
    for i, (name, uav_components) in enumerate(uav_models.items()):
        uav_model = uav_components[0]
        print("\nUAV name:", name)

        # show components of the current uav model
        for uav_component in uav_components:
            visibility = get_cp(uav_component, "visibility", default=True)
            uav_component.hide(not visibility)

        # select current UAV model
        select_object(uav_model)

        # get material slots which require material_randomization
        material_slots_groups = group_and_filter_material_slots_by_cp(uav_components)

        for i in range(samples):
            frame = randomize_drone_properties(
                uav_model, material_slots_groups, materials, x_range, y_range, z_range
            )
            set_camera_pose(frame, backward_amount=0.3)

            # render the whole pipeline
            bproc.utility.set_keyframe_render_interval(frame_start=frame)
            data = bproc.renderer.render()

            out_path = os.path.join(out_dir, name)
            os.makedirs(out_path, exist_ok=True)

            # write the color to a .png container in the run-specific output directory
            color = data["colors"][0]

            # find the bounding box
            bbox = find_bbox_by_alpha(color)

            image = Image.fromarray(color, mode="RGBA")
            draw_bounding_box(image, bbox)
            image.save(os.path.join(out_path, f"{i}_{frame}.png"))

            # write the data to a .hdf5 container in the run-specific output directory
            # bproc.writer.write_hdf5(out_path, data)

            # reset keyframes
            reset_keyframes(original_action_keys)

        # hide current components for next rendering
        for uav_component in uav_components:
            uav_component.hide()


if __name__ == "__main__":
    args = parse_args()
    os.environ["BLENDER_PROC_RANDOM_SEED"] = str(args.seed)
    main(
        args.scene_path,
        args.background_path,
        models=args.models,
        out_dir=args.out_dir,
        samples=args.samples,
        device_type=args.device_type,
        devices=args.devices,
        x_range=args.x_range,
        y_range=args.y_range,
        z_range=args.z_range,
    )
