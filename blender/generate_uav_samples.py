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
import tqdm
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.types.MeshObjectUtility import MeshObject
from mathutils import Euler, Vector

sys.path.append(os.path.dirname(__file__))

from utils import (
    collect_images,
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
        description="An UAV generation script powered by blender",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scene_path", type=str, help="Path to the .blend scene file")
    parser.add_argument(
        "background_path",
        type=str,
        help="Path to the background HDRI file",
    )
    parser.add_argument(
        "images_path",
        type=str,
        help="Path to the folder of image files",
    )

    # randomization settings
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
        "--max-samples",
        default=20,
        type=int,
        help="The maximum number of UAVs per image.",
    )
    parser.add_argument(
        "--scale-range",
        nargs=2,
        type=float,
        default=(0.2, 0.8),
        help="The size range of generated UAVs relative to the size of image.",
    )

    # blender's render settings
    parser.add_argument(
        "--motion-blur",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable motion blur or not",
    )
    parser.add_argument(
        "--render-resolution",
        nargs=2,
        type=int,
        default=(1920, 1920),
        help="The original resolution of rendered UAV samples",
    )
    parser.add_argument(
        "--render-max-samples",
        type=int,
        default=1024,
        help="The maximum number of samples to render for each pixel",
    )
    parser.add_argument(
        "--render-tile-size",
        type=int,
        default=1024,
        help="The tile size for rendering a image (less -> slower & lower VRAM requirement, higher -> faster & higher VRAM requirement)",
    )

    # misc settings
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Path to where the final files, will be saved",
    )
    parser.add_argument("--models", type=str, nargs="+", default=None)
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
    assert args.scene_path.endswith(".blend"), "The scene file should be a .blend file."
    assert os.path.isdir(args.images_path), "The images_path should be a folder path."

    for range_ in [args.x_range, args.y_range, args.z_range]:
        assert (
            range_[0] <= range_[1]
        ), f"The left of range should be less than or equal to the right, {range_}."

    assert (
        0 < args.max_samples
    ), "The maximum number of samples should be greater than 0."
    assert (
        0 < args.scale_range[0] <= args.scale_range[1] < 1
    ), "The scale range should be between 0 and 1."
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


def adjust_camera_pose(frame: int, backward_amount: float = 0.3):
    camera = bpy.context.scene.camera
    # align the camera view to fit the UAV model
    bpy.ops.view3d.camera_to_view_selected()

    # move the camera backward about 0.3m along local Z-axis (0, 0, 1)
    # by default, the camera view direction is local -Z axis in blender.
    # so, we just take the local Z-axis. (move backward)
    translate_axis(camera, "Z", backward_amount)

    # set the camera pose
    bproc.camera.add_camera_pose(camera.matrix_world, frame=frame)


def generate_uav_samples(
    uav_models: list[list[MeshObject]],
    materials: list[Material],
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    z_range: tuple[int, int],
):
    # cache the original state of animtations
    original_action_keys = bpy.data.actions.keys()

    for uav_components in uav_models:
        uav_model = uav_components[0]

        # show components of the current uav model
        for uav_component in uav_components:
            visibility = get_cp(uav_component, "visibility", default=True)
            uav_component.hide(not visibility)

        # select current UAV model
        select_object(uav_model)

        # get material slots which require material_randomization
        material_slots_groups = group_and_filter_material_slots_by_cp(uav_components)

        frame = randomize_drone_properties(
            uav_model, material_slots_groups, materials, x_range, y_range, z_range
        )
        adjust_camera_pose(frame, backward_amount=0.3)

        # render the whole pipeline
        bproc.utility.set_keyframe_render_interval(frame_start=frame)
        yield bproc.renderer.render()["colors"][0]

        # hide current components for next rendering
        for uav_component in uav_components:
            uav_component.hide()

        # reset keyframes
        reset_keyframes(original_action_keys)


def main(
    scene_path: str,
    background_path: str,
    images_path: str,
    models: Optional[list[str]] = None,
    max_samples: int = 20,
    x_range: tuple[int, int] = (-45, 45),
    y_range: tuple[int, int] = (-45, 45),
    z_range: tuple[int, int] = (0, 360),
    motion_blur: bool = True,
    render_resolution: tuple[int, int] = (1920, 1920),
    render_max_samples: int = 1024,
    render_tile_size: int = 1024,
    out_dir: str = "outputs",
    device_type: str = "OPTIX",
    devices: list[int] = [0],
):
    image_paths = collect_images(images_path)
    objs, uav_models = setup(
        scene_path,
        background_path,
        device_type,
        devices,
        motion_blur=motion_blur,
        resolution=render_resolution,
        max_samples=render_max_samples,
        tile_size=render_tile_size,
    )
    materials = collect_materials_by_cp()

    if models is not None:
        uav_models = {name: uav_models[name] for name in models}

    # place the camera in front of the UAV model
    camera = bpy.context.scene.camera
    camera.location = Vector((0, -1, 0))
    camera.rotation_euler = Euler((radians(90), 0, 0))
    bpy.context.view_layer.update()

    for image_path in tqdm.tqdm(image_paths, desc="Generating..."):
        # determine how many samples should be generated
        num_samples = random.randint(1, max_samples)
        selected_models = random.choices(list(uav_models.values()), k=num_samples)

        # create an foreground image with the same size as the image by Pillow

        bboxes = []
        for rendered_image in generate_uav_samples(
            selected_models, materials, x_range, y_range, z_range
        ):
            # utilize the alpha channel to find the bbox
            x1, y1, x2, y2 = find_bbox_by_alpha(rendered_image)

            # cut the image by bbox to get the actual size of UAV object
            rendered_image = rendered_image[y1:y2, x1:x2]

            # randomly sample the actual size of the object by scale_ratio
            # check if the original size is greater than the actual size
            # if not, retry it.

            # scale the object to it

            # randomly sample a position from image based on the actual size

            # check if there is no overlap (or below the overlap threshold) among bboxes list

            # use Pillow to paste the object on the foreground image

            # record the bbox
            # bboxes.append(bbox)

        # save the foreground image


if __name__ == "__main__":
    args = parse_args()
    os.environ["BLENDER_PROC_RANDOM_SEED"] = str(args.seed)
    main(
        args.scene_path,
        args.background_path,
        args.images_path,
        models=args.models,
        max_samples=args.max_samples,
        x_range=args.x_range,
        y_range=args.y_range,
        z_range=args.z_range,
        motion_blur=args.motion_blur,
        render_resolution=args.render_resolution,
        render_max_samples=args.render_max_samples,
        render_tile_size=args.render_tile_size,
        out_dir=args.out_dir,
        device_type=args.device_type,
        devices=args.devices,
    )
