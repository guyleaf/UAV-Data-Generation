import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc

import warnings

import bpy  # noqa: F401 # isort:skip
import argparse
import os
import random
import sys
from collections import defaultdict
from math import radians, sqrt
from typing import Optional

import idprop
import PIL.Image as Image
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.types.MeshObjectUtility import MeshObject
from mathutils import Euler, Vector

sys.path.append(os.path.dirname(__file__))

from coco import COCOWriter
from utils import (
    bbox_overlaps,
    collect_images,
    collect_materials_by_cp,
    find_bbox_xyxy_by_alpha,
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
        default=(0.2, 0.5),
        help="The scale range relative to the size of image.",
    )
    parser.add_argument(
        "--max-iou",
        type=float,
        default=0.2,
        help="The maximum IoU among UAVs in image. (max_iou > 0 -> accept occlusion)",
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
        0 < args.scale_range[0] <= args.scale_range[1] <= 1
    ), "The scale range should be in (0, 1]."
    assert 0 <= args.max_iou <= 1, "The maximum IoU should be in (0, 1)."
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
        image = bproc.renderer.render()["colors"][0]

        # hide current components for next rendering
        for uav_component in uav_components:
            uav_component.hide()

        # reset keyframes
        reset_keyframes(original_action_keys)

        yield image


def get_scaled_uav_size(
    scale_range: tuple[float, float],
    image_size: tuple[int, int],
    uav_size: tuple[int, int],
) -> tuple[int, int]:
    image_total_size = image_size[0] * image_size[1]
    uav_total_size = uav_size[0] * uav_size[1]

    # randomly sample a scale_ratio
    for _ in range(50):
        scale_ratio = random.uniform(*scale_range)

        # avoid upscaling becuse we don't constrain the original size
        if (image_total_size * scale_ratio) <= uav_total_size:
            # calculate scaled width, height
            # w * r, h * r = (W, H)
            # w * h * r^2 ~= image_total_size * scale_ratio
            # r = sqrt(image_total_size * scale_ratio / uav_total_size)
            uav_scale_ratio = sqrt(image_total_size * scale_ratio / uav_total_size)

            # round to the closet integer & round half to even (default rounding mode in IEEE 754)
            return round(uav_size[0] * uav_scale_ratio), round(
                uav_size[1] * uav_scale_ratio
            )

    raise RuntimeError(f"Cannot find an ideal scale fitting the range {scale_range}.")


def scale_uav(
    uav_image: Image.Image,
    scale_range: tuple[float, float],
    image_size: tuple[int, int],
) -> Image.Image:
    # determine the scaled size of UAV object
    scaled_uav_size = get_scaled_uav_size(scale_range, image_size, uav_image.size)

    # scale the UAV
    # Filter comparison: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table
    uav_image = uav_image.resize(scaled_uav_size, Image.LANCZOS)
    return uav_image


def get_uav_location(
    image_size: tuple[int, int],
    uav_size: tuple[int, int],
    bboxes: list[tuple[int, int, int, int]],
    max_iou: float = 0,
) -> Optional[tuple[int, int]]:
    end_w, end_h = image_size[0] - uav_size[0], image_size[1] - uav_size[1]

    for _ in range(50):
        # randomly sample a position from image based on the actual size
        x = random.randint(0, max(end_w, 0))
        y = random.randint(0, max(end_h, 0))

        # check if there is no overlap (or below the overlap threshold) among bboxes list
        ious = bbox_overlaps([[x, y, *uav_size]], bboxes)
        if (ious <= max_iou).all():
            return x, y

    warnings.warn(f"Cannot find an ideal location fitting the maximum IoU {max_iou}.")
    return None


def main(
    scene_path: str,
    background_path: str,
    images_path: str,
    models: Optional[list[str]] = None,
    x_range: tuple[int, int] = (-45, 45),
    y_range: tuple[int, int] = (-45, 45),
    z_range: tuple[int, int] = (0, 360),
    max_samples: int = 20,
    scale_range: tuple[float, float] = (0.2, 0.8),
    max_iou: float = 0.5,
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

    images_dir = os.path.join(out_dir, "images")
    annotations_dir = os.path.join(out_dir, "annotations")
    coco_writer = COCOWriter()
    for image_path in image_paths:
        # determine how many samples should be generated
        num_samples = random.randint(1, max_samples)
        selected_models = random.choices(list(uav_models.values()), k=num_samples)

        # create an foreground image with the same size as the image
        background_image = Image.open(image_path)
        foreground_image = Image.new("RGBA", background_image.size)
        background_image.close()

        uav_images: list[Image.Image] = []
        uav_bboxes: list[tuple[int, int, int, int]] = []
        image_size = foreground_image.size
        for rendered_image in generate_uav_samples(
            selected_models, materials, x_range, y_range, z_range
        ):
            # utilize the alpha channel to find the bbox
            x1, y1, x2, y2 = find_bbox_xyxy_by_alpha(rendered_image)

            # cut the image by bbox to get the actual size of UAV object
            uav_image = rendered_image[y1:y2, x1:x2]
            uav_image = Image.fromarray(uav_image, "RGBA")

            # scale UAV
            uav_image = scale_uav(uav_image, scale_range, image_size)

            # determine the location of UAV on the foreground image
            uav_location = get_uav_location(
                image_size, uav_image.size, uav_bboxes, max_iou=max_iou
            )
            # if it returns None, stop generating. (no space)
            if uav_location is None:
                print(f"Total UAVs: {len(uav_bboxes)}. Skipping...")
                break

            uav_images.append(uav_image)
            uav_bboxes.append((*uav_location, *uav_image.size))

        # paste UAVs starting from the most distant UAV
        # sort by uav_size
        indices = list(range(len(uav_bboxes)))
        indices.sort(key=lambda i: uav_bboxes[i][2] * uav_bboxes[i][3])
        for i in indices:
            uav_image, bbox = uav_images[i], uav_bboxes[i]
            foreground_image.paste(uav_image, box=bbox[:2], mask=uav_image)

        # get output path
        rel_path = os.path.relpath(os.path.dirname(image_path), images_path)
        image_name, _ = os.path.splitext(os.path.basename(image_path))
        out_file = os.path.join(rel_path, f"{image_name}.png")

        # add image info to COCOWriter
        image_id = coco_writer.add_image(out_file, *foreground_image.size)
        coco_writer.add_annotations(image_id, 1, uav_bboxes)

        # save the foreground image
        out_file = os.path.join(images_dir, out_file)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        foreground_image.save(out_file)

    # save annotations in COCO format
    os.makedirs(annotations_dir, exist_ok=True)
    out_file = os.path.join(annotations_dir, "all.json")
    coco_writer.export(out_file)


if __name__ == "__main__":
    args = vars(parse_args())
    os.environ["BLENDER_PROC_RANDOM_SEED"] = str(args.pop("seed"))
    # TODO: refactor to use class
    main(
        args.pop("scene_path"),
        args.pop("background_path"),
        args.pop("images_path"),
        **args,
    )
