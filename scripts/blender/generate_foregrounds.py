import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc

import bpy  # noqa: F401 # isort:skip
import argparse
import os
import random
from math import radians, sqrt
from operator import itemgetter
from typing import Optional

import PIL.Image as Image
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.types.MeshObjectUtility import MeshObject
from mathutils import Euler, Vector
from rich import print

from uav_data_generation.blender import Checkpoint, COCOWriter
from uav_data_generation.blender.randomization import (
    align_camera_pose,
    group_and_filter_material_slots_by_cp,
    randomize_drone_properties,
)
from uav_data_generation.blender.utils import (
    bbox_overlaps,
    collect_images,
    collect_materials_by_cp,
    find_bbox_xyxy_by_alpha,
    get_cp,
    load_config,
    reset_keyframes,
    setup,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="An UAV generation script powered by blender",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config_path", type=str, help="Path to the config file")

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Path to where the final files, will be saved. By default, use the value from config.",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Resume generation with the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to the checkpoint. If --resume is True and it is None, use the latest checkpoint in --out-dir.",
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

    return args


def generate_uav_samples(
    uav_models: list[list[MeshObject]],
    materials: list[Material],
    x_range: tuple[int, int] = (-45, 45),
    y_range: tuple[int, int] = (-45, 45),
    z_range: tuple[int, int] = (0, 360),
    adaptive_alignment: bool = True,
    alignment_z_offset: float = 0,
    alignment_z_step: float = 0.1,
):
    # cache the original state of animtations
    original_action_keys = bpy.data.actions.keys()

    for uav_components in uav_models:
        uav_model = uav_components[0]

        # show components of the current uav model
        for uav_component in uav_components:
            visibility = get_cp(uav_component, "visibility", default=True)
            uav_component.hide(not visibility)

        # get material slots which require material_randomization
        material_slots_groups = group_and_filter_material_slots_by_cp(uav_components)

        # randomize the material and rotation
        frame = randomize_drone_properties(
            uav_model,
            material_slots_groups,
            materials,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
        )

        # align the camera with the UAV
        align_camera_pose(
            frame,
            uav_components,
            adaptive_alignment=adaptive_alignment,
            alignment_z_offset=alignment_z_offset,
            alignment_z_step=alignment_z_step,
        )

        # render the whole pipeline
        bproc.utility.set_keyframe_render_interval(frame_start=frame)
        image = bproc.renderer.render()["colors"][0]

        # hide current components for next rendering
        for uav_component in uav_components:
            uav_component.hide()

        # reset keyframes
        reset_keyframes(original_action_keys)

        yield image


def sample_uav_size(
    scale_range: tuple[float, float],
    image_size: tuple[int, int],
    uav_size: tuple[int, int],
    min_uav_area: int = 1,
    allow_upscaling: bool = False,
) -> tuple[int, int]:
    image_total_size = image_size[0] * image_size[1]
    uav_total_size = uav_size[0] * uav_size[1]

    for _ in range(50):
        # randomly sample a scale_ratio
        scale_ratio = random.uniform(*scale_range)
        scaled_image_total_size = image_total_size * scale_ratio

        # if not allow upscaling or smaller than min_uav_area, then skip it.
        if (
            scaled_image_total_size > uav_total_size and not allow_upscaling
        ) or scaled_image_total_size < min_uav_area:
            continue

        # calculate scaled width, height
        # w * r, h * r = (W, H)
        # w * h * r^2 ~= image_total_size * scale_ratio
        # r = sqrt(image_total_size * scale_ratio / uav_total_size)
        uav_scale_ratio = sqrt(scaled_image_total_size / uav_total_size)

        # round to the closet integer & round half to even (default rounding mode in IEEE 754)
        return round(uav_size[0] * uav_scale_ratio), round(
            uav_size[1] * uav_scale_ratio
        )

    raise RuntimeError(f"Cannot find an ideal scale fitting the range {scale_range}.")


def sample_uav_location(
    image_size: tuple[int, int],
    uav_size: tuple[int, int],
    bboxes: list[tuple[int, int, int, int]],
    max_iof: float = 0,
) -> Optional[tuple[int, int]]:
    end_w, end_h = image_size[0] - uav_size[0], image_size[1] - uav_size[1]

    if end_w < 0 or end_h < 0:
        print(
            f"Warning! Cannot find an ideal location fitting the UAV size {uav_size}."
        )
        return None

    for _ in range(50):
        # randomly sample a position from image based on the actual size
        x = random.randint(0, end_w)
        y = random.randint(0, end_h)

        # check if there is no overlap (or below the overlap threshold) among bboxes list
        bbox = [[x, y, *uav_size]]
        iofs1 = bbox_overlaps(bbox, bboxes, mode="iof")
        iofs2 = bbox_overlaps(bboxes, bbox, mode="iof")
        if (iofs1 <= max_iof).all() and (iofs2 <= max_iof).all():
            return x, y

    print(f"Warning! Cannot find an ideal location fitting the maximum IoF {max_iof}.")
    return None


def generate_foregrounds(
    scene_path: str,
    background_path: str,
    images_path: str,
    models: Optional[list[str]] = None,
    x_range: tuple[int, int] = (-45, 45),
    y_range: tuple[int, int] = (-45, 45),
    z_range: tuple[int, int] = (0, 360),
    max_samples: int = 20,
    max_iof: float = 0.5,
    scale_range: tuple[float, float] = (0.2, 0.8),
    min_uav_area: int = 1,
    allow_upscaling: bool = False,
    adaptive_alignment: bool = True,
    alignment_z_offset: float = 0,
    alignment_z_step: float = 0.1,
    motion_blur: bool = True,
    render_resolution: tuple[int, int] = (1920, 1920),
    render_max_samples: int = 1024,
    render_tile_size: int = 1024,
    out_dir: str = "outputs",
    device_type: str = "OPTIX",
    devices: list[int] = [0],
    checkpoint: Optional[Checkpoint] = None,
    max_checkpoints: int = 3,
    checkpoint_interval: int = 5,
):
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

    # prepare COCO writer
    if checkpoint is not None:
        coco_writer = checkpoint.coco_writer
        category_id = coco_writer.find_category("drone", "UAV")
    else:
        coco_writer = COCOWriter()
        category_id = coco_writer.add_category("drone", "UAV")

    checkpoints_dir = os.path.join(out_dir, "checkpoints")
    fg_images_dir = os.path.join(out_dir, "foregrounds")
    annotations_dir = os.path.join(out_dir, "annotations")

    # collect image informations
    image_paths = collect_images(images_path)
    if checkpoint is not None:
        assert (
            checkpoint.image_paths == image_paths
        ), "Inconsistent images. The checkpoint may be not for this."
        for image in coco_writer.images:
            image_path = os.path.join(fg_images_dir, image["file_name"])
            assert os.path.exists(
                image_path
            ), f"The foreground image {image_path} is not Found."

    uav_models = list(uav_models.values())
    start_index = checkpoint.image_index + 1 if checkpoint is not None else 0
    for image_index in range(start_index, len(image_paths)):
        image_path = image_paths[image_index]

        # determine how many samples should be generated
        num_samples = random.randint(1, max_samples)
        selected_models = random.choices(uav_models, k=num_samples)

        # create an foreground image with the same size as the image
        background_image = Image.open(image_path)
        foreground_image = Image.new("RGBA", background_image.size)
        background_image.close()

        uav_images: list[Image.Image] = []
        uav_bboxes: list[tuple[int, int, int, int]] = []
        image_size = foreground_image.size
        uav_generator = generate_uav_samples(
            selected_models,
            materials,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            adaptive_alignment=adaptive_alignment,
            alignment_z_offset=alignment_z_offset,
            alignment_z_step=alignment_z_step,
        )
        for uav_image in uav_generator:
            # utilize the alpha channel to find the bbox
            x1, y1, x2, y2 = find_bbox_xyxy_by_alpha(uav_image)

            # cut the image by bbox to get the actual size of UAV
            uav_image = uav_image[y1:y2, x1:x2]
            uav_image = Image.fromarray(uav_image, "RGBA")

            # retry
            # for _ in range(50):
            # determine the scaled size of UAV
            scaled_uav_size = sample_uav_size(
                scale_range,
                image_size,
                uav_image.size,
                min_uav_area=min_uav_area,
                allow_upscaling=allow_upscaling,
            )

            # determine the location of UAV on the foreground image
            uav_location = sample_uav_location(
                image_size, scaled_uav_size, uav_bboxes, max_iof=max_iof
            )
            # if uav_location is not None:
            #     break
            if uav_location is None:
                print("Skipping...", end="")
                continue
            # else:
            # after trying 50 times, stop generating. (no space)
            # print("Skipping...", end="")
            # continue

            # scale the UAV
            # filter comparison: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table
            uav_image = uav_image.resize(scaled_uav_size, Image.LANCZOS)

            uav_images.append(uav_image)
            uav_bboxes.append((*uav_location, *uav_image.size))

        print(f"Success: {len(uav_bboxes)} of {num_samples}.")

        ## Post-processing
        # paste UAVs starting from the most distant UAV
        # sort by uav_size
        indices = list(range(len(uav_bboxes)))
        indices.sort(key=lambda i: uav_bboxes[i][2] * uav_bboxes[i][3])
        for i in indices:
            uav_image, bbox = uav_images[i], uav_bboxes[i]
            foreground_image.alpha_composite(uav_image, dest=bbox[:2])

        # get output path
        rel_path = os.path.relpath(os.path.dirname(image_path), images_path)
        image_name, _ = os.path.splitext(os.path.basename(image_path))
        out_file = os.path.join(rel_path, f"{image_name}.png")

        # add image info to COCOWriter
        image_id = coco_writer.add_image(out_file, *foreground_image.size)
        coco_writer.add_annotations(image_id, category_id, uav_bboxes)

        # save the foreground image
        out_file = os.path.join(fg_images_dir, out_file)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        foreground_image.save(out_file)

        print(f"Saved the foreground image as {out_file}")

        # save the checkpoint every checkpoint_interval
        if (image_index + 1) % checkpoint_interval == 0:
            # ensure the COCO writer is consistent.
            assert image_index + 1 == len(coco_writer.images)
            checkpoint = Checkpoint(image_index, image_paths, coco_writer)
            checkpoint.save_pickle(
                os.path.join(checkpoints_dir, f"blender_state_{image_index + 1}.pkl")
            )

            # remove oldest checkpoints
            checkpoint_files = [
                os.path.join(checkpoints_dir, checkpoint_file)
                for checkpoint_file in os.listdir(checkpoints_dir)
            ]
            checkpoint_files = [
                (checkpoint_file, os.path.getmtime(checkpoint_file))
                for checkpoint_file in checkpoint_files
            ]
            checkpoint_files = sorted(checkpoint_files, key=itemgetter(1))
            for checkpoint_file, mtime in checkpoint_files[:-max_checkpoints]:
                os.remove(checkpoint_file)

    assert len(image_paths) == len(coco_writer.images)

    # save annotations in COCO format
    os.makedirs(annotations_dir, exist_ok=True)
    out_file = os.path.join(annotations_dir, "foreground.json")
    coco_writer.export(out_file)


if __name__ == "__main__":
    # checkpoint
    # 1. random state
    # 2. image paths (checking if it is consistent with argument & validating if the generated images exist)
    # 3. current progress of generated images (index)
    # 4. COCOWriter

    args = vars(parse_args())
    cfg = load_config(args.pop("config_path"))

    out_dir = args.pop("out_dir")
    if out_dir is not None:
        cfg.out_dir = out_dir

    print()
    print(cfg)

    ckpt = None
    if args.pop("resume"):
        ckpt_path = args.pop("checkpoint")
        if ckpt_path is None:
            ckpt_root_path = os.path.join(cfg.out_dir, "checkpoints")
            # use the latest checkpoint
            ckpt_path = sorted(os.listdir(ckpt_root_path))[-1]
            ckpt_path = os.path.join(ckpt_root_path, ckpt_path)

        ckpt = Checkpoint.from_pickle(ckpt_path)
        ckpt.restore_random_states()
        print(f"Resume from the last checkpoint, {ckpt_path}.")
    else:
        os.environ["BLENDER_PROC_RANDOM_SEED"] = str(cfg.seed)

    # TODO: refactor to use class
    generate_foregrounds(
        **vars(cfg),
        **args,
        checkpoint=ckpt,
    )
