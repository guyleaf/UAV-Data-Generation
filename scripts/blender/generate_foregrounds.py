import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc

import json
import shutil
import time
from functools import partial
from pathlib import Path

import bpy  # noqa: F401 # isort:skip
import argparse
import os
import random
from math import radians, sqrt
from operator import itemgetter
from typing import Optional, Union

import PIL.Image as Image
from blenderproc.python.types.EntityUtility import Entity
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.utility.Utility import stdout_redirected
from mathutils import Euler, Vector
from uav_data_generation.blender import Checkpoint, COCOWriter
from uav_data_generation.blender.camera import align_camera_pose
from uav_data_generation.blender.config import AREA_RANGES, BaseConfig
from uav_data_generation.blender.drone import (
    randomize_drone_properties,
)
from uav_data_generation.blender.setup import setup
from uav_data_generation.blender.utils.bbox import (
    bbox_overlaps,
    find_bbox_xyxy_by_alpha,
    find_overlap_bbox,
)
from uav_data_generation.blender.utils.material import collect_materials_by_cp
from uav_data_generation.blender.utils.utils import (
    get_cp,
    reset_keyframes,
)
from uav_data_generation.data.datasets import ImageFolder
from uav_data_generation.distributed import (
    get_local_rank,
    get_rank,
    init_dist,
    is_distributed,
    is_main_process,
)
from uav_data_generation.logging import get_logger, raise_error
from uav_data_generation.utils.notification import notify
from uav_data_generation.utils.profiling import profile


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
        "--seed",
        default=2024,
        type=int,
        help="The seed for random sampling.",
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
    parser.add_argument(
        "--distributed",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable the distributed mode.",
    )
    parser.add_argument(
        "--profile",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable the profiling.",
    )
    parser.add_argument(
        "--profile-out-file",
        type=str,
        default="./cprofile.prof",
        help="Enable the profiling.",
    )
    args = parser.parse_args()

    return args


def generate_uav_samples(
    uav_models: list[list[Entity]],
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

    for uav_entities in uav_models:
        uav_meshes: list[MeshObject] = bproc.filter.all_with_type(
            uav_entities, filtered_data_type=MeshObject
        )

        # show entities of the current uav model
        for uav_entity in uav_entities:
            visibility = get_cp(uav_entity, "visibility", default=True)
            uav_entity.hide(not visibility)
            uav_entity.blender_obj.hide_viewport = not visibility

        # TODO: encapsulate UAV model as a class
        # TODO: validate UAV model configurations
        # save UAV model status, e.g. material slots
        original_material_slotss = [
            [
                material_slot.material
                for material_slot in uav_mesh.blender_obj.material_slots
            ]
            for uav_mesh in uav_meshes
        ]

        frame = random.randint(0, 249)

        # randomize the material and rotation
        randomize_drone_properties(
            frame,
            uav_entities,
            materials,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
        )

        # align the camera with the UAV
        align_camera_pose(
            frame,
            uav_entities,
            adaptive_alignment=adaptive_alignment,
            alignment_z_offset=alignment_z_offset,
            alignment_z_step=alignment_z_step,
        )

        # render the whole pipeline
        bproc.utility.set_keyframe_render_interval(frame_start=frame)
        if is_distributed():
            # make blender stdout silent
            with stdout_redirected():
                begin = time.time()
                result = bproc.renderer.render(verbose=True)
                end = time.time()
            logger.info(f"Finished rendering after {end - begin:.3f} seconds")
        else:
            result = bproc.renderer.render()
        image = result["colors"][0]

        # restore UAV model status
        for uav_mesh, original_material_slots in zip(
            uav_meshes, original_material_slotss
        ):
            for i, original_material_slot in enumerate(original_material_slots):
                uav_mesh.blender_obj.material_slots[i].material = original_material_slot

        # hide current entities for next rendering
        for uav_entity in uav_entities:
            uav_entity.hide()
            uav_entity.blender_obj.hide_viewport = True

        # reset keyframes
        reset_keyframes(original_action_keys)

        yield image


def sample_uav_size(
    area_ranges: AREA_RANGES,
    image_size: tuple[int, int],
    uav_size: tuple[int, int],
    allow_upscaling: bool = False,
) -> tuple[int, int]:
    image_w, image_h = image_size
    uav_w, uav_h = uav_size
    image_area = image_w * image_h
    uav_area = uav_w * uav_h

    for _ in range(50):
        # 1. select coco area
        area_start, area_end = random.choice(area_ranges)
        area_end = min(area_end, image_area + 1)

        # if image area is smaller or equal to the selected area range, then skip it.
        if area_start >= area_end:
            continue

        # 2. randomly sample a area from [a, b)
        area = random.randrange(area_start, area_end)

        # if not allow upscaling and larger than the original UAV area, then skip it.
        if area > uav_area and not allow_upscaling:
            continue

        # 3. calculate scaled width, height
        # w * r, h * r = (W, H)
        # w * h * r^2 ~= area
        # r = sqrt(area / uav_area)
        uav_scale_ratio = sqrt(area / uav_area)
        # round to the closet integer & round half to even (default rounding mode in IEEE 754)
        scaled_uav_size = (
            round(uav_w * uav_scale_ratio),
            round(uav_h * uav_scale_ratio),
        )

        # if edge size of scaled uav is still larger than image size, then skip it.
        if scaled_uav_size[0] > image_w or scaled_uav_size[1] > image_h:
            continue

        return scaled_uav_size

    raise_error(
        RuntimeError(
            f"Cannot find an ideal area to fit the ranges. UAV: {uav_size}, Image: {image_size}."
        )
    )


def sample_uav_location(
    image_size: tuple[int, int],
    uav_size: tuple[int, int],
    bboxes: list[tuple[int, int, int, int]],
    min_uav_image_iof: float = 1,
    max_uavs_iof: float = 0,
) -> Optional[tuple[int, int]]:
    logger = get_logger()
    half_w, half_h = image_size[0] // 2, image_size[1] // 2
    start_w, start_h = -half_w, -half_h
    end_w, end_h = image_size[0] + half_w, image_size[1] + half_h

    image_bbox = [[0, 0, *image_size]]
    for _ in range(50):
        # randomly sample a position from image based on the actual size
        x = random.randint(start_w, end_w)
        y = random.randint(start_h, end_h)

        bbox = [[x, y, *uav_size]]
        # check if there is no overlap (or below the overlap threshold) between bbox and image
        iof = bbox_overlaps(bbox, image_bbox, mode="iof")
        if iof[0] < min_uav_image_iof:
            continue

        # check if there is no overlap (or below the overlap threshold) among bboxes list
        iofs1 = bbox_overlaps(bbox, bboxes, mode="iof")
        iofs2 = bbox_overlaps(bboxes, bbox, mode="iof")
        if (iofs1 <= max_uavs_iof).all() and (iofs2 <= max_uavs_iof).all():
            return x, y

    logger.warning(
        f"[bold yellow]Warning! Cannot find an ideal location to fit the maximum IoF {max_uavs_iof}."
    )
    return None


def save_checkpoint(checkpoint: Checkpoint, out_file: Path, max_checkpoints: int):
    checkpoint.save_pickle(out_file)

    # remove oldest checkpoints
    checkpoint_files = out_file.parent.glob("*.pkl")
    checkpoint_files = map(
        lambda checkpoint_file: (checkpoint_file, os.path.getmtime(checkpoint_file)),
        checkpoint_files,
    )
    checkpoint_files = sorted(checkpoint_files, key=itemgetter(1))
    for checkpoint_file, mtime in checkpoint_files[:-max_checkpoints]:
        os.remove(checkpoint_file)


@notify()
def generate_foregrounds(
    config: BaseConfig,
    images_path: Union[str, Path],
    x_range: tuple[int, int] = (-45, 45),
    y_range: tuple[int, int] = (-45, 45),
    z_range: tuple[int, int] = (0, 360),
    sample_range: tuple[int, int] = (1, 20),
    max_uavs_iof: float = 0.5,
    min_uav_image_iof: float = 0.5,
    area_ranges: AREA_RANGES = (
        (1**2, 32**2),
        (32**2, 96**2),
        (96**2, 100000**2),
    ),
    allow_upscaling: bool = False,
    adaptive_alignment: bool = True,
    alignment_z_offset: float = 0,
    alignment_z_step: float = 0.1,
    out_dir: Union[str, Path] = "outputs",
    device_type: str = "OPTIX",
    devices: list[int] = [0],
    checkpoint: Optional[Checkpoint] = None,
    max_checkpoints: int = 3,
    checkpoint_interval: int = 5,
    **kwargs,
):
    logger = get_logger()
    rank = get_rank()
    objs, uav_models = setup(config, device_type, devices)
    materials = collect_materials_by_cp()

    # prepare folders
    out_dir = Path(out_dir)
    checkpoints_dir = out_dir / "checkpoints" / f"rank_{rank}"
    fg_images_dir = out_dir / "foregrounds"
    annotations_dir = out_dir / "annotations"

    # prepare COCO writer
    if checkpoint is not None:
        coco_writer = checkpoint.coco_writer
        category_id = coco_writer.find_category("drone", "UAV")
    else:
        coco_writer = COCOWriter()
        category_id = coco_writer.add_category("drone", "UAV")

    # prepare dataset & validate the checkpoint
    dataset = ImageFolder(images_path)
    if checkpoint is not None:
        if checkpoint.dataset != dataset:
            raise_error(RuntimeError("The dataset is invalid."))
        if not coco_writer.validate(fg_images_dir):
            raise_error(RuntimeError("The COCO writer is invalid."))

    # place the camera in front of the UAV model
    camera = bpy.context.scene.camera
    camera.location = Vector((0, -1, 0))
    camera.rotation_euler = Euler((radians(90), 0, 0))

    # restore random state
    if checkpoint is not None:
        checkpoint.restore_random_states()
        logger.info(":white_check_mark: Restored the random states!")

    uav_models = list(uav_models.values())
    start_index = checkpoint.image_index + 1 if checkpoint is not None else 0
    for image_index in range(start_index, len(dataset)):
        background_image, image_path = dataset[image_index]

        # create an foreground image with the same size as the image
        foreground_image = Image.new("RGBA", background_image.size)
        background_image.close()

        # determine how many samples should be generated
        num_samples = random.randint(*sample_range)
        selected_models = random.choices(uav_models, k=num_samples)

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

            # determine the scaled size of UAV
            scaled_uav_size = sample_uav_size(
                area_ranges,
                image_size,
                uav_image.size,
                allow_upscaling=allow_upscaling,
            )

            # determine the location of UAV on the foreground image
            uav_location = sample_uav_location(
                image_size,
                scaled_uav_size,
                uav_bboxes,
                min_uav_image_iof=min_uav_image_iof,
                max_uavs_iof=max_uavs_iof,
            )
            if uav_location is None:
                logger.info("Skipping...")
                continue

            # scale the UAV
            # filter comparison: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table
            uav_image = uav_image.resize(scaled_uav_size, Image.LANCZOS)

            uav_images.append(uav_image)
            uav_bboxes.append((*uav_location, *uav_image.size))

        logger.info(f"Success: {len(uav_bboxes)} of {num_samples}.")

        ## Post-processing
        # paste UAVs starting from the most distant UAV
        # sort by uav_size
        indices = list(range(len(uav_bboxes)))
        indices.sort(key=lambda i: uav_bboxes[i][2] * uav_bboxes[i][3])
        for i in indices:
            uav_image, bbox = uav_images[i], uav_bboxes[i]
            foreground_image.alpha_composite(uav_image, dest=bbox[:2])
            # remove the area outside of the image
            uav_bboxes[i] = find_overlap_bbox(bbox, (0, 0, *foreground_image.size))

        # get output path
        rel_path = image_path.relative_to(images_path)
        out_file = rel_path.with_suffix(".png")

        # add image info to COCOWriter
        image_id = coco_writer.add_image(out_file.as_posix(), *foreground_image.size)
        coco_writer.add_annotations(image_id, category_id, uav_bboxes)

        # save the foreground image
        out_file = fg_images_dir / out_file
        out_file.parent.mkdir(parents=True, exist_ok=True)
        foreground_image.save(out_file)
        logger.info(f"[bold bright_green]Saved the foreground image as {out_file}.")

        # save the checkpoint every checkpoint_interval
        if (image_index + 1) % checkpoint_interval == 0:
            checkpoint = Checkpoint(rank, image_index, dataset, coco_writer)
            out_file = checkpoints_dir / f"blender_state_{image_index + 1}.pkl"
            save_checkpoint(checkpoint, out_file, max_checkpoints)

    if len(dataset) != len(coco_writer.images):
        raise_error(RuntimeError("The COCO annotations are corrupted."))

    # save annotations in COCO format
    os.makedirs(annotations_dir, exist_ok=True)
    out_file = os.path.join(annotations_dir, "foreground.json")
    coco_writer.export(out_file)


def setup_dist(cfg: BaseConfig, args: argparse.Namespace):
    if args.distributed:
        # initialize the MPI
        init_dist()
    del args.distributed

    rank = get_rank()
    local_rank = get_local_rank()
    logger = get_logger()

    if args.out_dir is not None:
        cfg.out_dir = os.path.expanduser(args.out_dir)

    # resume from the checkpoint
    ckpt = None
    if args.resume:
        if args.checkpoint is None:
            ckpt_root_path = os.path.join(cfg.out_dir, "checkpoints", f"rank_{rank}")
            ckpt, args.checkpoint = Checkpoint.from_latest(ckpt_root_path)
        else:
            ckpt = Checkpoint.from_pickle(args.checkpoint)
        if ckpt.rank != rank:
            raise_error(RuntimeError("The checkpoint is not for this process."))
        logger.info(
            f":white_check_mark: Resumed from the last checkpoint, {args.checkpoint}."
        )

    if is_main_process():
        # save args and config
        with open(os.path.join(cfg.out_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        shutil.copy2(args.config_path, os.path.join(cfg.out_dir, "config.py"))
    del args.config_path

    # seeding with rank offset
    args.seed = args.seed + rank

    # allocate GPU devices based on rank
    # ex. rank 3 with [0, 1] => [3 * num_devices, 3 * num_devices + 1]
    num_devices = len(args.devices)
    base_offset = local_rank * num_devices
    args.devices = [base_offset + device for device in args.devices]

    return ckpt


if __name__ == "__main__":
    # checkpoint
    # 1. random state
    # 2. image paths (checking if it is consistent with argument & validating if the generated images exist)
    # 3. current progress of generated images (index)
    # 4. COCOWriter

    args = parse_args()
    cfg = BaseConfig.from_file(args.config_path)
    ckpt = setup_dist(cfg, args)

    logger = get_logger()

    args = vars(args)
    logger.info(f"Arguments: {args}")
    logger.info(cfg)

    args.pop("out_dir")
    args.pop("resume")
    args.pop("checkpoint")
    seed = args.pop("seed")
    os.environ["BLENDER_PROC_RANDOM_SEED"] = str(seed)

    # TODO: refactor to use class
    use_profile = args.pop("profile")
    profile_out_file = args.pop("profile_out_file")
    func = partial(generate_foregrounds, cfg, **cfg.to_dict(), **args, checkpoint=ckpt)
    if use_profile and is_main_process():
        profile(func, out_file=profile_out_file)
    else:
        func()
