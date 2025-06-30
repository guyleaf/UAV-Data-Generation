import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc
import bpy  # noqa: F401 # isort:skip

import argparse
import os
import random
from math import radians

from blenderproc.python.types.MeshObjectUtility import MeshObject
from mathutils import Euler, Vector
from PIL import Image
from rich import print
from uav_data_generation.blender.camera import align_camera_pose
from uav_data_generation.blender.config import BaseConfig
from uav_data_generation.blender.drone import randomize_drone_properties
from uav_data_generation.blender.setup import setup
from uav_data_generation.blender.utils.bbox import find_bbox_xyxy_by_alpha
from uav_data_generation.blender.utils.material import collect_materials_by_cp
from uav_data_generation.blender.utils.utils import (
    draw_bbox_xyxy,
    get_cp,
    reset_keyframes,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="An demo script for material randomization powered by blender",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config_path", type=str, help="Path to the config file")

    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Path to where the final files, will be saved",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Override the models setting in the config file.",
    )
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

    return args


def demo_randomization(
    config: BaseConfig,
    x_range: tuple[int, int] = (-45, 45),
    y_range: tuple[int, int] = (-45, 45),
    z_range: tuple[int, int] = (0, 360),
    adaptive_alignment: bool = False,
    alignment_z_offset: float = 0,
    alignment_z_step: float = 0.1,
    out_dir: str = "outputs",
    samples: int = 3,
    device_type: str = "OPTIX",
    devices: list[int] = [0],
    **kwargs,
):
    objs, uav_models = setup(config, device_type, devices)
    materials = collect_materials_by_cp()

    # place the camera in front of the UAV model
    camera = bpy.context.scene.camera
    camera.location = Vector((0, -1, 0))
    camera.rotation_euler = Euler((radians(90), 0, 0))

    original_action_keys = bpy.data.actions.keys()
    for name, uav_entities in uav_models.items():
        print("\nUAV name:", name)
        uav_meshes: list[MeshObject] = bproc.filter.all_with_type(
            uav_entities, filtered_data_type=MeshObject
        )

        # show entities of the current uav model
        for uav_entity in uav_entities:
            visibility = get_cp(uav_entity, "visibility", default=True)
            uav_entity.hide(not visibility)
            uav_entity.blender_obj.hide_viewport = not visibility

        # save UAV model status, e.g. material slots
        original_material_slotss = [
            [
                material_slot.material
                for material_slot in uav_mesh.blender_obj.material_slots
            ]
            for uav_mesh in uav_meshes
        ]

        for i in range(samples):
            frame = random.randint(0, 249)

            randomize_drone_properties(
                frame,
                uav_entities,
                materials,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
            )

            # align the camera with the UAV
            z_offset = align_camera_pose(
                frame,
                uav_entities,
                adaptive_alignment=adaptive_alignment,
                alignment_z_offset=alignment_z_offset,
                alignment_z_step=alignment_z_step,
            )

            # render the whole pipeline
            bproc.utility.set_keyframe_render_interval(frame_start=frame)
            color = bproc.renderer.render()["colors"][0]
            image = Image.fromarray(color, mode="RGBA")

            # visualize with the bounding box
            bbox = find_bbox_xyxy_by_alpha(color)
            draw_bbox_xyxy(image, bbox)

            # write the color to a .png container in the run-specific output directory
            out_path = os.path.join(out_dir, name)
            os.makedirs(out_path, exist_ok=True)
            image.save(os.path.join(out_path, f"{i}_{frame}_{z_offset:.3f}m.png"))

            # reset keyframes
            reset_keyframes(original_action_keys)

            # restore UAV model status
            for uav_mesh, original_material_slots in zip(
                uav_meshes, original_material_slotss
            ):
                for i, original_material_slot in enumerate(original_material_slots):
                    uav_mesh.blender_obj.material_slots[
                        i
                    ].material = original_material_slot

        # hide current entities for next rendering
        for uav_entity in uav_entities:
            uav_entity.hide()
            uav_entity.blender_obj.hide_viewport = True


if __name__ == "__main__":
    args = vars(parse_args())
    cfg = BaseConfig.from_file(args.pop("config_path"))

    print()
    print("Arguments:", args)

    models = args.pop("models")
    if models is not None:
        cfg.models = models
    cfg.out_dir = os.path.expanduser(args.pop("out_dir"))

    print(cfg)

    os.environ["BLENDER_PROC_RANDOM_SEED"] = str(args.pop("seed"))
    demo_randomization(cfg, **cfg.to_dict(), **args)
