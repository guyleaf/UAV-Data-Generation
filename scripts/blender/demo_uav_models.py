import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc

import argparse
import os

import bpy
import numpy as np
from blenderproc.python.types.MeshObjectUtility import MeshObject
from mathutils import Matrix, Vector
from rich import print

from uav_data_generation.blender.config import BaseConfig
from uav_data_generation.blender.drone import randomize_drone_materials
from uav_data_generation.blender.setup import setup
from uav_data_generation.blender.utils.geometry import translate_axis
from uav_data_generation.blender.utils.material import collect_materials_by_cp
from uav_data_generation.blender.utils.mesh import compute_poi
from uav_data_generation.blender.utils.utils import (
    get_cp,
    reset_keyframes,
    select_objects,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="An demo script for UAV models powered by blender",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config_path", type=str, help="Path to the config file")

    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Path to where the final files, will be saved.",
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


def calculate_matrix_world_from_poi(
    poi: np.ndarray, radius: float, frame: int, samples: int, z_offset: float = 0.3
):
    location_cam = np.array(
        [
            radius * np.cos(frame / samples * np.pi * 2),
            radius * np.sin(frame / samples * np.pi * 2),
            poi[-1] + z_offset,
        ]
    )
    # compute rotation based on vector going from location towards poi + drift
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location_cam)
    # add homog cam pose based on location and rotation
    cam2world_matrix = Matrix(
        bproc.math.build_transformation_mat(location_cam, rotation_matrix)
    )
    return cam2world_matrix


def demo_uav_models(
    config: BaseConfig,
    alignment_z_offset: float = 0,
    samples: int = 60,
    out_dir: str = "outputs",
    device_type: str = "OPTIX",
    devices: list[int] = [0],
    **kwargs,
):
    objs, uav_models = setup(config, device_type, devices)
    materials = collect_materials_by_cp()
    camera = bpy.context.scene.camera

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

        select_objects(uav_entities)

        # randomize the material and rotation
        randomize_drone_materials(uav_entities, materials)

        # find point of interest, all cam poses should look towards it
        poi, min_poi, max_poi = compute_poi(uav_meshes)

        for z in [max_poi[2] + 0.2, min_poi[2] - 0.2]:
            z_offset = z - poi[2]
            # confirm the distance between the camera and the model
            bpy.context.scene.frame_set(0)
            camera.matrix_world = calculate_matrix_world_from_poi(
                poi, 1, 0, samples, z_offset=z_offset
            )
            bpy.ops.view3d.camera_to_view_selected()
            radius = (camera.location - Vector(poi)).magnitude

            for frame in range(samples):
                bpy.context.scene.frame_set(frame)

                # align the camera view to the model
                camera.matrix_world = calculate_matrix_world_from_poi(
                    poi, radius, frame, samples, z_offset=z_offset
                )
                translate_axis(camera, "Z", alignment_z_offset)
                bproc.camera.add_camera_pose(camera.matrix_world, frame=frame)

            # render the whole pipeline
            data = bproc.renderer.render()

            # write the data to a .hdf5 container in the run-specific output directory
            bproc.writer.write_gif_animation(
                os.path.join(out_dir, name),
                data,
                frame_duration_in_ms=300,
                # frame_duration_in_ms=round(1000 / bpy.context.scene.render.fps),
                append_to_existing_output=True,
            )
            # bproc.writer.write_hdf5(os.path.join(out_dir, name), data)

            # reset keyframes
            reset_keyframes(original_action_keys)

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
    demo_uav_models(cfg, **cfg.to_dict(), **args)
