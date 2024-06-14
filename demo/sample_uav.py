import blenderproc as bproc  # isort:skip, this should be at the top due to the check of blenderproc

import argparse
import os
from collections import defaultdict
from typing import Any, Optional

import bpy
import numpy as np
from blenderproc.python.renderer import RendererUtility
from blenderproc.python.types.EntityUtility import Entity


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("scene_path", type=str, help="Path to the .blend scene file")
    parser.add_argument(
        "--out_dir",
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
        "--device_type",
        default="OPTIX",
        type=str,
        help="The GPU device type for rendering. Possible choices are [CPU, OPTIX, CUDA, METAL, HIP]",
    )
    parser.add_argument(
        "--device",
        default=0,
        type=int,
        help="The GPU device id for rendering. You can check the id by executing list_gpu_devices.py",
    )

    args = parser.parse_args()
    assert args.scene_path.endswith(".blend") and os.path.isfile(args.scene_path)
    return args


def find_collection_by_attr(
    collections: list[bpy.types.Collection], attr_name: str, value: Any
):
    collections = list(
        filter(lambda collection: getattr(collection, attr_name) == value, collections)
    )
    if len(collections) > 1:
        raise Exception(
            "More than one collection with the given condition has been found."
        )
    if len(collections) == 0:
        raise Exception("No collection with the given condition has been found.")
    return collections[0]


def setup(scene_path: str, device_type: str, device: int):
    bproc.init()

    # set render device
    use_only_cpu = device_type == "CPU"
    RendererUtility.set_render_devices(
        use_only_cpu=use_only_cpu,
        desired_gpu_device_type=device_type if not use_only_cpu else None,
        desired_gpu_ids=device,
    )

    # load collections to get UAV names
    collections = bproc.loader.load_blend(scene_path, data_blocks="collections")

    # note: only objects in the obj_types can be loaded
    # otherwise, such as scene settings, they aren't loaded by blenderprc
    objs = bproc.loader.load_blend(scene_path, obj_types=["mesh", "light", "camera"])

    # Setup scene settingss
    bpy.context.scene.render.fps = 30
    bproc.camera.set_resolution(512, 512)

    # find UAV collection
    uav_collection = find_collection_by_attr(collections, "name", "UAVs")
    # nested objects always have at least one comma
    uav_names = list(
        filter(lambda name: name.count(".") == 0, uav_collection.objects.keys())
    )

    pattern = "|".join(uav_names)
    uav_objs: list[Entity] = bproc.filter.by_attr(
        objs, "name", f"^({pattern})\.001$", regex=True
    )

    # organize components for each uav model as dict
    uav_models = defaultdict[str, list[Entity]](list)
    for obj in uav_objs:
        uav_name = obj.get_name().split(".")[0]
        uav_models[uav_name] += [obj] + obj.get_children(return_all_offspring=True)

    assert len(uav_models) > 0, "UAV model is not found."
    return objs, uav_models


def sample_uav(
    scene_path: str,
    models: Optional[list[str]] = None,
    out_dir: str = "outputs",
    samples: int = 3,
    device_type: str = "OPTIX",
    device: int = 0,
):
    objs, uav_models = setup(scene_path, device_type, device)

    # hide all uav components and set categorid_id to 0 as drone category
    for uav_components in uav_models.values():
        for uav_component in uav_components:
            uav_component.set_cp("category_id", 0)
            uav_component.hide()

    if models is not None:
        uav_models = {name: uav_models[name] for name in models}

    for i, (name, uav_components) in enumerate(uav_models.items()):
        print("\nUAV name:", name)

        # show components of the current uav model
        for uav_component in uav_components:
            uav_component.hide(False)

        # find point of interest, all cam poses should look towards it
        poi = bproc.object.compute_poi(uav_components)

        # Add translational random walk on top of the POI
        # poi_drift = bproc.sampler.random_walk(
        #     total_length=samples,
        #     dims=3,
        #     step_magnitude=0.005,
        #     window_size=5,
        #     interval=[-0.03, 0.03],
        #     distribution="uniform",
        # )

        for frame in range(samples):
            # Camera trajectory that defines a quater circle at constant height
            location_cam = np.array(
                [
                    1 * np.cos(frame / samples * np.pi * 2),
                    1 * np.sin(frame / samples * np.pi * 2),
                    1,
                ]
            )
            # Compute rotation based on vector going from location towards poi + drift
            # rotation_matrix = bproc.camera.rotation_from_forward_vec(
            #     poi + poi_drift[frame] - location_cam
            # )
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location_cam)
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(
                location_cam, rotation_matrix
            )
            bproc.camera.add_camera_pose(cam2world_matrix, frame=frame)

        # activate normal rendering
        bproc.renderer.enable_normals_output()
        bproc.renderer.enable_segmentation_output(
            map_by="category_id", default_values=dict(category_id=-1)
        )

        # render the whole pipeline
        data = bproc.renderer.render()

        # write the data to a .hdf5 container in the run-specific output directory
        bproc.writer.write_gif_animation(
            os.path.join(out_dir, name),
            data,
            frame_duration_in_ms=round(1 / bpy.context.scene.render.fps),
            append_to_existing_output=True,
        )

        # hide current components for next rendering
        for uav_component in uav_components:
            uav_component.hide()

        bproc.utility.reset_keyframes()


if __name__ == "__main__":
    args = parse_args()
    os.environ["BLENDER_PROC_RANDOM_SEED"] = str(args.seed)
    sample_uav(
        args.scene_path,
        models=args.models,
        out_dir=args.out_dir,
        samples=args.samples,
        device_type=args.device_type,
        device=args.device,
    )
