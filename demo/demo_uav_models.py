import blenderproc as bproc  # isort:skip, this should be at the top due to the check of blenderproc

import argparse
import os
from collections import defaultdict
from typing import Optional

import bpy
import numpy as np
from blenderproc.api.types import MeshObject
from mathutils import Matrix


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
        "--devices",
        default=[0],
        type=int,
        nargs="+",
        help="The GPU device ids for rendering. You can check the id by executing list_gpu_devices.py",
    )

    args = parser.parse_args()
    assert args.scene_path.endswith(".blend") and os.path.isfile(args.scene_path)
    return args


def setup(
    scene_path: str,
    background_path: str,
    device_type: str,
    devices: list[int],
):
    bproc.init()

    # set render device
    use_only_cpu = device_type == "CPU"
    device_type = device_type if not use_only_cpu else None
    bproc.renderer.set_render_devices(
        use_only_cpu=use_only_cpu,
        desired_gpu_device_type=device_type if not use_only_cpu else None,
        desired_gpu_ids=devices,
    )

    # note: only objects in the obj_types can be loaded
    # otherwise, such as scene settings, they aren't loaded by blenderprc
    objs = bproc.loader.load_blend(scene_path, obj_types=["mesh", "light"])

    # Setup scene settingss
    bpy.context.scene.render.fps = 60
    bproc.camera.set_resolution(1920, 1080)
    bproc.renderer.set_output_format(enable_transparency=True)
    # bproc.renderer.set_max_amount_of_samples(4096)
    bproc.renderer.enable_motion_blur(motion_blur_length=0.5)

    print("Loading the background:", os.path.basename(background_path))
    bproc.world.set_world_background_hdr_img(background_path)

    # collect UAV models by custom property
    uav_objs = bproc.filter.by_cp(objs, "UAV_model", True)

    # organize components for each uav model as dict
    uav_models: dict[str, list[MeshObject]] = defaultdict(list)
    for obj in uav_objs:
        uav_name = obj.get_name()
        uav_models[uav_name] += [obj] + obj.get_children(return_all_offspring=True)

    assert len(uav_models) > 0, "UAV model is not found."
    print("Find", len(uav_models), "UAV models")
    return objs, uav_models


def reset_keyframes(original_action_keys: list[str]) -> None:
    """Removes registered keyframes from all objects which are not in original_action_keys and resets frame_start and frame_end"""
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0

    # clear all camera poses among keyframes
    tbd_action_keys = set(bpy.data.actions.keys()) - set(original_action_keys)
    for action_key in tbd_action_keys:
        action = bpy.data.actions[action_key]
        bpy.data.actions.remove(action)


def get_cp(object: MeshObject, key: str, default=None):
    if object.has_cp(key):
        return object.get_cp(key)
    else:
        return default


def demo_uav_models(
    scene_path: str,
    background_path: str,
    models: Optional[list[str]] = None,
    out_dir: str = "outputs",
    samples: int = 3,
    device_type: str = "OPTIX",
    devices: list[int] = [0],
):
    objs, uav_models = setup(scene_path, background_path, device_type, devices)

    # hide all uav components and set categorid_id to 0 as drone category
    for uav_components in uav_models.values():
        for uav_component in uav_components:
            uav_component.set_cp("category_id", 1)
            uav_component.hide()

    if models is not None:
        uav_models = {name: uav_models[name] for name in models}

    original_action_keys = bpy.data.actions.keys()
    for i, (name, (uav_model, *uav_components)) in enumerate(uav_models.items()):
        print("\nUAV name:", name)

        # show components of the current uav model
        for uav_component in uav_components:
            visibility = get_cp(uav_component, "visibility", default=True)
            uav_component.hide(not visibility)

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

        # Select current UAV model
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = uav_model.blender_obj
        uav_model.select()
        bpy.ops.object.select_hierarchy(direction="CHILD", extend=True)

        for frame in range(samples):
            # Camera trajectory that defines a quater circle at constant height
            location_cam = np.array(
                [
                    0.5 * np.cos(frame / samples * np.pi * 2),
                    0.5 * np.sin(frame / samples * np.pi * 2),
                    0.8,
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

            # Align the camera view to fit UAV
            print(cam2world_matrix)
            camera = bpy.context.scene.camera
            camera.matrix_world = Matrix(cam2world_matrix)
            bpy.ops.view3d.camera_to_view_selected()
            cam2world_matrix = camera.matrix_world
            print(cam2world_matrix)

            bproc.camera.add_camera_pose(cam2world_matrix, frame=frame)

        # activate segment rendering
        bproc.renderer.enable_segmentation_output(
            map_by="category_id", default_values=dict(category_id=0)
        )

        # render the whole pipeline
        data = bproc.renderer.render()

        # write the data to a .hdf5 container in the run-specific output directory
        bproc.writer.write_gif_animation(
            os.path.join(out_dir, name),
            data,
            # frame_duration_in_ms=round(1 / bpy.context.scene.render.fps),
            append_to_existing_output=True,
        )

        # hide current components for next rendering
        for uav_component in uav_components:
            uav_component.hide()

        # reset keyframes
        reset_keyframes(original_action_keys)


if __name__ == "__main__":
    args = parse_args()
    os.environ["BLENDER_PROC_RANDOM_SEED"] = str(args.seed)
    demo_uav_models(
        args.scene_path,
        args.background_path,
        models=args.models,
        out_dir=args.out_dir,
        samples=args.samples,
        device_type=args.device_type,
        devices=args.devices,
    )
