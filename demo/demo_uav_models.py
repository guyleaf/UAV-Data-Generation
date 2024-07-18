import blenderproc as bproc  # isort:skip, this should be at the top due to the check of blenderproc

import argparse
import os
import sys
from typing import Optional

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.append(os.path.dirname(__file__))

from utils import get_cp, reset_keyframes, setup


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
    for i, (name, uav_components) in enumerate(uav_models.items()):
        print("\nUAV name:", name)

        # show components of the current uav model
        for uav_component in uav_components:
            visibility = get_cp(uav_component, "visibility", default=True)
            uav_component.hide(not visibility)

        # find point of interest, all cam poses should look towards it
        poi = bproc.object.compute_poi(uav_components)

        # add translational random walk on top of the POI
        # poi_drift = bproc.sampler.random_walk(
        #     total_length=samples,
        #     dims=3,
        #     step_magnitude=0.005,
        #     window_size=5,
        #     interval=[-0.03, 0.03],
        #     distribution="uniform",
        # )

        # select current UAV model
        bpy.ops.object.select_all(action="DESELECT")
        uav_model = uav_components[0]
        bpy.context.view_layer.objects.active = uav_model.blender_obj
        uav_model.select()
        bpy.ops.object.select_hierarchy(direction="CHILD", extend=True)

        for frame in range(samples):
            # set the current frame in order to get the correct animtation to let the camera fit the object
            bpy.context.scene.frame_set(frame)

            # 1. determine the location of camera
            # camera trajectory that defines a quater circle at constant height
            location_cam = np.array(
                [
                    1 * np.cos(frame / samples * np.pi * 2),
                    1 * np.sin(frame / samples * np.pi * 2),
                    0,
                ]
            )
            # compute rotation based on vector going from location towards poi + drift
            # rotation_matrix = bproc.camera.rotation_from_forward_vec(
            #     poi + poi_drift[frame] - location_cam
            # )
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location_cam)
            # add homog cam pose based on location and rotation
            cam2world_matrix = bproc.math.build_transformation_mat(
                location_cam, rotation_matrix
            )

            # 2. align the camera view to fit UAV
            camera = bpy.context.scene.camera
            camera.matrix_world = Matrix(cam2world_matrix)
            bpy.ops.view3d.camera_to_view_selected()

            # 3. move the camera -0.1m along local Z-axis (0, 0, 1)
            move_amount = 0.05

            # by default, the camera view direction is local -Z axis in blender.
            # so, we just take the local Z-axis. (move backward)
            backward_vector = Vector((0, 0, move_amount))

            # transform local vector to global vector by rotation matrix (ignore scale)
            backward_vector = camera.rotation_euler.to_matrix() @ backward_vector

            # translate location
            # it is equivalent to `camera.location += backward_vector; bpy.context.view_layer.update(); cam2world_matrix = camera.matrix_world`
            cam2world_matrix = Matrix.Translation(backward_vector) @ camera.matrix_world

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
            frame_duration_in_ms=round(1000 / bpy.context.scene.render.fps),
            append_to_existing_output=True,
        )
        # bproc.writer.write_hdf5(os.path.join(out_dir, name), data)

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
