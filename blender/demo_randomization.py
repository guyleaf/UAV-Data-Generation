import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc
import bpy  # noqa: F401 # isort:skip
import argparse
import os
import sys
from math import radians
from typing import Optional

from mathutils import Euler, Vector
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(__file__))

from randomization import (
    align_camera_pose,
    group_and_filter_material_slots_by_cp,
    randomize_drone_properties,
)
from utils import (
    collect_materials_by_cp,
    find_bbox_xyxy_by_alpha,
    get_cp,
    reset_keyframes,
    setup,
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
        "--adaptive-alignment",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable adaptive alignment with the step (useful with motion blur)",
    )
    parser.add_argument(
        "--alignment-z-offset",
        type=float,
        default=0,
        help="Align the camera with the UAV and move backward with the offset (m) (useful with motion blur)",
    )
    parser.add_argument(
        "--alignment-z-step",
        type=float,
        default=0.001,
        help="Increase the alignment distance with the step (m) (--adaptive-alignment only)",
    )

    # render settings
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
        default=2048,
        help="The tile size for rendering a image (less -> slower & lower VRAM requirement, higher -> faster & higher VRAM requirement)",
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
    x_range: tuple[int, int] = (-45, 45),
    y_range: tuple[int, int] = (-45, 45),
    z_range: tuple[int, int] = (0, 360),
    adaptive_alignment: bool = False,
    alignment_z_offset: float = 0,
    alignment_z_step: float = 0.1,
    motion_blur: bool = True,
    render_resolution: tuple[int, int] = (1920, 1920),
    render_max_samples: int = 1024,
    render_tile_size: int = 1024,
    out_dir: str = "outputs",
    samples: int = 3,
    device_type: str = "OPTIX",
    devices: list[int] = [0],
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

    original_action_keys = bpy.data.actions.keys()
    for i, (name, uav_components) in enumerate(uav_models.items()):
        uav_model = uav_components[0]
        print("\nUAV name:", name)

        # show components of the current uav model
        for uav_component in uav_components:
            visibility = get_cp(uav_component, "visibility", default=True)
            uav_component.hide(not visibility)

        # get material slots which require material_randomization
        material_slots_groups = group_and_filter_material_slots_by_cp(uav_components)

        for i in range(samples):
            frame = randomize_drone_properties(
                uav_model,
                material_slots_groups,
                materials,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
            )

            # align the camera with the UAV
            z_offset = align_camera_pose(
                frame,
                uav_components,
                adaptive_alignment=adaptive_alignment,
                alignment_z_offset=alignment_z_offset,
                alignment_z_step=alignment_z_step,
                motion_blur=motion_blur,
            )

            # render the whole pipeline
            bproc.utility.set_keyframe_render_interval(frame_start=frame)
            data = bproc.renderer.render()
            color = data["colors"][0]
            image = Image.fromarray(color, mode="RGBA")

            # visualize with the bounding box
            bbox = find_bbox_xyxy_by_alpha(color)
            draw_bounding_box(image, bbox)

            # write the color to a .png container in the run-specific output directory
            out_path = os.path.join(out_dir, name)
            os.makedirs(out_path, exist_ok=True)
            image.save(os.path.join(out_path, f"{i}_{frame}_{z_offset:.3f}.png"))

            # reset keyframes
            reset_keyframes(original_action_keys)

        # hide current components for next rendering
        for uav_component in uav_components:
            uav_component.hide()


if __name__ == "__main__":
    args = vars(parse_args())
    os.environ["BLENDER_PROC_RANDOM_SEED"] = str(args.pop("seed"))
    main(args.pop("scene_path"), args.pop("background_path"), **args)
