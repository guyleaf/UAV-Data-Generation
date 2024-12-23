import argparse
import warnings
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers
from matplotlib.cm import get_cmap
from packaging.version import Version
from PIL import Image
from rich.progress import track
from torch.nn import functional as F
from transformers import DepthEstimationPipeline, pipeline

from uav_data_generation.utils.io import collect_images

# About the focal length
# according to the training method of metric depth estimation, the model is finetuned on Virtual KITTI 2 directly without makeing any affine-invariant operations
# so, for the dataset which camera intrinsic is unknown. I follow Virtual KITTI 2's camera intrinsic for weather augmentation.
# https://github.com/DepthAnything/Depth-Anything-V2/issues/152#issuecomment-2350775873

# reference: https://github.com/DepthAnything/Depth-Anything-V2/issues/152#issuecomment-2350952121
# Focal depth X = 725.0087, focal depth y = 725.0087, W/2 = 620.5, H/2 = 187
# Which corresponds to:
# Field of View (degrees):
#   fov_x = 81.117°
#   fov_y = 28.926°


class Filter(str, Enum):
    FastBilateral = "fastbilateral"
    Guided = "guided"

    def __str__(self):
        return self.value


def parse_args():
    parser = argparse.ArgumentParser(
        description="estimate metric depth map by monocular depth estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "images_dir",
        type=str,
        help="images folder for estimating relative depths",
    )
    parser.add_argument("out_dir", type=str, help="output folder of the depth images")

    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        help="the monocular depth estimation model name from huggingface",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--numpy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="store the raw depth map with .npy file",
    )
    parser.add_argument(
        "--demo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="store the depth map in demo mode.",
    )
    parser.add_argument(
        "--scale-in-meters",
        type=float,
        default=256,
        help="scale the depth map before saving in .png file. some works require 16-bit png with depth_in_meter = depth/256.",
    )
    parser.add_argument(
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to refine depth map.",
    )
    parser.add_argument(
        "--rfilter",
        type=str,
        default=Filter.Guided,
        choices=list(Filter),
        help="the filter is used to refine depth map.",
    )
    args = parser.parse_args()
    args.rfilter = Filter(args.rfilter)
    return args


def calculate_confidence(im_gray: np.ndarray):
    e = cv2.Laplacian(im_gray, cv2.CV_8U)
    e = cv2.dilate(e, np.ones((5, 5)))
    e = cv2.blur(e, (5, 5))
    e = 255 - e

    return e


def refine_depth(
    image: np.ndarray,
    depth: np.ndarray,
    max_depth: float = 255,
    rfilter: Filter = Filter.Guided,
):
    # unit8
    reference = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # float32
    src = np.atleast_3d(depth) / max_depth

    if rfilter == Filter.Guided:
        # api: https://docs.opencv.org/4.10.0/da/d17/group__ximgproc__filters.html#gaee6c3f577560a7693b2bfac9301fbff0
        # follow the default arguments in mathlab, https://www.mathworks.com/help/images/ref/imguidedfilter.html
        radius = 5
        eps = 650.25
        depth = cv2.ximgproc.guidedFilter(reference, src, radius, eps)
    elif rfilter == Filter.FastBilateral:
        # reference: https://github.com/astra-vision/rain-rendering/issues/3#issuecomment-810960053
        image_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        confidence = calculate_confidence(image_gray)
        # api: https://docs.opencv.org/4.10.0/da/d17/group__ximgproc__filters.html#ga503eac1d6f580fefb61f465bccc3cb11
        # confidence = np.ones_like(src)
        depth = cv2.ximgproc.fastBilateralSolverFilter(reference, src, confidence)
    else:
        raise NotImplementedError("Unsupported filter.")
    return depth * max_depth


@torch.inference_mode()
def inference(
    pipe: DepthEstimationPipeline,
    image_paths: list[Path],
    max_depth: int,
    refine: bool = True,
    rfilter: Filter = Filter.Guided,
):
    images = (image_path.as_posix() for image_path in image_paths)
    for image_path, depth in zip(image_paths, pipe(images)):
        # load image
        image = Image.open(image_path).convert("RGB")
        image_size = image.size[::-1]

        f"""
        official: https://github.com/DepthAnything/Depth-Anything-V2/blob/31dc97708961675ce6b3a8d8ffa729170a4aa273/metric_depth/depth_anything_v2/dpt.py#L193
        transformers: {DepthEstimationPipeline.postprocess}

        *Notice: from transformers >= 4.46.0, the predicted_depth will be resized to the image size.
        """
        predicted_depth = depth["predicted_depth"]
        if Version(transformers.__version__) < Version("4.46.0"):
            predicted_depth = predicted_depth.squeeze().unsqueeze(0).unsqueeze(1)
            predicted_depth = F.interpolate(
                predicted_depth,
                size=image_size,
                mode="bicubic",
                align_corners=False,
            )

        # [H, W]
        depth = predicted_depth.squeeze().cpu().numpy()
        assert image_size == depth.shape

        if refine:
            # refine the depth map by filter
            depth = refine_depth(
                np.array(image), depth, max_depth=max_depth, rfilter=rfilter
            )

        yield depth


if __name__ == "__main__":
    args = parse_args()

    # load pipe
    pipe = pipeline(task="depth-estimation", model=args.model, device=args.device)
    config = pipe.model.config.to_dict()

    # load images
    image_paths = collect_images(args.images_dir)

    # inference
    cmap = get_cmap("Spectral")
    # depth-anything v2 only
    config = pipe.model.config.to_dict()
    max_depth = config.get("max_depth", 255)
    out_path = Path(args.out_dir)

    inferencer = inference(
        pipe, image_paths, max_depth, refine=args.refine, rfilter=args.rfilter
    )
    for i, depth in track(
        enumerate(inferencer),
        description="Estimating...",
        total=len(image_paths),
    ):
        image_path = image_paths[i]
        rel_path = image_path.relative_to(args.images_dir)

        if args.numpy:
            out = out_path / "numpy" / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out.with_suffix(".npy"), depth)

        if args.demo:
            normalized_depth = depth.copy()
            if max_depth is None:
                warnings.warn(
                    "max_depth config is not found. it will be normalized by max. it may not be accurate in meters."
                )
                normalized_depth /= normalized_depth.max()
                # normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())
            else:
                normalized_depth /= max_depth
            colored_depth = cmap(normalized_depth, bytes=True)
            colored_depth = Image.fromarray(colored_depth)

            out = out_path / "demo" / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            colored_depth.save(out.with_suffix(".png"))

        # save 16-bit png depth image
        f"Reference: {Image._fromarray_typemap}"
        depth = depth * args.scale_in_meters
        depth = Image.fromarray(depth.astype("uint16"))

        out = out_path / "images" / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        depth.save(out.with_suffix(".png"))
