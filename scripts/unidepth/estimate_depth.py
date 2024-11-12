import argparse
import mimetypes
from enum import Enum
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from matplotlib.pyplot import get_cmap
from PIL import Image
from rich.progress import track
from torchvision.transforms import functional as TF


class Filter(str, Enum):
    FastBilateral = "fastbilateral"
    Guided = "guided"

    def __str__(self):
        return self.value


def parse_args():
    parser = argparse.ArgumentParser(
        description="estimate metric depth map by UniDepth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "images_dir",
        type=str,
        help="images folder for estimating relative depths",
    )
    parser.add_argument("out_dir", type=str, help="output folder of the depth images")

    parser.add_argument(
        "--repo",
        type=str,
        default="lpiccinelli-eth/UniDepth",
        help="the repo name from torch hub",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="UniDepth",
        help="the model name from torch hub repo",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        help="the version of the model",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="vitl14",
        help="the backbone of the model",
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


def collect_images(path: Union[str, Path]) -> list[Path]:
    mime_checker = mimetypes.MimeTypes()

    def validate_file_type(path: Path):
        mime_type = mime_checker.guess_type(path)[0]
        return mime_type is not None and mime_type.startswith("image")

    path = Path(path)
    if path.is_dir():
        images = filter(validate_file_type, path.rglob("*.*"))
    else:
        images = [path]

    return sorted(images)


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
    model: torch.nn.Module,
    image_paths: list[Path],
    refine: bool = True,
    rfilter: Filter = Filter.Guided,
):
    for image_path in image_paths:
        # load image
        image = Image.open(image_path).convert("RGB")
        image_size = image.size[::-1]
        image_pt = TF.to_tensor(image)

        # solve creating a new tensor in xformers
        # bug issue: https://github.com/facebookresearch/xformers/issues/681#issuecomment-1942098559
        with torch.cuda.device(model.device):
            # the model will do pre-processing and .to(device) automatically
            # api: https://github.com/lpiccinelli-eth/UniDepth/blob/bebc4b2fdda5c223693c390739db76897aee42e5/unidepth/models/unidepthv2/unidepthv2.py#L202
            outputs: dict = model.infer(image_pt)

        # [H, W]
        predicted_depth: torch.Tensor = outputs["depth"]
        depth = predicted_depth.squeeze().cpu().numpy()
        assert image_size == depth.shape

        if refine:
            # refine the depth map by filter
            depth = refine_depth(
                np.array(image), depth, max_depth=depth.max(), rfilter=rfilter
            )

        yield depth


if __name__ == "__main__":
    args = parse_args()

    # load model
    model: torch.nn.Module = torch.hub.load(
        args.repo,
        args.model,
        version=args.version,
        backbone=args.backbone,
        pretrained=True,
        force_reload=True,
        trust_repo="check",
    )
    model.to(args.device)

    # load images
    image_paths = collect_images(args.images_dir)

    # inference
    cmap = get_cmap("Spectral")
    out_path = Path(args.out_dir)

    inferencer = inference(model, image_paths, refine=args.refine, rfilter=args.rfilter)
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
            normalized_depth /= normalized_depth.max()
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
