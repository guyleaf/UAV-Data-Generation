# requires
# 2. foreground images
# 3. COCO annotations
# 4. stylized images
# output
# 1. filtered COCO annotations (filtered.json)
# 2. filtered stylized images
# 3. failed stylized images

import argparse
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor
from transformers.models.sam.modeling_sam import SamImageSegmentationOutput

from uav_data_generation.datasets import CocoDetectionWithMask
from uav_data_generation.utils.bbox import convert_xywh_to_xyxy

DEFAULT_WEATHERS = ["foggy", "rainy", "snowy"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter stylized images with SAM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root_dir", type=str, help="root path of synthetic dataset")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="stylized",
        help="relative path of stylized images folder",
    )
    parser.add_argument(
        "--foregrounds-dir",
        type=str,
        default="foregrounds",
        help="relative path of foregrounds (masks) folder",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="annotations/foreground.json",
        help="relative path of COCO annotation file",
    )
    parser.add_argument(
        "--images-out-dir",
        type=str,
        default="filtered_stylized",
        help="relative path of the images output folder.",
    )
    parser.add_argument(
        "--annotation-out-dir",
        type=str,
        default="annotations",
        help="relative path of annotation output folder",
    )

    parser.add_argument(
        "--weathers",
        type=str,
        nargs="+",
        default=DEFAULT_WEATHERS,
        help="filter these weathers",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/sam-vit-huge",
        help="the SAM model repo from huggingface",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    return args


# annotation reference: https://github.com/python/typeshed/blob/27286c682117cafe1efc1305737a4ea25adfc7bb/stubs/pycocotools/pycocotools/coco.pyi#L28C1-L35C17
def preprocess(
    inputs: tuple[Image.Image, Image.Image],
    annotations: list[dict],
    weathers: list[str] = DEFAULT_WEATHERS,
):
    image, mask = inputs
    bboxes = list(map(itemgetter("bbox"), annotations))

    image_size = image.size
    mask_size = mask.size

    # if stylized image size is changed,
    # we need to scale the mask and bboxes to match the size
    if image_size != mask_size:
        mask = mask.resize(image_size, resample=Image.Resampling.NEAREST)
        x_scale = image_size[0] / mask_size[0]
        y_scale = image_size[1] / mask_size[1]
        bboxes = [
            (
                round(x * x_scale),
                round(y * y_scale),
                round(w * x_scale),
                round(h * y_scale),
            )
            for x, y, w, h in bboxes
        ]

    # image = to_tensor(image)
    # mask = to_tensor(mask)
    return (image, mask), convert_xywh_to_xyxy(bboxes)


@torch.inference_mode()
def generate_mask(
    processor: SamProcessor,
    model: SamModel,
    image: Image.Image,
    bboxes: list[tuple[int, int, int, int]],
    mask_threshold: float = 0,
) -> tuple[SamImageSegmentationOutput, torch.Tensor]:
    inputs = processor(image, input_boxes=[bboxes], return_tensors="pt").to(
        model.device
    )
    outputs = model(**inputs)

    # [1, B, C, H, W]
    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
        mask_threshold=mask_threshold,
    )
    # scores = outputs.iou_scores

    # masks = masks[0].squeeze()
    # index = torch.argmax(scores[0])
    # mask_image = masks[index] * 1.0

    return outputs, masks


def filter_stylized_images(
    root_dir: Union[str, Path],
    images_dir: str = "stylized",
    foregrounds_dir: str = "foregrounds",
    annotation_file: str = "annotations/foreground.json",
    images_out_dir: str = "filtered_stylized",
    annotation_out_dir: str = "annotations",
    weathers: list[str] = DEFAULT_WEATHERS,
    model: str = "facebook/sam-vit-huge",
    device: str = "cuda:0",
):
    processor = SamProcessor.from_pretrained(model)
    model = SamModel.from_pretrained(model).to(device)

    root_path = Path(root_dir)
    dataset = CocoDetectionWithMask(
        root_path / images_dir,
        root_path / foregrounds_dir,
        root_path / annotation_file,
        transforms=partial(preprocess, weathers=weathers),
    )
    dataloader = DataLoader(
        dataset,
        # disable automatic batching
        batch_size=None,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    generate_mask_ = partial(generate_mask, processor, model)
    for (image, mask), bboxes in dataloader:
        outputs, masks = generate_mask_(image, bboxes)
        print(outputs.iou_scores.shape)
        break


if __name__ == "__main__":
    args = vars(parse_args())
    filter_stylized_images(**args)
