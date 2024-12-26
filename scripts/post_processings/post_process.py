import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

from PIL import Image
from pycocotools.coco import COCO
from rich import print
from rich.progress import track

from uav_data_generation.utils.dataset import split_into_train_val

NOT_TO_COPY_KEYS = {"images", "annotations"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="A post-processing script for synthetic dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root_dir", type=str, help="root path of synthetic dataset")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="stylized",
        help="relative path of images folder",
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
        default=None,
        help="relative path of the images output folder. If None, not to copy images.",
    )
    parser.add_argument(
        "--annotation-out-dir",
        type=str,
        default="annotations",
        help="relative path of annotation output folder",
    )
    # parser.add_argument(
    #     "--copy",
    #     action=argparse.BooleanOptionalAction,
    #     default=False,
    #     help="always copy images. If False and root_dir == out_dir, then not perform copying. Otherwise, perform copying.",
    # )

    parser.add_argument(
        "--val-ratio", type=float, default=0.3333, help="ratio for validation subset"
    )
    parser.add_argument(
        "--seed", type=int, default=2024, help="seed for random splitting"
    )

    args = parser.parse_args()

    assert 0 <= args.val_ratio <= 1, "The val_ratio should be in [0, 1]."

    return args


def update_coco_annotation(coco: COCO, images_path: Path):
    coco = deepcopy(coco)
    for image_id in track(coco.getImgIds(), description="Updating COCO..."):
        image = coco.loadImgs(image_id)[0]
        file_name = image["file_name"]

        # add the label(caption) to image
        # e.g. Image2Weather/{weather}/xxxx.png
        label = file_name.split("/")[1]
        image["caption"] = label
        image["weight"] = 1.0

        # update width, height
        image_size = image["width"], image["height"]
        with Image.open(images_path / file_name) as img:
            new_image_size = img.size
        image["width"], image["height"] = new_image_size

        # adjust the scale of annotations if image size is changed
        # here, make this function for every cases. so, we don't check the aspect ratio.
        if image_size != new_image_size:
            x_scale = new_image_size[0] / image_size[0]
            y_scale = new_image_size[1] / image_size[1]
            annotation_ids = coco.getAnnIds(imgIds=image_id)
            for annotation in coco.loadAnns(annotation_ids):
                x, y, w, h = annotation["bbox"]
                x = round(x * x_scale)
                y = round(y * y_scale)
                w = round(w * x_scale)
                h = round(h * y_scale)
                annotation["bbox"] = (x, y, w, h)
                annotation["area"] = w * h
    return coco


def split_coco_annotation(
    coco: COCO, val_ratio: float, seed: int = 2024
) -> dict[str, COCO]:
    print("Splitting COCO...")

    image_ids = coco.getImgIds()
    labels = [image["caption"] for image in coco.loadImgs(image_ids)]

    # split into train and validation subset
    subsets = split_into_train_val(image_ids, val_ratio, seed, labels=labels)

    # create new COCO instances
    cocos = {}
    for name, image_ids in subsets.items():
        new_coco = COCO()
        new_coco.dataset = {
            k: v for k, v in coco.dataset.items() if k not in NOT_TO_COPY_KEYS
        }
        new_coco.dataset["images"] = coco.loadImgs(image_ids)
        new_coco.dataset["annotations"] = coco.loadAnns(
            coco.getAnnIds(imgIds=image_ids)
        )
        new_coco.createIndex()
        cocos[name] = new_coco
    return cocos


def post_process(
    root_dir: Union[str, Path],
    images_dir: str = "stylized",
    annotation_file: str = "annotations/foreground.json",
    images_out_dir: Optional[str] = None,
    annotation_out_dir: str = "annotations",
    val_ratio: float = 0.333,
    seed: int = 2024,
):
    root_path = Path(root_dir)
    images_path = root_path / images_dir

    # load coco annotation
    coco = COCO(root_path / annotation_file)

    # 1. update coco annotation
    coco = update_coco_annotation(coco, images_path)

    # 2. save the updated coco annotation
    out_annotations_path = root_path / annotation_out_dir
    out_annotations_path.mkdir(parents=True, exist_ok=True)
    with open(out_annotations_path / "all.json", "w") as f:
        json.dump(coco.dataset, f)

    # 3. split coco annotation
    cocos = split_coco_annotation(coco, val_ratio, seed=seed)

    # 4. save the splitted coco annotations
    for subset, coco in cocos.items():
        with open(out_annotations_path / f"{subset}.json", "w") as f:
            json.dump(coco.dataset, f)

    # 5. copy images if necessary
    if images_out_dir is not None:
        out_images_path = root_path / images_out_dir
        if out_images_path.exists():
            shutil.rmtree(out_images_path)
        out_images_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(images_path, out_images_path)


if __name__ == "__main__":
    args = vars(parse_args())
    post_process(**args)
