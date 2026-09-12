import argparse
from pathlib import Path

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from matplotlib import font_manager
from pycocotools.coco import COCO
from rich.progress import track

ANNOTATION_SEARCH_PATTERN = "*.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate COCO dataset (annotations)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root_dir", type=str, help="root folder of the COCO dataset")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="images folder of tthe COCO dataset",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="annotations",
        help="annotations folder of the COCO dataset",
    )
    parser.add_argument(
        "--show",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="show the annotation or not",
    )
    parser.add_argument(
        "--show-num-images", type=int, default=5, help="number of images to be shown"
    )
    args = parser.parse_args()

    return args


# check if ids are unique
def _validate_ids(coco: COCO):
    assert len(coco.imgs) == len(coco.dataset["images"])
    assert len(coco.anns) == len(coco.dataset["annotations"])


def _validate_annotation_areas(coco: COCO):
    assert all(
        (ann["bbox"][2] * ann["bbox"][3]) == ann["area"] and ann["area"] > 0
        for ann in coco.dataset["annotations"]
    )


def _validate_imgs(coco: COCO, images_path: Path):
    for img in track(coco.dataset["images"], description="Checking images..."):
        img_shape = (img["width"], img["height"])
        with Image.open(images_path / img["file_name"]) as f:
            assert img_shape == f.size


def _draw_bounding_box(image: Image.Image, coord: tuple[int, int, int, int]):
    x, y, w, h = coord

    draw = ImageDraw.Draw(image)
    draw.rectangle([x, y, x + w, y + h], outline="red")

    text = f"Object size: {w * h:,}"
    font_file = font_manager.findfont("arial")
    font = ImageFont.truetype(font_file, 36)
    text_xy = [x, y]
    text_coord = text_xy + list(draw.textbbox(text_xy, text=text, font=font)[2:])

    draw.rectangle(text_coord, fill="red")
    draw.text(text_coord[:2], text=text, font=font, fill="white")


def _show_annotations(coco: COCO, images_path: Path, num_annotations: int = 5):
    ann_ids = coco.getAnnIds()[:num_annotations]
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        image_name = coco.loadImgs(ann["image_id"])[0]["file_name"]
        image = Image.open(images_path / image_name)
        _draw_bounding_box(image, ann["bbox"])
        image.show(title=image_name)


def validate_coco_dataset(
    root_dir: Path,
    images_dir: str = "images",
    annotations_dir: str = "annotations",
    show: bool = False,
    show_num_images: int = 5,
):
    if not root_dir.exists():
        raise RuntimeError(f"The root folder {root_dir.name} is not found.")
    images_path = root_dir / images_dir
    annotations_path = root_dir / annotations_dir

    # check main annotation files
    annotation_files = annotations_path.glob(ANNOTATION_SEARCH_PATTERN)
    for annotation_file in annotation_files:
        coco = COCO(annotation_file)
        _validate_ids(coco)
        _validate_annotation_areas(coco)
        _validate_imgs(coco, images_path)
        if show:
            _show_annotations(coco, images_path, num_annotations=show_num_images)

    print("No errors!")


if __name__ == "__main__":
    args = vars(parse_args())
    print(args)
    validate_coco_dataset(Path(args.pop("root_dir")), **args)
