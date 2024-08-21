import argparse
import itertools
import warnings
from collections import defaultdict
from pathlib import Path

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from matplotlib import font_manager
from pycocotools.coco import COCO

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
        help="annotations folder of tthe COCO dataset",
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
    parser.add_argument(
        "--types-of-label",
        type=str,
        nargs="*",
        default=[],
        help="validate these extra annotations. Note, the folder structure should follow the format of COCO dataset with labels",
    )
    parser.add_argument(
        "--votes", type=str, nargs="+", default=["major_vote", "weighted_vote"]
    )
    args = parser.parse_args()

    return args


# check if ids are unique
def _validate_ids(coco: COCO):
    assert len(coco.imgs) == len(coco.dataset["images"])
    assert len(coco.anns) == len(coco.dataset["annotations"])


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
    types_of_label: list[str] = [],
    votes: list[str] = ["major_vote", "weighted_vote"],
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

        if show:
            _show_annotations(coco, images_path, num_annotations=show_num_images)

    # check extra annotation files (e.g. weather)
    for type_of_label in types_of_label:
        type_of_label_dir = root_dir / type_of_label
        if not type_of_label_dir.exists():
            raise RuntimeError(
                f"The type of label folder {type_of_label_dir.name} is not found."
            )

        for vote in votes:
            root_annotation_dir = type_of_label_dir / vote
            if not root_annotation_dir.exists():
                warnings.warn(
                    f"The vote of folder {root_annotation_dir.name} is not found."
                )
                continue
            is_major_vote = vote == "major_vote"

            print(f"Checking {root_annotation_dir}...")

            annotation_dirs = filter(
                lambda label_dir: label_dir.name != "all",
                root_annotation_dir.glob("*"),
            )
            annotation_files = list(
                itertools.chain.from_iterable(
                    annotation_dir.glob(ANNOTATION_SEARCH_PATTERN)
                    for annotation_dir in annotation_dirs
                )
            )

            print(f"Found {len(annotation_files)} annotation files.")

            subset_image_ids = defaultdict(list)
            subset_annotation_ids = defaultdict(list)
            for annotation_file in annotation_files:
                coco = COCO(annotation_file)
                _validate_ids(coco)
                if is_major_vote:
                    subset_image_ids[annotation_file.stem].extend(coco.imgs.keys())
                    subset_annotation_ids[annotation_file.stem].extend(coco.anns.keys())

                if show:
                    _show_annotations(
                        coco, images_path, num_annotations=show_num_images
                    )

            # by default, the annotation files in "all" folder must be matched with its split annotation files
            if is_major_vote:
                all_annotation_files = (type_of_label_dir / "all").glob(
                    ANNOTATION_SEARCH_PATTERN
                )
                for annotation_file in all_annotation_files:
                    coco = COCO(annotation_file)
                    _validate_ids(coco)
                    assert sorted(coco.imgs.keys()) == sorted(
                        subset_image_ids[annotation_file.stem]
                    )
                    assert sorted(coco.anns.keys()) == sorted(
                        subset_annotation_ids[annotation_file.stem]
                    )

    print("No errors!")


if __name__ == "__main__":
    args = vars(parse_args())
    print(args)
    validate_coco_dataset(Path(args.pop("root_dir")), **args)
