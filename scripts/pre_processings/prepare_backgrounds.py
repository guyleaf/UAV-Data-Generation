import argparse
import json
import math
import os
import random
import shutil
from collections import defaultdict
from mimetypes import MimeTypes
from pathlib import Path
from typing import Literal, Union, get_args

import PIL.Image as Image
from rich import print
from rich.progress import track
from rich.prompt import Confirm

SAMPLE_TYPES = Literal["avg"]


def to_num_samples(value: str) -> Union[int, SAMPLE_TYPES]:
    try:
        return int(value)
    except ValueError:
        if value not in get_args(SAMPLE_TYPES):
            raise
        return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="A preparation script for backgrounds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dirs", type=str, nargs="+", help="root folders of classification datasets"
    )
    parser.add_argument("out_dir", type=str, help="root folder of the output dataset")

    parser.add_argument(
        "--dataset-names",
        type=str,
        nargs="+",
        default=None,
        help="dataset names of root folders, used for naming in the backgrounds folder. use the basename of the root_dir by default.",
    )
    parser.add_argument(
        "--max-samples",
        type=to_num_samples,
        default="avg",
        help="maximum number of samples per category (weathers)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=["clear", "cloudy"],
        help="collect background images from labels",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        nargs=2,
        default=(1920, 1920),
        help="maximum resolution of background images, (width, height)",
    )

    parser.add_argument(
        "--seed", type=int, default=777, help="seed for random sampling"
    )
    parser.add_argument(
        "--dry-run",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="execute without preprocessing",
    )
    args = parser.parse_args()

    assert all(os.path.isdir(root_dir) for root_dir in args.root_dirs), (
        "Not all root folders exists."
    )

    if args.dataset_names is None:
        args.dataset_names = [os.path.basename(root_dir) for root_dir in args.root_dirs]
    else:
        assert len(args.dataset_names) == len(args.root_dirs), (
            "The number of root dirs and dataset names should be the same."
        )

    assert args.max_resolution[0] > 0 and args.max_resolution[1] > 0, (
        "The maximum resolution should be larger than 0."
    )
    args.max_resolution = tuple(args.max_resolution)

    return args


def collect_images(root_path: Path):
    mime_checker = MimeTypes()

    def validate_file_type(path: Path):
        mime_type = mime_checker.guess_type(path)[0]
        return mime_type is not None and "image" in mime_type

    return sorted(
        filter(
            validate_file_type,
            root_path.rglob("*.*"),
        )
    )


def prepare_backgrounds(
    datasets: dict[str, Path],
    out_dir: Path,
    max_samples: Union[int, SAMPLE_TYPES] = "avg",
    labels: list[str] = ["clear", "cloudy"],
    max_resolution: tuple[int, int] = (1920, 1920),
    seed: int = 777,
    dry_run: bool = False,
) -> None:
    # out_dir = out_dir / "backgrounds"
    if not dry_run:
        if out_dir.exists():
            print("[bold green]Backgrounds folder exists![/bold green]")
            answer = Confirm.ask(
                "Do you want to continue? ([bold red]will be deleted before processing[/bold red])",
                default=False,
            )
            if not answer:
                exit()

            shutil.rmtree(out_dir)

        out_dir.mkdir(parents=True, exist_ok=True)

    label_to_images = defaultdict[str, list[tuple[str, Path]]](list)
    for name, root_dir in track(datasets.items(), description="Collecting..."):
        for label in labels:
            label_to_images[label].extend(
                (name, image) for image in collect_images(root_dir / label)
            )

    if max_samples == "avg":
        max_samples = sum(len(images) for images in label_to_images.values()) / len(
            label_to_images
        )
        max_samples = math.ceil(max_samples)
        print("Average samples:", max_samples)

    # set the random seed
    random.seed(seed)

    # downsample images per category
    label_to_selected_images = dict[str, list[tuple[str, Path]]]()
    for label, images in track(label_to_images.items(), description="Down-sampling..."):
        if len(images) > max_samples:
            # random.shuffle(images)
            images = random.sample(images, max_samples)

        label_to_selected_images[label] = images

    # construct backgrounds
    if not dry_run:
        for label, images in label_to_selected_images.items():
            for name, image in track(
                images, description=f"Processing {label} images..."
            ):
                label_dir = out_dir / name / label
                label_dir.mkdir(parents=True, exist_ok=True)

                data = Image.open(image)
                if data.size[0] > max_resolution[0] or data.size[1] > max_resolution[1]:
                    data.thumbnail(max_resolution, resample=Image.LANCZOS)
                    image = image.with_suffix(".png")
                    data.save(
                        label_dir / image.name,
                        icc_profile=data.info.get("icc_profile"),
                        exif=data.info.get("exif"),
                    )
                else:
                    shutil.copy2(image, label_dir / image.name)

    print("[bold blue]Total[/bold blue] images:")
    label_to_counter = {
        label: len(images) for label, images in label_to_selected_images.items()
    }
    print(json.dumps(label_to_counter, indent=4))


if __name__ == "__main__":
    args = parse_args()
    datasets = {
        name: Path(path) for name, path in zip(args.dataset_names, args.root_dirs)
    }
    out_dir = Path(args.out_dir)

    prepare_backgrounds(
        datasets,
        out_dir,
        max_samples=args.max_samples,
        labels=args.labels,
        max_resolution=args.max_resolution,
        seed=args.seed,
        dry_run=args.dry_run,
    )
