import argparse
import itertools
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Literal, Union, get_args

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from rich import print
from rich.progress import track

WEATHER_MAP = {
    "rain": "rainy",
    # rests are in clear
    "": "clear",
}

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
        description="A script for preparing the Image2Weather dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dir", type=str, help="root directory to read annotations and images"
    )
    parser.add_argument(
        "out_dir", type=str, help="output directory to write annotations and images"
    )
    parser.add_argument(
        "--version", type=str, default="v1.0-trainval", help="the version of nuScenes"
    )
    parser.add_argument(
        "--max-samples",
        type=to_num_samples,
        default="avg",
        help="maximum number of samples per category (weathers)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="val dataset ratio"
    )
    parser.add_argument(
        "--all-in-one",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="combine the train, val subset into one",
    )
    parser.add_argument(
        "--night",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="include night?",
    )
    parser.add_argument(
        "--seed", type=int, default=777, help="seed for random sampling"
    )
    args = parser.parse_args()

    return args


def collect_images_from_scenes(
    nusc: NuScenes, labels: list[str], excluded_labels: list[str] = []
):
    val_scenes = set(create_splits_scenes()["val"])

    subsets = defaultdict[str, list](list)
    image_to_labels = {}
    for label in labels:
        # find scenes belonging to the label
        scenes = filter(lambda scene: label in scene["description"].lower(), nusc.scene)
        scenes = filter(
            lambda scene: not any(
                excluded_label in scene["description"].lower()
                for excluded_label in excluded_labels
            ),
            scenes,
        )
        mapped_label = WEATHER_MAP[label]

        # collect images
        # schema: https://www.nuscenes.org/public/images/nuscenes-schema.svg
        for scene in scenes:
            split = "val" if scene["name"] in val_scenes else "train"
            token = scene["first_sample_token"]
            last_token = scene["last_sample_token"]

            while len(token) != 0:
                sample = nusc.get("sample", token)

                # collect multi-views
                for sensor, data_token in sample["data"].items():
                    if not sensor.startswith("CAM_"):
                        continue

                    data = nusc.get("sample_data", data_token)
                    image = nusc.dataroot / Path(data["filename"])
                    image_to_labels[image] = mapped_label
                    subsets[split].append(image)

                # point to the next token if exists
                if token == last_token:
                    break
                token = sample["next"]

        # exclude previous label to avoid overlap
        excluded_labels.append(label)

    return image_to_labels, subsets


def prepare_nuscenes_dataset(
    root_dir: str,
    out_dir: str,
    version: str = "v1.0-trainval",
    max_samples: Union[int, SAMPLE_TYPES] = "avg",
    val_ratio: float = 0.1,
    all_in_one: bool = False,
    night: bool = False,
    seed: int = 777,
) -> None:
    # load nuScenes dataset
    nusc = NuScenes(version=version, dataroot=root_dir, verbose=True)

    os.makedirs(out_dir, exist_ok=True)

    # collect images
    # keep the insertion order of WEATHER_MAP
    labels = list(WEATHER_MAP.keys())
    excluded_labels = []
    if not night:
        excluded_labels.append("night")
    image_to_labels, subsets = collect_images_from_scenes(
        nusc, labels, excluded_labels=excluded_labels
    )
    # only two subsets (train, val)
    assert len(subsets) == 2

    if all_in_one:
        subsets = {"": list(itertools.chain.from_iterable(subsets.values()))}

    # calculate max_samples for each subset
    # keep the insertion order of WEATHER_MAP
    labels = list(WEATHER_MAP.values())
    if max_samples == "avg":
        max_samples = math.ceil(len(image_to_labels) / len(labels))
    val_max_samples = math.ceil(max_samples * val_ratio)
    subset_to_max_samples = dict(
        train=max_samples - val_max_samples, val=val_max_samples
    )
    print("Max samples:", subset_to_max_samples)

    random.seed(seed)
    for subset, images in subsets.items():
        max_samples = subset_to_max_samples[subset]
        subset_dir = os.path.join(out_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        for label in labels:
            label_dir = os.path.join(subset_dir, label)
            os.makedirs(label_dir, exist_ok=True)

        # reverse indexing
        label_to_images = defaultdict[str, list](list)
        for image in track(
            images, description=f"{subset.capitalize()} images - reverse indexing"
        ):
            label = image_to_labels[image]
            label_to_images[label].append(image)

        # copy images to the corresponding folder
        for label, images in label_to_images.items():
            # down-sample images
            if len(images) > max_samples:
                random.shuffle(images)
                images = images[:max_samples]

            for image in track(
                images,
                description=f"{subset.capitalize()} images - copying {label}",
            ):
                shutil.copy2(image, os.path.join(subset_dir, label))

        print(subset.capitalize())
        for label in labels:
            print(f"{label}:", min(len(label_to_images[label]), max_samples))


if __name__ == "__main__":
    args = vars(parse_args())
    prepare_nuscenes_dataset(**args)
