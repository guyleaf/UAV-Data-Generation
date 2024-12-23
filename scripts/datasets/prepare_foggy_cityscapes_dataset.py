import argparse
import itertools
import os
import shutil
from collections import ChainMap
from pathlib import Path

from rich import print
from rich.progress import track


def parse_args():
    parser = argparse.ArgumentParser(
        description="A script for preparing the Foggy Zurich dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dir", type=str, help="root directory to read annotations and images"
    )
    parser.add_argument(
        "out_dir", type=str, help="output directory to write annotations and images"
    )
    parser.add_argument(
        "--all-in-one",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="combine the train, val subset into one",
    )
    args = parser.parse_args()

    return args


def collect_images_from_list(root_dir: str, file_list: str):
    root_dir_ = Path(root_dir)
    with open(file_list) as f:
        for path in iter(f.readline, ""):
            path = path.strip()
            yield root_dir_ / path


def prepare_foggy_zurich_dataset(
    root_dir: str, out_dir: str, all_in_one: bool = False
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    file_list_dir = os.path.join(root_dir, "lists_file_names")
    train_image_files = dict[Path, str]()
    val_image_files = dict[Path, str]()

    # collect train images
    for strength in FOGGY_STRENGTH:
        file_list = os.path.join(file_list_dir, f"RGB_{strength}_filenames.txt")
        for image in collect_images_from_list(root_dir, file_list):
            train_image_files[image] = f"foggy_{strength}"

    # collect val(test) images
    file_list = os.path.join(file_list_dir, "RGB_testv2_filenames.txt")
    for image in collect_images_from_list(root_dir, file_list):
        val_image_files[image] = "foggy"

    if all_in_one:
        subsets = {
            "": list(itertools.chain(train_image_files.keys(), val_image_files.keys()))
        }
    else:
        subsets = {
            "train": list(train_image_files.keys()),
            "val": list(val_image_files.keys()),
        }

    # copy images to the corresponding folder
    image_files = ChainMap(train_image_files, val_image_files)
    for subset, images in subsets.items():
        subset_dir = os.path.join(out_dir, subset)

        for image in track(images, description=f"{subset.capitalize()} images"):
            label = image_files[image]
            out_dir_ = os.path.join(subset_dir, label)
            os.makedirs(out_dir_, exist_ok=True)

            shutil.copy2(image, out_dir_)
            # combine all into foggy
            if label != "foggy":
                out_dir_ = os.path.join(subset_dir, "foggy")
                os.makedirs(out_dir_, exist_ok=True)
                shutil.copy2(image, out_dir_)

        print(f"{subset.capitalize()} samples:", len(images))


if __name__ == "__main__":
    args = vars(parse_args())
    prepare_foggy_zurich_dataset(**args)
