import argparse
import os
import shutil

from rich import print
from rich.progress import track

from uav_data_generation.utils.dataset import split_into_train_val
from uav_data_generation.utils.io import collect_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="A script for splitting the classification dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dir", type=str, help="root directory to read annotations and images"
    )
    parser.add_argument(
        "out_dir", type=str, help="output directory to write annotations and images"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="val dataset ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=777, help="the seed of random generator"
    )
    args = parser.parse_args()

    return args


def split_classification_dataset(
    root_dir: str,
    out_dir: str,
    val_ratio: float = 0.1,
    seed: int = 777,
):
    os.makedirs(out_dir, exist_ok=True)

    labels = sorted(os.listdir(root_dir))

    image_files = {}
    for label in labels:
        # collect images
        folder = os.path.join(root_dir, label)
        for image in collect_images(folder):
            image_files[image] = label

    # split the data into train and test set
    subsets = split_into_train_val(
        list(image_files.keys()),
        val_ratio,
        seed,
        labels=list(image_files.values()),
    )

    # copy images to the corresponding folder
    for subset, images in subsets.items():
        subset_dir = os.path.join(out_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)
        for label in labels:
            label_dir = os.path.join(subset_dir, label)
            os.makedirs(label_dir, exist_ok=True)

        counter = {label: 0 for label in labels}
        for image in track(images, description=f"{subset.capitalize()} images"):
            label = image_files[image]
            shutil.copy2(image, os.path.join(subset_dir, label))
            counter[label] += 1

        print(subset.capitalize())
        for label, count in counter.items():
            print(f"{label}:", count)


if __name__ == "__main__":
    args = vars(parse_args())
    split_classification_dataset(**args)
