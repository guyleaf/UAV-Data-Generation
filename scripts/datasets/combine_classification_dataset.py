import argparse
import os
import shutil

from rich import print
from rich.progress import track


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine multiple classification datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dirs",
        type=str,
        nargs="+",
        help="Root directories of classificaton datasets",
    )
    parser.add_argument(
        "out_dir", type=str, help="Output directory of combined dataset"
    )
    parser.add_argument(
        "--all-in-one",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether datasets are all in one (no subsets, such as train/val/test...)",
    )
    args = parser.parse_args()

    return args


def create_subset_dir(subset_dir: str, labels: list[str]):
    os.makedirs(subset_dir, exist_ok=True)
    for label in labels:
        label_dir = os.path.join(subset_dir, label)
        os.makedirs(label_dir, exist_ok=True)


def combine_datasets(
    root_dirs: list[str], out_dir: str, all_in_one: bool = False
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # collect subsets
    if all_in_one:
        subsets = [""]
    else:
        # 2024.09.14 better reproducibility
        subsets = sorted(os.listdir(root_dirs[0]))

    # copy images to the corresponding folder
    for subset in subsets:
        counter = 0
        for root_dir in root_dirs:
            subset_dir = os.path.join(root_dir, subset)
            out_subset_dir = os.path.join(out_dir, subset)
            if not os.path.exists(subset_dir):
                continue

            # 2024.09.14 better reproducibility
            labels = sorted(os.listdir(subset_dir))
            create_subset_dir(out_subset_dir, labels)

            for label in labels:
                label_dir = os.path.join(subset_dir, label)
                out_label_dir = os.path.join(out_subset_dir, label)

                # 2024.09.14 better reproducibility
                images = sorted(os.listdir(label_dir))
                for image in track(
                    images, description=f"{subset} - {root_dir} - {label} images"
                ):
                    extension = image.split(".")[1]
                    # rename image to %5d.%ext format
                    out_image = f"{counter}".zfill(5) + f".{extension}"

                    image_path = os.path.join(label_dir, image)
                    out_image_path = os.path.join(out_label_dir, out_image)

                    shutil.copy2(image_path, out_image_path)
                    counter += 1

        print(subset.capitalize())
        print("Total:", counter)


if __name__ == "__main__":
    args = parse_args()
    combine_datasets(
        args.root_dirs,
        args.out_dir,
        all_in_one=args.all_in_one,
    )
