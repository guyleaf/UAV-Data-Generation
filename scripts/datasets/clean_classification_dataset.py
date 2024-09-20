import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from rich import print
from rich.progress import track

from uav_data_generation.utils.io import collect_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="A script for cleaning the classification dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dirs",
        type=str,
        nargs="+",
        help="root directories of datasets.",
    )
    parser.add_argument(
        "--duplicated-dir",
        type=str,
        default="duplicates",
        help="output directory to save duplicated images",
    )
    parser.add_argument(
        "--main",
        type=int,
        default=0,
        help="specify dataset as the main dataset (compare & clean others and duplicates in main).",
    )
    parser.add_argument(
        "--dry-run",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="execute without cleaning",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="show the duplicated hashes and files",
    )
    args = parser.parse_args()

    return args


def collect_image_hashes(root_dir: Path):
    image_hashes = defaultdict[str, list[Path]](list)
    for file in track(
        collect_images(root_dir), description=f"Collecting {root_dir}..."
    ):
        with open(file, "rb") as f:
            image_hash = hashlib.sha512(f.read()).hexdigest()
        image_hashes[image_hash].append(file)
    return image_hashes


def clean_images(
    dataset_to_image_hashes: dict[Path, defaultdict[str, list[Path]]],
    main_root_dir: Path,
    duplicated_dir: Path,
):
    def _move(root_dir: Path, files: list[Path], target: Path):
        for file in files:
            rel_path = file.relative_to(root_dir).parent
            target_dir = target / root_dir.name / rel_path
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(file, target_dir)

    main_image_hashes = dataset_to_image_hashes.pop(main_root_dir)
    for main_image_hash, file_paths in track(
        main_image_hashes.items(), description="Cleaning..."
    ):
        # remove duplicates in main
        _move(main_root_dir, file_paths[1:], duplicated_dir)

        # remove duplicates in other datasets with respect to main
        for root_dir, image_hashes in dataset_to_image_hashes.items():
            file_paths = image_hashes.get(main_image_hash, [])
            _move(root_dir, file_paths, duplicated_dir)


if __name__ == "__main__":
    # TODO: Refactor this script
    args = parse_args()
    root_dirs = [Path(root_dir) for root_dir in args.root_dirs]
    duplicated_dir = Path(args.duplicated_dir)

    dataset_to_image_hashes = {
        root_dir: collect_image_hashes(root_dir) for root_dir in root_dirs
    }

    number_of_images = sum(
        sum(len(files) for files in image_hashes.values())
        for image_hashes in dataset_to_image_hashes.values()
    )

    main_image_hashes = dataset_to_image_hashes[root_dirs[args.main]]
    number_of_uniques = len(main_image_hashes)
    number_of_uniques += sum(
        len(set(image_hashes) - set(main_image_hashes))
        for image_hashes in dataset_to_image_hashes.values()
    )
    print("Total images:", number_of_images)
    print("Total uniques:", number_of_uniques)
    print("Total duplicates:", number_of_images - number_of_uniques)

    if args.verbose:
        # find duplicated hashes
        for root_dir, image_hashes in dataset_to_image_hashes.items():
            image_hashes = dict(
                filter(lambda item: len(item[1]) > 1, image_hashes.items())
            )
            print("Dataset:", root_dir)
            print(json.dumps(image_hashes, indent=4))
            print()

    if not args.dry_run:
        clean_images(dataset_to_image_hashes, root_dirs[args.main], duplicated_dir)
