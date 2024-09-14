import argparse
import glob
import hashlib
import json
import shutil
from collections import defaultdict
from mimetypes import MimeTypes
from pathlib import Path

import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="A script for cleaning the classification dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dirs",
        type=str,
        nargs="+",
        help="root directory of datasets, if only one root dir, the script will clean it in in-place operation.",
    )
    parser.add_argument(
        "--duplicated-dir",
        type=str,
        default="duplicates",
        help="output directory to save duplicated images",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=0,
        help="specify dataset as the main dataset (compare & clean others)",
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


def clean_dataset(root_dir: Path, duplicated_dir: Path, dry_run: bool = False):
    if not dry_run:
        duplicated_dir.mkdir(parents=True, exist_ok=True)

    mime_checker = MimeTypes()
    image_hashes_to_files = defaultdict(list)
    number_of_images = 0
    number_of_duplicates = 0
    for file in tqdm.tqdm(
        sorted(glob.iglob(str(root_dir / "**/*.*"), recursive=True)), desc="Target"
    ):
        mime_type = mime_checker.guess_type(file)[0]
        if mime_type is None or "image" not in mime_type:
            continue

        with open(file, "rb") as f:
            image_hash = hashlib.sha512(f.read()).hexdigest()

        if image_hash in image_hashes_to_files:
            number_of_duplicates += 1
            if not dry_run:
                rel_path = Path(file).relative_to(root_dir).parent
                target_dir = duplicated_dir / root_dir.name / rel_path
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(file, target_dir)

        image_hashes_to_files[image_hash].append(file)
        number_of_images += 1
    return number_of_images, number_of_duplicates, image_hashes_to_files


def clean_datasets(
    root_dirs: list[Path], duplicated_dir: Path, target: int = 0, dry_run: bool = False
):
    root_dir = root_dirs.pop(target)
    if not dry_run:
        duplicated_dir.mkdir(parents=True, exist_ok=True)

    mime_checker = MimeTypes()
    image_hashes_to_files = defaultdict(list)
    number_of_images = 0
    for file in tqdm.tqdm(
        sorted(glob.iglob(str(root_dir / "**/*.*"), recursive=True)), desc="Target"
    ):
        mime_type = mime_checker.guess_type(file)[0]
        if mime_type is None or "image" not in mime_type:
            continue

        with open(file, "rb") as f:
            image_hash = hashlib.sha512(f.read()).hexdigest()

        image_hashes_to_files[image_hash].append(file)
        number_of_images += 1

    # clean other datasets
    number_of_duplicates = 0
    for root_dir in root_dirs:
        for file in tqdm.tqdm(
            sorted(glob.iglob(str(root_dir / "**/*.*"), recursive=True)), desc="Others"
        ):
            mime_type = mime_checker.guess_type(file)[0]
            if mime_type is None or "image" not in mime_type:
                continue

            with open(file, "rb") as f:
                image_hash = hashlib.sha512(f.read()).hexdigest()

            if image_hash in image_hashes_to_files:
                number_of_duplicates += 1
                if not dry_run:
                    rel_path = Path(file).relative_to(root_dir).parent
                    target_dir = duplicated_dir / root_dir.name / rel_path
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(file, target_dir)

            image_hashes_to_files[image_hash].append(file)
            number_of_images += 1
    return number_of_images, number_of_duplicates, image_hashes_to_files


if __name__ == "__main__":
    # TODO: Refactor this script
    args = parse_args()
    args.root_dirs = [Path(root_dir) for root_dir in args.root_dirs]
    args.duplicated_dir = Path(args.duplicated_dir)

    if len(args.root_dirs) == 1:
        number_of_images, number_of_duplicates, image_hashes_to_files = clean_dataset(
            args.root_dirs[0], args.duplicated_dir, args.dry_run
        )
    else:
        number_of_images, number_of_duplicates, image_hashes_to_files = clean_datasets(
            args.root_dirs, args.duplicated_dir, args.target, args.dry_run
        )

    print("Total images:", number_of_images)
    print("Total uniques:", number_of_images - number_of_duplicates)
    print("Total duplicates:", number_of_duplicates)

    if args.verbose:
        # find duplicated hashes
        image_hashes_to_files = dict(
            filter(lambda item: len(item[1]) > 1, image_hashes_to_files.items())
        )
        print(json.dumps(image_hashes_to_files, indent=4))
