import argparse
import os
import shutil
from mimetypes import MimeTypes

import PIL.Image as Image
import tqdm
from utils import split_into_train_val

WEATHER_MAP = {
    "rainy": "rainy",
    "snowy": "snowy",
    "sunny": "clear",
    "cloudy": "cloudy",
    "foggy": "foggy",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="A script for preparing the FWID dataset",
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
    parser.add_argument(
        "--all-in-one",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="combine the train, val subset into one",
    )
    args = parser.parse_args()

    return args


def prepare_fwid_dataset(
    root_dir: str,
    out_dir: str,
    val_ratio: float = 0.1,
    seed: int = 777,
    all_in_one: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # collect labels
    labels = os.listdir(root_dir)

    mime_checker = MimeTypes()
    image_files = {}
    for label in labels:
        # collect images
        folder = os.path.join(root_dir, label)
        for image in sorted(os.listdir(folder)):
            mime_type = mime_checker.guess_type(image)[0]
            if mime_type is None or "image" not in mime_type:
                continue

            image = os.path.join(folder, image)
            image_files[image] = WEATHER_MAP[label]

    labels = [WEATHER_MAP[label] for label in labels]

    if all_in_one:
        subsets = {"": list(image_files.keys())}
    else:
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
        for image in tqdm.tqdm(images, desc=f"{subset.capitalize()} images"):
            image_name = os.path.basename(image)
            image_name_without_ext = os.path.splitext(image_name)[0]

            label = image_files[image]
            target_dir = os.path.join(subset_dir, label)

            # some images are not in jpg format but extension is .jpg or jfif_version is wrong
            # save it to png format (rewrite the image to avoid 'libpng warning: sBIT: invalid' or 'Warning: unknown JFIF revision number 32.23')
            with Image.open(image) as img:
                if img.format == "JPEG" and img.info.get("jfif_version") != (32, 23):
                    shutil.copy2(image, target_dir)
                else:
                    img.save(
                        os.path.join(target_dir, f"{image_name_without_ext}.png"),
                        icc_profile=img.info.get("icc_profile"),
                        exif=img.info.get("exif"),
                    )

            counter[label] += 1

        print(subset.capitalize())
        for label, count in counter.items():
            print(f"{label}:", count)


if __name__ == "__main__":
    args = parse_args()
    prepare_fwid_dataset(
        args.root_dir,
        args.out_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        all_in_one=args.all_in_one,
    )
