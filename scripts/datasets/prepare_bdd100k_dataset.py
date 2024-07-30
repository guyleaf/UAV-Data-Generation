import argparse
import json
import shutil
from pathlib import Path

import tqdm

WEATHER_MAP = {
    "rainy": "rainy",
    "snowy": "snowy",
    "clear": "clear",
    "overcast": "cloudy",
    "partly cloudy": "cloudy",
    "foggy": "foggy",
}

DEFAULT_WEATHERS = [
    "rainy",
    "snowy",
    "clear",
    "overcast",
    "partly cloudy",
    "foggy",
]
DEFAULT_TIMES = ["daytime", "night", "dawn/dusk"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="A script for preparing the BDD100K dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dir", type=str, help="root directory to read annotations and images"
    )
    parser.add_argument(
        "out_dir", type=str, help="output directory to write annotations and images"
    )
    parser.add_argument(
        "--weathers",
        type=str,
        nargs="+",
        default=DEFAULT_WEATHERS,
        help="processing these weathers",
    )
    parser.add_argument(
        "--times",
        type=str,
        nargs="+",
        default=DEFAULT_TIMES,
        help="process these times of day",
    )
    parser.add_argument(
        "--all-in-one",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="combine the train, val subset into one",
    )
    parser.add_argument(
        "--legacy",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="use legacy labels",
    )
    args = parser.parse_args()

    return args


def collect_attributes(
    labels: list[dict], names: list[str], keywordss: list[set[str]]
) -> dict[str, dict[str, str]]:
    length_of_names = len(names)
    result = {}
    for label in tqdm.tqdm(labels, desc="Collecting attribute"):
        tmp = {}
        for name, keywords in zip(names, keywordss):
            value = label["attributes"][name]
            if value not in keywords:
                break
            tmp[name] = value

        if len(tmp.keys()) == length_of_names:
            result[label["name"]] = tmp
    return result


def prepare_bdd100k_dataset(
    root_dir: Path,
    out_dir: Path,
    weathers: set[str] = set(DEFAULT_WEATHERS),
    times: set[str] = set(DEFAULT_TIMES),
    all_in_one: bool = False,
    legacy: bool = False,
):
    subsets = ["train", "val"]
    if not all_in_one:
        subsets.append("test")

    for subset in subsets:
        subset_root_dir = root_dir / "images" / "100k" / subset
        subset_out_dir = out_dir
        if not all_in_one:
            subset_out_dir /= subset

        for weather in WEATHER_MAP.values():
            (subset_out_dir / weather).mkdir(parents=True, exist_ok=True)

        if legacy:
            label_file = root_dir / "labels" / f"bdd100k_labels_images_{subset}.json"
        else:
            label_file = root_dir / "labels" / "det_20" / f"det_{subset}.json"
        if label_file.exists():
            with open(label_file, "r") as f:
                labels = json.load(f)

            labels = collect_attributes(
                labels, ["weather", "timeofday"], [weathers, times]
            )
            for image_name, label in tqdm.tqdm(
                labels.items(), desc=f"{subset.capitalize()} image"
            ):
                weather = WEATHER_MAP[label["weather"]]

                image_file = subset_root_dir / image_name
                weather_out_dir = subset_out_dir / weather
                shutil.copy2(image_file, weather_out_dir)
        else:
            for image_file in tqdm.tqdm(
                subset_root_dir.iterdir(), desc=f"{subset.capitalize()} image"
            ):
                subset_out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_file, subset_out_dir)


if __name__ == "__main__":
    args = parse_args()
    prepare_bdd100k_dataset(
        Path(args.root_dir),
        Path(args.out_dir),
        set(args.weathers),
        set(args.times),
        all_in_one=args.all_in_one,
        legacy=args.legacy,
    )
