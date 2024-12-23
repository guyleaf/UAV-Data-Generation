import argparse
import concurrent
import concurrent.futures
import itertools
import math
import os
import random
from collections import defaultdict
from functools import reduce
from typing import Literal, Union, get_args

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from rich import print
from rich.progress import track

from uav_data_generation.utils.dataset import split_into_train_val

# config: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
# retries: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/retries.html
AWS_CONFIG = Config(signature_version=UNSIGNED, retries={"mode": "standard"})
PAGE_SIZE = 10000

# label reference: https://www.boreas.utias.utoronto.ca/#/download
# clear -> sun, clouds, overcast, snow (the scene is captured after snowing) (exlude main weathers, e.g. rain, snowing)
WEATHER_SEQUENCES = {
    "clear": [
        "boreas-2020-12-18-13-44",
        "boreas-2021-01-15-12-17",
        "boreas-2021-02-09-12-55",
        "boreas-2021-03-02-13-38",
        "boreas-2021-03-09-14-23",
        "boreas-2021-03-30-14-23",
        "boreas-2021-04-08-12-44",
        "boreas-2021-04-13-14-49",
        "boreas-2021-05-06-13-19",
        "boreas-2021-05-13-16-11",
        "boreas-2021-06-03-16-00",
        "boreas-2021-06-17-17-52",
        "boreas-2021-06-29-20-43",
        "boreas-2021-08-05-13-34",
        "boreas-2021-09-02-11-42",
        "boreas-2021-09-07-09-35",
        "boreas-2021-09-09-15-28",
        "boreas-2021-11-02-11-16",
        "boreas-2021-11-23-14-27",
        "boreas-2021-01-19-15-08",
        "boreas-2021-04-15-18-55",
        "boreas-2021-04-20-14-11",
        "boreas-2021-07-27-14-43",
        "boreas-2021-10-15-12-35",
        "boreas-2021-10-22-11-36",
        "boreas-2021-11-16-14-10",
        "boreas-2020-11-26-13-58",
        "boreas-2020-12-04-14-00",
        "boreas-2021-02-02-14-07",
        "boreas-2021-03-23-12-43",
        "boreas-2021-10-05-15-35",
        "boreas-2021-11-14-09-47",
        # night
        # "boreas-2021-09-08-21-00",
        # "boreas-2021-09-14-20-00",
        # "boreas-2021-11-06-18-55",
    ],
    "snowing": [
        "boreas-2020-12-01-13-26",
        "boreas-2021-01-26-10-59",
        "boreas-2021-01-26-11-22",
        "boreas-2021-11-28-09-18",
        # "boreas-2021-04-22-15-00",
    ],
    "rain": [
        "boreas-2021-07-20-17-33",
        "boreas-2021-04-29-15-55",
        "boreas-2021-06-29-18-53",
        "boreas-2021-10-26-12-35",
    ],
}

# inverse weather sequences for better downsampling and splitting
SEQUENCE_WEATHERS = reduce(
    lambda acc, items: acc | {sequence: items[0] for sequence in items[1]},
    WEATHER_SEQUENCES.items(),
    dict[str, str](),
)

WEATHER_MAP = {
    "snowing": "snowy",
    "rain": "rainy",
    "clear": "clear",
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
        description="A script for downloading and preparing the Boreas dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "out_dir", type=str, help="output directory to write annotations and images"
    )
    parser.add_argument(
        "--s3-bucket", type=str, default="boreas", help="source S3 bucket name"
    )
    parser.add_argument(
        "--sample-frequency",
        type=int,
        default=2,
        help="sampling frequency of a sequence",
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
        "--seed", type=int, default=777, help="seed for random sampling"
    )
    parser.add_argument(
        "--all-in-one",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="combine the train, val subset into one",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="maximum number of workers to download files",
    )
    args = parser.parse_args()

    return args


def collect_sequences_from_s3(
    bucket_name: str,
    sequences: list[str] = list(SEQUENCE_WEATHERS.keys()),
    page_size: int = PAGE_SIZE,
):
    s3 = boto3.resource("s3", config=AWS_CONFIG)
    bucket = s3.Bucket(bucket_name)

    image_sequences = dict[str, list[str]]()
    for sequence in track(sequences, description="Collecting..."):
        query = bucket.objects.filter(Prefix=f"{sequence}/camera")
        query = query.page_size(page_size)
        images = (obj.key for obj in query)

        # the order of s3 bucker could be unordered
        image_sequences[sequence] = sorted(images)

    return image_sequences


def subsample_sequences(
    image_sequences: dict[str, list[str]],
    frequency: int,
    # second in microsecond
    timestamp_unit: int = 1000000,
):
    new_image_sequences = defaultdict[str, list[str]](list)
    for sequence_name, images in track(
        image_sequences.items(), description="Subsampling..."
    ):
        images = sorted(images)

        # according to the doc, the filename is the timestamp in microsecond format.
        # group by timestamp_unit
        group_images = defaultdict(list)
        for image in images:
            timestamp, _ = os.path.splitext(os.path.basename(image))
            group = int(timestamp) // timestamp_unit
            group_images[group].append(image)

        # subsample per second
        # for example, 16 images, 2 frequency
        # length between samples is 16 // 2 = 8
        # length of one side is (16 - (8 * (freq - 1)) - freq) // 2 => minus the gap and samples, divided by 2
        for images in group_images.values():
            total_length = len(images)
            gap_length = total_length // frequency
            side_length = (total_length - (frequency - 1) * gap_length - frequency) // 2
            images = list(
                images[i] for i in range(side_length, total_length, gap_length + 1)
            )
            assert len(images) == frequency

            new_image_sequences[sequence_name].extend(images)
    return new_image_sequences


def downsample_images_by_weather(
    image_sequences: dict[str, list[str]],
    key_weathers: list[str] = list(WEATHER_SEQUENCES.keys()),
    weather_map: dict[str, str] = WEATHER_MAP,
    max_samples: Union[int, SAMPLE_TYPES] = "avg",
):
    # map source's weathers into the target weathers first
    # the mapping could be many to one.
    weather_images = defaultdict[str, list[str]](list)
    for weather in track(key_weathers, description="Mapping weathers..."):
        sequences = WEATHER_SEQUENCES[weather]
        weather = weather_map[weather]

        # group by weather
        images = itertools.chain.from_iterable(
            image_sequences.get(sequence, []) for sequence in sequences
        )

        weather_images[weather].extend(images)

    # determine maximum samples per weather
    if max_samples == "avg":
        num_images = sum(len(images) for images in image_sequences.values())
        max_samples = math.ceil(num_images / len(weather_images))

    # shuffle & downsample
    image_weathers = dict[str, str]()
    for weather, images in track(weather_images.items(), description="Downsampling..."):
        random.shuffle(images)
        image_weathers.update(zip(images[:max_samples], itertools.repeat(weather)))

    return image_weathers


def download_image_from_s3(bucket_name: str, key: str, out_file: str):
    # thread-safe
    session = boto3.Session()
    s3 = session.resource("s3", config=AWS_CONFIG)
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(key, out_file)


def prepare_boreas_dataset(
    s3_bucket: str,
    out_dir: str,
    sample_frequency: int = 2,
    max_samples: Union[int, SAMPLE_TYPES] = "avg",
    val_ratio: float = 0.1,
    seed: int = 777,
    all_in_one: bool = False,
    max_workers: int = 5,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # collect images
    image_sequences = collect_sequences_from_s3(s3_bucket)

    # subsample sequences
    image_sequences = subsample_sequences(image_sequences, sample_frequency)

    random.seed(seed)

    # downsample images by weather
    image_weathers = downsample_images_by_weather(
        image_sequences, max_samples=max_samples
    )

    # split images if not all_in_one
    if all_in_one:
        subset_image_weathers = {"": image_weathers}
    else:
        subset_image_weathers = split_into_train_val(
            list(image_weathers.items()),
            val_ratio,
            seed + 2024,
            labels=list(image_weathers.values()),
        )
        subset_image_weathers = {
            subset: dict(image_weathers)
            for subset, image_weathers in subset_image_weathers.items()
        }

    # download images to the corresponding folder
    # TODO: add progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exec:
        futures = []
        for subset, image_weathers in subset_image_weathers.items():
            subset_dir = os.path.join(out_dir, subset)
            counter = defaultdict(int)
            for image, weather in image_weathers.items():
                sequence = image.split("/", 1)[0]
                out_file = os.path.join(subset_dir, weather)
                os.makedirs(out_file, exist_ok=True)
                out_file = os.path.join(
                    out_file, f"{sequence}_" + os.path.basename(image)
                )

                futures.append(
                    exec.submit(
                        download_image_from_s3,
                        s3_bucket,
                        image,
                        out_file,
                    )
                )
                counter[weather] += 1

            print(subset.capitalize())
            for weather, count in counter.items():
                print(f"{weather}:", count)

        print("[yellow]Downloading...")

        try:
            dones, pendings = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_EXCEPTION
            )
            # cancel all pending tasks if raising exception
            for pending in pendings:
                pending.cancel()

            for done in dones:
                ex = done.exception()
                if ex is not None:
                    raise RuntimeError from ex
        except KeyboardInterrupt:
            exec.shutdown(cancel_futures=True)
            raise

    print("[green]Done!")


if __name__ == "__main__":
    args = vars(parse_args())
    prepare_boreas_dataset(**args)
