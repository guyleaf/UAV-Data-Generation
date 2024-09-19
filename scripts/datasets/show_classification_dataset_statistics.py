import argparse
import os
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter, ScalarFormatter
from rich import print
from utils import collect_images

"""expected structure of root folder
root
 - train
    - label_1
      - image
    - label_2
 - val
 - ...
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="A visualization script for show statistics of the classification dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dir", type=str, help="root path of the classification dataset"
    )
    parser.add_argument(
        "--title", type=str, default=None, help="the title of the figure"
    )
    parser.add_argument(
        "--all-in-one",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="the dataset is all in one.",
    )
    parser.add_argument(
        "--excluded-subsets",
        type=str,
        nargs="+",
        default=[],
        help="don't count in these subsets",
    )
    args = parser.parse_args()

    return args


def analyze_images(
    root_dir: str,
    all_in_one: bool = False,
    excluded_subsets: list[str] = [],
):
    if all_in_one:
        subsets = [""]
    else:
        subsets = [
            subset for subset in os.listdir(root_dir) if subset not in excluded_subsets
        ]
        subsets = sorted(subsets)

    labels = {"all"}
    for subset in subsets:
        subset_dir = os.path.join(root_dir, subset)
        labels.update(os.listdir(subset_dir))
    labels = sorted(labels)

    areas = []
    label_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for subset in subsets:
        subset_dir = os.path.join(root_dir, subset)

        for label in labels:
            label_dir = os.path.join(subset_dir, label)
            if os.path.isdir(label_dir):
                images = collect_images(label_dir)

                # find max image size
                for image in images:
                    with Image.open(image) as im:
                        area = im.size[0] * im.size[1]
                        areas.append(area)

                count = len(images)
            else:
                count = 0

            label_counts[subset][label] = count

        count = sum(label_counts[subset].values())
        label_counts[subset]["all"] = count

    return labels, label_counts, areas


def make_label_dist_plot(
    axes: Axes,
    labels: list[str],
    label_counts: dict[str, dict[str, int]],
    all_in_one: bool = False,
):
    width = 0.2  # the width of the bars

    for multiplier, (subset, counts) in enumerate(label_counts.items()):
        offset = width * multiplier
        rects = axes.bar(
            np.arange(len(counts)) + offset, list(counts.values()), width, label=subset
        )
        axes.bar_label(rects, padding=3)

    axes.set_title("Label")
    axes.set_ylabel("Counts")
    # axes.xaxis.set_ticks(x + (width * len(subsets)) / 2, labels)
    axes.xaxis.set_ticks(axes.xaxis.get_ticklocs()[1:-1], labels)
    if not all_in_one:
        axes.legend(loc="upper left", ncols=3)

    max_count = max(max(counts.values()) for counts in label_counts.values())
    axes.set_ylim(0, max_count + 5000)


def make_area_plot(axes: Axes, areas: list[float]):
    n_bins = 100
    N, bins, patches = axes.hist(areas, n_bins, color="orange")
    mean_bins = bins[:-1] + bins[1:]
    mean_bins /= 2

    mode_i = N.argmax()
    mode_bin = mean_bins[mode_i]
    patches[mode_i].set_facecolor("red")

    print("Min, Max image area:", min(areas), max(areas))
    print("Average image area:", sum(areas) / len(areas))
    print(f"Mode image area: {bins[mode_i]} ~ {bins[mode_i + 1]}")

    xticks = axes.get_xticks()[1:-1]
    xticklabels = axes.get_xticklabels()[1:-1]
    axes.set_xticks([*xticks, mode_bin], [*xticklabels, mode_bin])
    axes.set_xlabel("Area (pixels)")
    axes.set_ylabel("Number of images")
    axes.set_title("Image Area")

    x_axis_formatter = ScalarFormatter()
    x_axis_formatter.set_scientific(True)
    axes.xaxis.set_major_formatter(x_axis_formatter)
    axes.yaxis.set_major_formatter(PercentFormatter(xmax=len(areas)))


def show_classification_statistics(
    root_dir: str,
    title: Optional[str] = None,
    all_in_one: bool = False,
    excluded_subsets: list[str] = [],
):
    labels, label_counts, areas = analyze_images(
        root_dir, all_in_one=all_in_one, excluded_subsets=excluded_subsets
    )

    if title is None:
        title = root_dir.removesuffix(os.sep).split(os.sep)[-1]

    axes_1: Axes
    axes_2: Axes
    fig, (axes_1, axes_2) = plt.subplots(ncols=2, figsize=(15, 5))
    fig.suptitle(title)

    make_label_dist_plot(axes_1, labels, label_counts, all_in_one=all_in_one)
    make_area_plot(axes_2, areas)

    fig.tight_layout()
    fig.savefig(f"{title}.png")
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    show_classification_statistics(
        args.root_dir, args.title, args.all_in_one, args.excluded_subsets
    )
