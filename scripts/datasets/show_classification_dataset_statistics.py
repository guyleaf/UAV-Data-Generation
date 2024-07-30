import argparse
import os
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

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


def show_classification_statistics(
    root_dir: str,
    title: Optional[str] = None,
    all_in_one: bool = False,
    excluded_subsets: list[str] = [],
):
    root_dir = root_dir.removesuffix(os.sep)

    if all_in_one:
        subsets = [""]
    else:
        subsets = [
            subset for subset in os.listdir(root_dir) if subset not in excluded_subsets
        ]

    max_count = 0
    subset_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for subset in subsets:
        subset_dir = os.path.join(root_dir, subset)

        labels = os.listdir(subset_dir)
        for label in labels:
            label_dir = os.path.join(subset_dir, label)
            if not os.path.isdir(label_dir):
                continue
            count = len(os.listdir(label_dir))
            subset_counts[subset][label] = count
        count = sum(subset_counts[subset].values())
        subset_counts[subset]["all"] = count
        max_count = max(max_count, count)

    if title is None:
        title = root_dir.split(os.sep)[-1]
    labels = list(list(subset_counts.values())[0].keys())
    x = np.arange(len(labels))
    width = 0.25  # the width of the bars

    fig, axe = plt.subplots()
    for multiplier, (subset, counts) in enumerate(subset_counts.items()):
        offset = width * multiplier
        rects = axe.bar(x + offset, list(counts.values()), width, label=subset)
        axe.bar_label(rects, padding=3)

    axe.set_title(title)
    axe.set_ylabel("Counts")
    axe.xaxis.set_ticks(x + width, labels)
    if not all_in_one:
        axe.legend(loc="upper left", ncols=3)
    axe.set_ylim(0, max_count + 5000)
    fig.savefig(f"{title}.png")


if __name__ == "__main__":
    args = parse_args()
    show_classification_statistics(
        args.root_dir, args.title, args.all_in_one, args.excluded_subsets
    )
