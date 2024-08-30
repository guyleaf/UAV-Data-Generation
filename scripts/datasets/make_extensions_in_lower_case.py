import argparse
from pathlib import Path

import rich
import rich.progress


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "root_folder", type=str, help="The root folder containing files"
    )
    parser.add_argument(
        "--dry-run",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="execute without renaming",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_folder = Path(args.root_folder)

    renaming_counter = 0
    for path in rich.progress.track(root_folder.rglob("*.*")):
        suffix = path.suffix.lower()
        if suffix != path.suffix:
            if not args.dry_run:
                path.rename(path.with_suffix(suffix))
            renaming_counter += 1

    print(f"Renamed {renaming_counter} files in {root_folder}.")
