import argparse
from pathlib import Path

from PIL import Image
from rich.progress import track

from uav_data_generation.utils.io import collect_images, collect_images_from_images


def combine_fg_bg(fg: str | Path, bg: str | Path, out_dir: str | Path):
    fg_paths = collect_images(fg)
    bg_paths = collect_images_from_images(fg_paths, fg, bg)

    out_dir = Path(out_dir)
    for bg_path, fg_path in track(zip(bg_paths, fg_paths)):
        bg_img = Image.open(bg_path).convert("RGB")
        fg_img = Image.open(fg_path)

        assert fg_img.mode == "RGBA"
        fg_mask = fg_img.getchannel("A")

        composite_img = Image.composite(fg_img, bg_img, fg_mask)

        target_dir = out_dir / fg_path.relative_to(fg).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        name = fg_path.stem

        composite_img.save(target_dir / f"{name}.png")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "bg", type=str, help="Directory or Path to the background image(s)."
    )
    parser.add_argument(
        "fg", type=str, help="Directory or Path to the foreground image(s)."
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Path to export results.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    combine_fg_bg(**vars(args))
