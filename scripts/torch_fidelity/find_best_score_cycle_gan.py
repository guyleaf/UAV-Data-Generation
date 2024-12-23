import argparse
import json
import mimetypes
import os
from pathlib import Path
from pprint import pprint
from typing import Union

import torch_fidelity
import torch_fidelity.defaults
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Find the best epoch based on metrics.",
    )
    # region arguments
    parser.add_argument(
        "images_dir",
        type=str,
        help="The images folder for comparison, e.g. ./datasets/horse2zebra/testB.",
    )
    parser.add_argument(
        "result_dir",
        type=str,
        help="The result folder of the experiment from cycle-gan forked repository.",
    )
    parser.add_argument(
        "out_dir",
        type=str,
        help="The output folder of the result.",
    )
    parser.add_argument(
        "--fake-dir",
        type=str,
        default="fake_*",
        help="Used for searching the fake images folder",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Which epochs are to be evaluated. By default, evaluate all epochs.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="test",
        help="Which phase are to be evaluated.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=[torch_fidelity.KEY_METRIC_FID, torch_fidelity.KEY_METRIC_KID_MEAN],
        choices=[
            getattr(torch_fidelity, k)
            for k in dir(torch_fidelity)
            if k.startswith("KEY_")
        ],
        help="Which metrics are used to compare from torch-fidelity output.",
    )
    parser.add_argument(
        "--order",
        type=str,
        nargs="+",
        default=["asc", "asc"],
        choices=["asc", "desc"],
        help="Which order of metric values is used in sorting. Sort in metrics.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Show top-k epochs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use. Like cuda, cuda:0 or cpu",
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="Not to load the existed fid results from the result folder",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Not to save fid results in the result folder",
    )
    # endregion

    # region torch-fidelity arguments
    torch_fid_group = parser.add_argument_group("torch-fidelity arguments")
    torch_fid_group.add_argument(
        "-i",
        "--isc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Calculate ISC (Inception Score)",
    )
    torch_fid_group.add_argument(
        "-f",
        "--fid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calculate FID (Frechet Inception Distance)",
    )
    torch_fid_group.add_argument(
        "-k",
        "--kid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calculate KID (Kernel Inception Distance)",
    )
    torch_fid_group.add_argument(
        "-r",
        "--prc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Calculate PRC (Precision and Recall)",
    )
    torch_fid_group.add_argument(
        "-p",
        "--ppl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Calculate PPL (Perceptual Path Length)",
    )
    torch_fid_group.add_argument(
        "-b", "--batch-size", default=64, type=int, help="Batch size to use"
    )
    # endregion

    args = parser.parse_args()
    torch_fid_args = argparse.Namespace(
        **{
            a.dest: getattr(args, a.dest, a.default)
            for a in torch_fid_group._group_actions
        }
    )

    assert os.path.isdir(args.images_dir) and os.path.isdir(args.result_dir)
    assert args.topk > 0
    return args, torch_fid_args


def collect_images(path: Union[str, Path]) -> list[Path]:
    mime_checker = mimetypes.MimeTypes()

    def validate_file_type(path: Path):
        mime_type = mime_checker.guess_type(path)[0]
        return mime_type is not None and mime_type.startswith("image")

    path = Path(path)
    if path.is_dir():
        return sorted(filter(validate_file_type, path.rglob("*.*")))
    else:
        return [path]


if __name__ == "__main__":
    args, torch_fid_args = parse_args()

    if args.device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device[5:]
        torch_fid_args.gpu = True
    else:
        torch_fid_args.gpu = False

    # collect epoch folders
    result_dir = Path(args.result_dir)
    if args.epochs is None:
        epoch_dirs = list(result_dir.glob(f"{args.phase}_*"))
    else:
        epoch_dirs = [result_dir / f"{args.phase}_{epoch}" for epoch in args.epochs]

    results = {}

    # load previous calculated results
    out_dir = Path(args.out_dir)
    scores_json = out_dir / "scores.json"
    if not args.no_load and scores_json.exists():
        with open(scores_json) as f:
            results.update(json.load(f))

    real_image_count = len(collect_images(args.images_dir))
    torch_fid_args = vars(torch_fid_args)
    for epoch_dir in tqdm(epoch_dirs):
        fake_dir = epoch_dir / "images"
        fake_dirs = sorted(fake_dir.glob(args.fake_dir))

        for fake in fake_dirs:
            name = f"{epoch_dir.name}_{fake.name}"
            # skip calculated epochs
            if name in results:
                continue

            torch_fid_args["input1"] = args.images_dir
            torch_fid_args["input2"] = fake.as_posix()

            if torch_fid_args["kid"]:
                # reference: https://github.com/layer6ai-labs/dgm-eval/blob/9694be578dc2438a44fe85c91955bf6192ea2d4e/dgm_eval/metrics/mmd.py#L6
                image_count = min(real_image_count, len(collect_images(fake)))
                torch_fid_args["kid_subset_size"] = min(
                    image_count, torch_fidelity.defaults.DEFAULTS["kid_subset_size"]
                )

            metrics = torch_fidelity.calculate_metrics(**torch_fid_args)
            results[name] = metrics

    # python's sort is stableF
    # https://docs.python.org/3.10/howto/sorting.html#sort-stability-and-complex-sorts
    topk_results = list(results.items())
    for metric, order in reversed(list(zip(args.metrics, args.order))):
        topk_results.sort(key=lambda v: v[1][metric], reverse=(order == "desc"))

    print(f"Sort scores in metrics {args.order} order:", args.metrics)
    pprint(dict(topk_results[: args.topk]), sort_dicts=False)

    if not args.no_save:
        scores_json.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_json, "w") as f:
            json.dump(results, f, indent=4)
