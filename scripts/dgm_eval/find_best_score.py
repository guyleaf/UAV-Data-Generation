import argparse
from pathlib import Path
from pprint import pprint

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Find the best epoch based on metrics.",
    )
    parser.add_argument(
        "scores_file", type=str, help="The output scores file from dgm-eval."
    )
    parser.add_argument(
        "--iters",
        type=str,
        nargs="+",
        default=None,
        help="Which iters are to be evaluated. By default, evaluate all iters.",
    )
    # parser.add_argument(
    #     "--phase",
    #     type=str,
    #     default="test",
    #     help="Which phase are to be evaluated.",
    # )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["fd", "kd_value"],
        help="Which metrics are used to compare.",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="asc",
        choices=["asc", "desc"],
        help="Which order of metric values is used in sorting. Sort in metrics.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Show top-k epochs.",
    )
    args = parser.parse_args()
    assert args.topk > 0
    return args


if __name__ == "__main__":
    args = parse_args()

    scores_file = Path(args.scores_file)
    scores = np.load(scores_file, allow_pickle=True)

    test_datasets = scores["run_params"].item()["test_datasets"]
    scores = scores["scores"].item()
    scores = {k: (test_datasets[i], v) for i, (k, v) in enumerate(scores.items())}

    if args.iters is not None:
        iters = args.iters
        scores = {i: scores[i] for i in iters}

    results = sorted(
        scores.items(),
        key=lambda score: (score[1][1][metric] for metric in args.metrics),
        reverse=(args.order == "desc"),
    )[: args.topk]

    print(f"Sort scores in metrics {args.order} order:", args.metrics)
    print("=========================")
    for i, (dataset, score) in results:
        print(f"Iter: {i}, Dataset: {dataset}")
        pprint(score)
        print("=========================")
