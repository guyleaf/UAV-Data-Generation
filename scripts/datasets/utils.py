from mimetypes import MimeTypes
from pathlib import Path
from typing import Optional, Union

from sklearn.model_selection import train_test_split


def split_into_train_val_test(x: list, val_ratio: float, test_ratio: float, seed: int):
    if test_ratio < 1.0:
        train, test = train_test_split(x, test_size=test_ratio, random_state=seed)
        ratio_remaining = 1.0 - test_ratio
        val_ratio = val_ratio / ratio_remaining
        if val_ratio < 1.0:
            train, val = train_test_split(train, test_size=val_ratio, random_state=seed)
        else:
            val = train
            train = []
    else:
        test = x
        train = val = []

    return {"train": train, "val": val, "test": test}


def split_into_train_val(
    x: list, val_ratio: float, seed: int, labels: Optional[list] = None
):
    if val_ratio < 1.0:
        train, val = train_test_split(
            x, test_size=val_ratio, random_state=seed, stratify=labels
        )
    else:
        val = x
        train = []

    return {"train": train, "val": val}


def collect_images(root_path: Union[str, Path]) -> list[Path]:
    mime_checker = MimeTypes()

    def validate_file_type(path: Path):
        mime_type = mime_checker.guess_type(path)[0]
        return mime_type is not None and "image" in mime_type

    return sorted(
        filter(
            validate_file_type,
            Path(root_path).rglob("*.*"),
        )
    )
