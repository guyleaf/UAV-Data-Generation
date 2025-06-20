import os
import pickle
import random
from pathlib import Path
from typing import Union

import numpy as np

from ..data import Dataset
from .coco import COCOWriter


class Checkpoint:
    random_state: tuple
    np_random_state: tuple
    image_index: int
    dataset: Dataset
    coco_writer: COCOWriter

    def __init__(
        self, rank: int, image_index: int, dataset: Dataset, coco_writer: COCOWriter
    ) -> None:
        # ensure the state is consistent.
        assert isinstance(rank, int) and rank >= 0
        assert isinstance(dataset, Dataset)
        assert isinstance(image_index, int) and 0 <= image_index < len(dataset)
        assert isinstance(coco_writer, COCOWriter) and image_index + 1 == len(
            coco_writer.images
        )

        self.rank = rank
        self.image_index = image_index
        self.dataset = dataset
        self.coco_writer = coco_writer
        self.save_random_states()

    def save_random_states(self):
        self.random_state = random.getstate()
        self.np_random_state = np.random.get_state()

    def restore_random_states(self):
        random.setstate(self.random_state)
        np.random.set_state(self.np_random_state)

    @classmethod
    def from_pickle(cls, path: str) -> "Checkpoint":
        with open(path, "rb") as f:
            instance = pickle.load(f)
        assert isinstance(instance, Checkpoint)
        return instance

    @classmethod
    def from_latest(cls, root_path: str) -> tuple["Checkpoint", str]:
        path = sorted(os.listdir(root_path))[-1]
        path = os.path.join(root_path, path)
        return cls.from_pickle(path), path

    def save_pickle(self, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)
        assert path.suffix == ".pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
