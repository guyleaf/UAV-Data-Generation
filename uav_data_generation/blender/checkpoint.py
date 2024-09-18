import os
import pickle
import random

import numpy as np

from .coco import COCOWriter


class Checkpoint:
    random_state: tuple
    np_random_state: tuple
    image_index: int
    image_paths: list[str]
    coco_writer: COCOWriter

    def __init__(
        self, image_index: int, image_paths: list[str], coco_writer: COCOWriter
    ) -> None:
        self.image_index = image_index
        self.image_paths = image_paths
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

    def save_pickle(self, path: str):
        assert os.path.splitext(path)[1] == ".pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
