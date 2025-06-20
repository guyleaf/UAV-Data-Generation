from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import PIL.Image as Image

from .. import distributed as dist
from ..utils.io import collect_images


class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError("__getitem__ is not implemented.")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("__len__ is not implemented.")


# TODO: Support num_workers?
class ImageFolder(Dataset):
    def __init__(self, root_path: Union[str, Path]):
        self.root_path = root_path
        self.image_paths = collect_images(root_path)

        # TODO: move to sampler?
        # split by rank
        if dist.is_distributed():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            self.image_paths = self.image_paths[rank::world_size]

    def __eq__(self, other: "ImageFolder"):
        if not isinstance(other, ImageFolder):
            return NotImplemented

        if len(self.image_paths) != len(other.image_paths):
            return False

        # consider the root path is changed.
        for path, other_path in zip(self.image_paths, other.image_paths):
            name = path.relative_to(self.root_path)
            other_name = other_path.relative_to(other.root_path)
            if name != other_name:
                return False
        return True

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        return Image.open(image_path), image_path

    def __len__(self):
        return len(self.image_paths)
