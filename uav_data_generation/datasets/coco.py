import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

from PIL import Image
from torchvision.datasets import CocoDetection


class CocoDetectionWithMask(CocoDetection):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (str or ``pathlib.Path``): Root directory where images are downloaded to.
        mask_root (str or ``pathlib.Path``): Root directory where masks are downloaded to.
        ann_file (str or ``pathlib.Path``): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: Union[str, Path],
        mask_root: Union[str, Path],
        ann_file: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, ann_file, transforms, transform, target_transform)
        if isinstance(mask_root, str):
            mask_root = os.path.expanduser(mask_root)
        self.mask_root = mask_root

    def _load_mask(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = Image.open(os.path.join(self.mask_root, path))
        assert image.mode == "RGBA"
        mask = image.getchannel("A")
        return mask

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        if not isinstance(index, int):
            raise ValueError(
                f"Index must be of type integer, got {type(index)} instead."
            )

        id = self.ids[index]
        image = self._load_image(id)
        mask = self._load_mask(id)
        target = self._load_target(id)

        inputs = (image, mask)
        if self.transforms is not None:
            inputs, target = self.transforms(inputs, target)

        return inputs, target
