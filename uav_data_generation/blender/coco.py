import logging

from ..distributed import is_distributed, get_world_comm, get_rank, is_main_process  # noqa: F401 # isort:skip, this should be at the top

import datetime
import json
import os
from pathlib import Path
from typing import Optional, Union


class COCOWriter:
    def __init__(
        self,
        year: int = 2024,
        version: int = 1,
        description: str = "Weather Anti-UAV dataset",
        url: str = "https://github.com/guyleaf/UAV-Data-Generation",
        licenses: list[dict] = [],
    ) -> None:
        self.coco = dict(
            info=dict(
                year=year,
                version=version,
                description=description,
                url=url,
                date_created=datetime.date.today().isoformat(),
            ),
            licenses=licenses,
            images=[],
            annotations=[],
            categories=[],
        )
        self.images = self.coco["images"]
        self.annotations = self.coco["annotations"]
        self.categories = self.coco["categories"]

        self.category_counter = 1
        self.image_counter = 1
        self.annotation_counter = 1

        self.logger = logging.getLogger()

    @staticmethod
    def get_image_format(
        id: int,
        file_name: str,
        w: int,
        h: int,
        license: Optional[int] = None,
    ):
        image = dict(id=id, file_name=file_name, width=w, height=h, license=license)
        return image

    @staticmethod
    def get_annotation_format(
        id: int,
        image_id: int,
        category_id: int,
        x: int,
        y: int,
        w: int,
        h: int,
        attributes: dict = {},
    ):
        annotation = dict(
            id=id,
            image_id=image_id,
            category_id=category_id,
            area=w * h,
            bbox=[x, y, w, h],
            iscrowd=0,
            attributes=attributes,
        )
        return annotation

    def add_category(self, name: str, supercategory: str) -> int:
        assert isinstance(name, str) and isinstance(supercategory, str)

        id = self.category_counter
        category = dict(id=id, name=name, supercategory=supercategory)
        self.categories.append(category)
        self.category_counter += 1
        return id

    def find_category(self, name: str, supercategory: Optional[str] = None) -> int:
        categories = filter(lambda cat: cat["name"] == name, self.categories)
        if supercategory is not None:
            categories = filter(
                lambda cat: cat["supercategory"] == supercategory, categories
            )
        categories = list(categories)
        assert len(categories) == 1
        return categories[0]["id"]

    def add_image(
        self,
        file_name: str,
        width: int,
        height: int,
    ) -> int:
        assert isinstance(file_name, str) and len(file_name) > 0
        assert isinstance(width, int) and isinstance(height, int)
        assert width > 0 and height > 0, f"Invalid size, ({width}, {height})"

        id = self.image_counter
        image = self.get_image_format(id, file_name, width, height)
        self.images.append(image)
        self.image_counter += 1
        return id

    def add_annotations(
        self,
        image_id: int,
        category_id: int,
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[int]:
        assert isinstance(image_id, int) and isinstance(category_id, int)
        assert image_id < self.image_counter, f"Unknown image_id {image_id}"
        assert any(category["id"] == category_id for category in self.categories), (
            f"Unknown category {category_id}"
        )

        image = self.images[image_id - 1]
        width = image["width"]
        height = image["height"]

        ids = []
        for bbox in bboxes:
            assert all(isinstance(val, int) for val in bbox), "Invalid bbox data type."
            x, y, w, h = bbox
            assert width >= x + w > x >= 0 and height >= y + h > y >= 0

            id = self.annotation_counter
            annotation = self.get_annotation_format(id, image_id, category_id, *bbox)
            ids.append(id)
            self.annotations.append(annotation)
            self.annotation_counter += 1
        return ids

    def _gather_coco(self):
        comm = get_world_comm()
        rank = get_rank(comm)

        self.logger.info("Gathering statistics...")

        # gather statistics for each rank
        statistics: list[tuple[int, int]]
        statistics = comm.allgather((len(self.images), len(self.annotations)))

        # calculate the rank offset for id
        image_offset = anno_offset = 0
        for num_images, num_annos in statistics[:rank]:
            image_offset += num_images
            anno_offset += num_annos

        self.logger.info("Gathering images and annotations...")

        # add the offset to id
        for image in self.images:
            image["id"] += image_offset
        for annotation in self.annotations:
            annotation["id"] += anno_offset
            annotation["image_id"] += image_offset

        data: list[tuple[list[dict], list[dict]]]
        data = comm.gather((self.images, self.annotations))

        images = []
        annotations = []
        if is_main_process():
            for rank_images, rank_annotations in data:
                images.extend(rank_images)
                annotations.extend(rank_annotations)
        return images, annotations

    def export(self, path: Union[str, Path]):
        if is_distributed():
            images, annotations = self._gather_coco()
        else:
            images, annotations = self.images, self.annotations

        if is_main_process():
            self.coco["images"] = images
            self.coco["annotations"] = annotations

            if isinstance(path, str):
                path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.coco, f)

            self.logger.info(
                ":white_check_mark: Exported the COCO annotations successfully."
            )

    def clear(self):
        self.image_counter = 1
        self.annotation_counter = 1
        self.images.clear()
        self.annotations.clear()

    def validate(self, root_path: Union[str, Path]):
        """Check if all images exist."""
        for image in self.images:
            image_path = os.path.join(root_path, image["file_name"])
            if not os.path.exists(image_path):
                return False
        return True
