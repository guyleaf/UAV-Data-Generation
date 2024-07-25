import datetime
import json
import os
from typing import Optional


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

    @classmethod
    def get_image_format(
        id: int,
        file_name: str,
        w: int,
        h: int,
        license: Optional[int] = None,
    ):
        image = dict(id=id, file_name=file_name, width=w, height=h, license=license)
        return image

    @classmethod
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
        id = self.category_counter
        category = dict(id=id, name=name, supercategory=supercategory)
        self.categories.append(category)
        self.category_counter += 1
        return id

    def add_image(
        self,
        file_name: str,
        width: int,
        height: int,
    ) -> int:
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
        assert image_id < self.image_counter, f"Unknown image_id {image_id}"
        assert any(
            category["id"] == category_id for category in self.categories
        ), f"Unknown category {category_id}"

        ids = []
        for bbox in bboxes:
            id = self.annotation_counter
            annotation = self.get_annotation_format(id, image_id, category_id, *bbox)
            ids.append(id)
            self.annotations.append(annotation)
            self.annotation_counter += 1
        return ids

    def export(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.coco, f)

    def clear(self):
        self.image_counter = 1
        self.annotation_counter = 1
        self.images.clear()
        self.annotations.clear()
