import datetime
from typing import Optional


class COCOWriter:
    def __init__(
        self,
        year: int = 2024,
        version: int = 1,
        description: str = "Weather Anti-UAV dataset",
        url: str = "https://github.com/guyleaf/UAV-Data-Generation",
        categories: list[dict] = [
            dict(id=1, name="drone", supercategory="UAV"),
        ],
    ) -> None:
        self.coco = dict(
            info=dict(
                year=year,
                version=version,
                description=description,
                url=url,
                date_created=datetime.date.today().isoformat(),
            ),
            licenses=[
                dict(
                    id=1,
                    name="GNU General Public License v3.0",
                    url="https://www.gnu.org/licenses/gpl-3.0.en.html",
                )
            ],
            images=[],
            annotations=[],
            categories=categories,
        )
        self.images = self.coco["images"]
        self.annotations = self.coco["annotations"]

    @classmethod
    def get_image_format(
        id: int,
        file_name: str,
        h: int,
        w: int,
        license: Optional[int] = None,
        caption: Optional[str] = None,
        weight: Optional[float] = None,
    ):
        image = dict(id=id, file_name=file_name, height=h, width=w, license=license)
        if caption is not None:
            weight = weight if weight is not None else 1.0
            image["caption"] = caption
            image["weight"] = weight
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
        result = dict(
            id=id,
            image_id=image_id,
            category_id=category_id,
            area=w * h,
            bbox=[x, y, w, h],
            iscrowd=0,
            attributes=attributes,
        )
        return result
