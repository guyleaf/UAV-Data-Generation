from typing import Iterable, Union


def convert_xywh_to_xyxy(bboxes: Iterable[Union[list[int], tuple[int, int, int, int]]]):
    return [
        type(bbox)([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        for bbox in bboxes
    ]
