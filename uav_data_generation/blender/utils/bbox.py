from typing import Literal, Union

import numpy as np


def find_bbox_xyxy_by_alpha(image: np.ndarray) -> tuple[int, int, int, int]:
    assert image.shape[-1] == 4, "The color format should be in RGBA."
    y_indices, x_indices = image[..., -1].nonzero()
    # follow the COCO format which is 0-indexed
    # reference: https://cocodataset.org/#format-data
    min_x = np.amin(x_indices)
    max_x = np.amax(x_indices) + 1
    min_y = np.amin(y_indices)
    max_y = np.amax(y_indices) + 1
    return min_x, min_y, max_x, max_y


def find_overlap_bbox(
    bbox1: Union[np.ndarray, tuple[int, int, int, int]],
    bbox2: Union[np.ndarray, tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)

    bbox1 = bboxes_xywh_to_xyxy(bbox1[None, :])
    bbox2 = bboxes_xywh_to_xyxy(bbox2[None, :])
    bbox1 = bbox1[0].tolist()
    bbox2 = bbox2[0].tolist()

    x_start = max(bbox1[0], bbox2[0])
    y_start = max(bbox1[1], bbox2[1])
    x_end = min(bbox1[2], bbox2[2])
    y_end = min(bbox1[3], bbox2[3])
    w = max(x_end - x_start, 0)
    h = max(y_end - y_start, 0)
    return x_start, y_start, w, h


def bboxes_xywh_to_xyxy(bboxes: np.ndarray):
    bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
    return bboxes


# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/evaluation/functional/bbox_overlaps.py
def bbox_overlaps(
    bboxes1: Union[np.ndarray, list[tuple[int, int, int, int]]],
    bboxes2: Union[np.ndarray, list[tuple[int, int, int, int]]],
    mode: Literal["iou", "iof"] = "iou",
    eps: float = 1e-6,
):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (Union[np.ndarray, list[tuple[int, int, int, int]]]): Shape (n, 4), Format (x, y, w, h)
        bboxes2 (Union[np.ndarray, list[tuple[int, int, int, int]]]): Shape (k, 4), Format (x, y, w, h)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)

    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ["iou", "iof"]
    if not isinstance(bboxes1, np.ndarray):
        bboxes1 = np.array(bboxes1)
    if not isinstance(bboxes2, np.ndarray):
        bboxes2 = np.array(bboxes2)

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious

    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True

    area1 = bboxes1[:, 2] * bboxes1[:, 3]
    area2 = bboxes2[:, 2] * bboxes2[:, 3]
    bboxes1 = bboxes_xywh_to_xyxy(bboxes1)
    bboxes2 = bboxes_xywh_to_xyxy(bboxes2)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(y_end - y_start, 0)
        if mode == "iou":
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union

    if exchange:
        ious = ious.T
    return ious
