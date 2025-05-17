import numpy as np
from blenderproc.python.types.MeshObjectUtility import MeshObject


def compute_poi(objects: list[MeshObject]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes a point of interest in the scene. Point is defined as a location of the one of the selected objects
    that is the closest one to the mean location of the bboxes of the selected objects.

    :param objects: The list of mesh objects that should be considered.
    :return: Point of interest in the scene.
    """
    # Init matrix for all points of all bounding boxes
    mean_bb_points = []

    for obj in objects:
        # Get bounding box corners
        bb_points = obj.get_bound_box()
        # Compute mean coords of bounding box
        mean_bb_points.append(np.mean(bb_points, axis=0))

    # Query point - mean of means
    mean_bb_point = np.mean(mean_bb_points, axis=0)
    min_bb_point = np.min(mean_bb_points, axis=0)
    max_bb_point = np.max(mean_bb_points, axis=0)

    # Closest point (from means) to query point (mean of means)
    poi = mean_bb_points[
        np.argmin(np.linalg.norm(mean_bb_points - mean_bb_point, axis=1))
    ]
    min_poi = mean_bb_points[
        np.argmin(np.linalg.norm(mean_bb_points - min_bb_point, axis=1))
    ]
    max_poi = mean_bb_points[
        np.argmin(np.linalg.norm(mean_bb_points - max_bb_point, axis=1))
    ]

    return poi, min_poi, max_poi
