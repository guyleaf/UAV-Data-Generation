import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc

import argparse
from operator import methodcaller
from typing import Iterable

from uav_data_generation import logging
from uav_data_generation.blender.setup import load_blend
from uav_data_generation.blender.utils.material import collect_materials_by_cp


def print_list(data: Iterable[str]):
    logger = logging.get_logger()
    for i, item in enumerate(data, start=1):
        logger.info(f"  {i}. {item}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script for listing all models and materials for random in blender",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scene_path", type=str, help="Path to the .blend scene file")
    args = parser.parse_args()

    logger = logging.get_logger()

    bproc.init()

    logger.info(f"Loading the scene: {args.scene_path}")
    entities, uav_model_dict = load_blend(args.scene_path)
    materials = collect_materials_by_cp()
    material_names = list(map(methodcaller("get_name"), materials))

    logger.info(f"Models: {len(uav_model_dict)} models")
    print_list(uav_model_dict.keys())
    logger.info(f"Materials for random: {len(material_names)} materials")
    print_list(material_names)
