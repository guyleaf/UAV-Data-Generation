import os

import blenderproc as bproc  # noqa: F401 # isort:skip, this should be at the top due to the check of blenderproc

from rich import print

from uav_data_generation.blender.config import AREA_RANGES, BaseConfig


class Config(BaseConfig):
    @property
    def scene_path(self) -> str:
        return os.path.expanduser("~/data/UAV/blender/UAVs.blend")

    @property
    def background_path(self) -> str:
        return os.path.expanduser(
            "~/data/UAV/blender/assets/backgrounds/studiolights/city.exr"
        )

    @property
    def images_path(self) -> str:
        return os.path.expanduser(
            "~/data/UAV/Weather_Anti_UAV_T_NO_RETRY_AREA/backgrounds"
        )

    x_range: tuple[int, int] = (-30, 30)
    y_range: tuple[int, int] = (-30, 30)
    max_iof: float = 0.2
    sample_range: tuple[int, int] = (1, 6)
    area_ranges: AREA_RANGES = (
        (20**2, 32**2),
        (32**2, 96**2),
        (96**2, 100000**2),
    )

    # faster prototyping
    render_max_samples: int = 128

    out_dir: str = os.path.expanduser("~/data/UAV/Weather_Anti_UAV_T_NO_RETRY_AREA_0")


if __name__ == "__main__":
    cfg = Config.from_file(__file__)
    print()
    print(cfg)
