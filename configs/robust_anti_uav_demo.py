import os
from typing import Optional

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

    x_range: tuple[int, int] = (-45, 45)
    y_range: tuple[int, int] = (-45, 45)
    max_uavs_iof: float = 0.2
    min_uav_image_iof: float = 0.5
    sample_range: tuple[int, int] = (1, 4)
    area_ranges: AREA_RANGES = (
        (20**2, 32**2),
        (32**2, 96**2),
        (96**2, 100000**2),
    )

    render_max_samples: int = 512
    motion_blur: bool = False
    render_denoiser: str = "OPTIX"

    # control levels of all subdivision modifiers
    # our models are using level 1.
    # so, 0 means low poly.
    simplify_subdivision_render: int = 0
    models: Optional[list[str]] = None

    out_dir: str = os.path.expanduser("~/data/UAV/Robust_Anti_UAV_Low_Test")

    # skip_check: bool = True


if __name__ == "__main__":
    cfg = Config.from_file(__file__)
    print()
    print(cfg)
