import os

from rich import print

from uav_data_generation.blender.config import BaseConfig


class Config(BaseConfig):
    @property
    def scene_path(self) -> str:
        return os.path.expanduser("~/data/UAV/blender/UAVs.blend")

    @property
    def background_path(self) -> str:
        return os.path.expanduser(
            "~/data/UAV/blender/assets/backgrounds/studiolights/city.exr"
        )

    render_max_samples: int = 512

    motion_blur: bool = False
    # Align the camera with the UAV and move backward with the offset (m) (useful with motion blur).
    alignment_z_offset: float = 0

    skip_check: bool = True


if __name__ == "__main__":
    cfg = Config.from_file(__file__)
    print()
    print(cfg)
