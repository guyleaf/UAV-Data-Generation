import os
from abc import ABC, abstractmethod
from typing import Optional


class BaseConfig(ABC):
    @property
    @abstractmethod
    def scene_path(self) -> str:
        """Path to the .blend scene file"""
        raise NotImplementedError

    @property
    @abstractmethod
    def background_path(self) -> str:
        """Path to the background HDRI file"""
        raise NotImplementedError

    @property
    @abstractmethod
    def images_path(self) -> str:
        """Path to the folder of image files"""
        raise NotImplementedError

    # # Path to the .blend scene file
    # scene_path: str
    # # Path to the background HDRI file
    # background_path: str
    # # Path to the folder of image files
    # images_path: str

    # randomization settings

    # The angle range of x-axis in degree.
    x_range: tuple[int, int] = (-45, 45)
    # The angle range of y-axis in degree.
    y_range: tuple[int, int] = (-45, 45)
    # The angle range of z-axis in degree.
    z_range: tuple[int, int] = (0, 360)
    # The maximum number of UAVs per image.
    max_samples: int = 20

    # Enable adaptive alignment with the step (useful with motion blur).
    adaptive_alignment: bool = True
    # Align the camera with the UAV and move backward with the offset (m) (useful with motion blur).
    alignment_z_offset: float = 0
    # Increase the alignment distance with the step (m) (--adaptive-alignment only).
    alignment_z_step: float = 1e-3

    # The maximum IoF (check overlap / bbox1 & overlap / bbox2) among UAVs in image. (max_iof > 0 -> accept occlusion)
    max_iof: float = 0.2
    # The scale range relative to the size of image.
    scale_range: tuple[float, float] = (0.2, 0.5)
    # The minimum UAV area after scaling.
    min_uav_area: int = 1
    # Allow upscaling if UAV is smaller than the sampled scale.
    allow_upscaling: bool = False

    # render settings

    # Enable motion blur or not.
    motion_blur: bool = True
    # The original resolution of rendered UAV samples.
    render_resolution: tuple[int, int] = (1920, 1920)
    # The maximum number of samples to render for each pixel.
    render_max_samples: int = 1024
    # The tile size for rendering a image (less -> slower & lower VRAM requirement, higher -> faster & higher VRAM requirement).
    render_tile_size: int = 1920

    # misc settings

    # Path to where the final files will be saved.
    out_dir: str = "outputs"
    # Use UAV models specified by the list.
    models: Optional[list[str]] = None
    # The seed for random sampling.
    seed: int = 2024

    # The maximum number of checkpoints can keep.
    max_checkpoints: int = 3
    # The interval for saving a checkpoint.
    checkpoint_interval: int = 5

    def __init__(self) -> None:
        self.validate_args()
        self.__dict__ = {arg: getattr(self, arg) for arg in self._get_args()}

    def _get_args(self):
        attrs = filter(
            lambda attr: not attr.startswith("_") and not callable(getattr(self, attr)),
            dir(self),
        )
        return list(attrs)

    def __repr__(self) -> str:
        repr_str = f"Class name: '{self.__class__.__name__}' \n"
        attrs = filter(
            lambda attr: not attr.startswith("_") and not callable(getattr(self, attr)),
            dir(self),
        )
        repr_str += "Settings:\n"
        for attr in attrs:
            repr_str += f"\t{attr}: {getattr(self, attr)}\n"
        return repr_str

    def validate_args(self) -> None:
        assert self.scene_path.endswith(
            ".blend"
        ), "The scene file should be a .blend file."
        assert os.path.isdir(
            os.path.expanduser(self.images_path)
        ), "The images_path should be a folder path."

        for range_ in [self.x_range, self.y_range, self.z_range]:
            assert (
                range_[0] <= range_[1]
            ), f"The left of range should be less than or equal to the right, {range_}."

        assert (
            0 < self.max_samples
        ), "The maximum number of samples should be greater than 0."
        assert (
            0 < self.scale_range[0] <= self.scale_range[1] <= 1
        ), "The scale range should be in (0, 1]."
        assert 0 <= self.max_iof <= 1, "The maximum IoF should be in (0, 1)."


if __name__ == "__main__":
    try:
        BaseConfig()
        assert False
    except TypeError:
        pass
