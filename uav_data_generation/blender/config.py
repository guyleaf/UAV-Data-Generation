import importlib.util
import os
from abc import ABC, abstractmethod
from typing import Optional

from blenderproc.python.utility.Initializer import DefaultConfig

from .utils.utils import get_blender_preset

AREA_RANGES = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


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
    def images_path(self) -> str:
        """Path to the folder of image files"""
        return os.path.join(self.out_dir, "backgrounds")

    # randomization settings

    # The angle range of x-axis in degree.
    x_range: tuple[int, int] = (-45, 45)
    # The angle range of y-axis in degree.
    y_range: tuple[int, int] = (-45, 45)
    # The angle range of z-axis in degree.
    z_range: tuple[int, int] = (0, 360)
    # The maximum number of UAVs per image.
    sample_range: tuple[int, int] = (1, 20)

    # Enable adaptive alignment with the step (only works when motion_blur is True).
    adaptive_alignment: bool = True
    # Align the camera with the UAV and move backward with the offset (m).
    alignment_z_offset: float = 0
    # Increase the alignment distance with the step (m) (adaptive-alignment only).
    alignment_z_step: float = 1e-3

    # The maximum IoF among UAVs in image. (max_uavs_iof > 0 -> accept occlusion)
    # IoF (intersection over foreground) = overlap / bbox (UAV1 or UAV2)
    # referenced from mmdetection
    max_uavs_iof: float = 0.2
    # The minimum IoF between UAV and the image. (min_uav_image_iof < 1 -> accept occlusion)
    # IoF (intersection over foreground) = overlap / bbox (UAV)
    min_uav_image_iof: float = 1
    # The area ranges [a, b) for small, medium, large.
    area_ranges: AREA_RANGES = (
        (1**2, 32**2),
        (32**2, 96**2),
        (96**2, 100000**2),
    )
    # The scale range relative to the size of image.
    # scale_range: tuple[float, float] = (0.2, 0.5)
    # The minimum UAV area after scaling.
    # min_uav_area: int = 1
    # Allow upscaling if UAV is smaller than the sampled size.
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
    # Adaptive noise threshold
    render_sampling_noise_threshold: float = 1e-2
    # The denoiser for rendering
    render_denoiser = "OPENIMAGEDENOISE"

    # Frame per second, unused in pipeline
    fps: int = 60
    # refers to https://docs.blender.org/api/3.5/bpy_types_enum_items/image_type_items.html
    file_format: str = "PNG"
    color_depth: int = 8
    # Transparency must be opened
    enable_transparency = True
    jpg_quality = 95
    # Control levels of all subdivision modifiers
    simplify_subdivision_render = 3

    # ligh paths settings

    # use the light preset or the custom parameters
    use_light_preset: bool = True
    light_preset_path: str = get_blender_preset("cycles/integrator", "Default")

    # custom ligh paths parameters
    max_bounces: int = DefaultConfig.max_bounces
    caustics_reflective: bool = True
    caustics_refractive: bool = True
    diffuse_bounces: int = DefaultConfig.diffuse_bounces
    glossy_bounces: int = DefaultConfig.glossy_bounces
    transmission_bounces: int = DefaultConfig.transmission_bounces
    volume_bounces: int = DefaultConfig.volume_bounces
    transparent_max_bounces: int = DefaultConfig.transparency_bounces

    # fast GI
    use_fast_gi: bool = False
    ao_bounces: int = 1
    ao_bounces_render: int = DefaultConfig.ao_bounces_render

    # misc settings

    # Path to where the final files will be saved.
    out_dir: str = "outputs"
    # Use UAV models specified by the list.
    models: Optional[list[str]] = None

    # The maximum number of checkpoints can keep.
    max_checkpoints: int = 3
    # The interval for saving a checkpoint.
    checkpoint_interval: int = 5

    skip_check: bool = False

    def __init__(self) -> None:
        if not self.skip_check:
            self.validate_args()

    def __repr__(self) -> str:
        # repr_str = f"Class name: '{self.__class__.__name__}' \n"
        repr_str = "Settings:\n"
        for attr in self._get_args():
            repr_str += f" {attr}: {getattr(self, attr)}\n"
        return repr_str

    def _get_args(self):
        attrs = filter(
            lambda attr: not attr.startswith("_") and not callable(getattr(self, attr)),
            dir(self),
        )
        return list(attrs)

    @classmethod
    def from_file(cls, path: str, class_name: str = "Config") -> "BaseConfig":
        spec = importlib.util.spec_from_file_location("cfg", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cfg = getattr(module, class_name)
        return cfg()

    def validate_args(self) -> None:
        assert self.scene_path.endswith(".blend"), (
            "The scene file should be a .blend file."
        )
        assert os.path.isdir(os.path.expanduser(self.images_path)), (
            "The images_path should be a folder path."
        )

        for range_ in [self.x_range, self.y_range, self.z_range]:
            assert range_[0] <= range_[1], (
                f"The left of range should be less than or equal to the right, {range_}."
            )

        assert 0 <= self.sample_range[0] <= self.sample_range[1], (
            "Invalid sample range format."
        )
        assert len(self.area_ranges) == 3, (
            "The area ranges should contain three ranges, small, medium, and large."
        )
        for area_range in self.area_ranges:
            assert 0 < area_range[0] < area_range[1], "Invalid area range format."
        for x, y in zip(self.area_ranges[:-1], self.area_ranges[1:]):
            assert x[1] <= y[0], (
                "The start of area range should be larger or equal to previous end one."
            )
        assert 0 <= self.max_uavs_iof <= 1, "The maximum UAVs IoF should be in [0, 1]."
        assert 0 < self.min_uav_image_iof <= 1, (
            "The minimum IoF between UAV and image should be in (0, 1]."
        )

    def to_dict(self):
        return {arg: getattr(self, arg) for arg in self._get_args()}


if __name__ == "__main__":
    try:
        BaseConfig()
        assert False
    except TypeError:
        pass
