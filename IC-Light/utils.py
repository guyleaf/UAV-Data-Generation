import glob
import os
from mimetypes import MimeTypes
from typing import Union

import numpy as np
import PIL.Image as Image
import PIL.ImageOps as ImageOps

# def resize_and_center_crop(
#     image: Union[np.ndarray, Image.Image], target_width: int, target_height: int
# ):
#     pil_image = image
#     if isinstance(pil_image, np.ndarray):
#         pil_image = Image.fromarray(pil_image)

#     original_width, original_height = pil_image.size
#     scale_factor = max(target_width / original_width, target_height / original_height)
#     resized_width = int(round(original_width * scale_factor))
#     resized_height = int(round(original_height * scale_factor))
#     resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
#     left = (resized_width - target_width) / 2
#     top = (resized_height - target_height) / 2
#     right = (resized_width + target_width) / 2
#     bottom = (resized_height + target_height) / 2
#     cropped_image = resized_image.crop((left, top, right, bottom))

#     if isinstance(image, np.ndarray):
#         return np.array(cropped_image)
#     return cropped_image


def resize_and_pad(
    image: Union[np.ndarray, Image.Image],
    target_width: int,
    target_height: int,
    color=(127, 127, 127),
):
    pil_image = image
    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)

    resized_padded_image = ImageOps.pad(
        pil_image, (target_width, target_height), Image.LANCZOS, color=color
    )

    if isinstance(image, np.ndarray):
        return np.array(resized_padded_image)
    return resized_padded_image


def resize(
    image: Union[np.ndarray, Image.Image], target_width: int, target_height: int
):
    pil_image = image
    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)

    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)

    if isinstance(image, np.ndarray):
        return np.array(resized_image)
    return resized_image


def pad_to_divisible(
    image: Union[np.ndarray, Image.Image], divisor: int = 8, color=(127, 127, 127)
):
    pil_image = image
    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)

    # calculate border length
    right = (divisor - pil_image.size[0] % divisor) % divisor
    bottom = (divisor - pil_image.size[1] % divisor) % divisor

    if right != 0 or bottom != 0:
        padded_image = ImageOps.expand(pil_image, (0, 0, right, bottom), fill=color)
    else:
        padded_image = pil_image.copy()

    if isinstance(image, np.ndarray):
        return np.array(padded_image)
    return padded_image


def remove_pad_from_divisible(
    image: Union[np.ndarray, Image.Image], target_width: int, target_height: int
):
    pil_image = image
    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)

    # calculate border length
    right = pil_image.size[0] - target_width
    bottom = pil_image.size[1] - target_height
    assert right >= 0 and bottom >= 0

    if right != 0 or bottom != 0:
        cropped_image = ImageOps.crop(pil_image, (0, 0, right, bottom))
    else:
        cropped_image = pil_image.copy()

    if isinstance(image, np.ndarray):
        return np.array(cropped_image)
    return cropped_image


def rescale(
    image: Union[np.ndarray, Image.Image], target_width: int, target_height: int
):
    pil_image = image
    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)

    resized_image = ImageOps.cover(
        pil_image, (target_width, target_height), Image.LANCZOS
    )

    if isinstance(image, np.ndarray):
        return np.array(resized_image)
    return resized_image


def collect_images_from_dir(root_path: str, absolute: bool = True):
    mime_checker = MimeTypes()

    def validate_file_type(path: str):
        mime_type = mime_checker.guess_type(path)[0]
        return mime_type is not None and "image" in mime_type

    images = filter(
        validate_file_type,
        glob.iglob("**/*.*", root_dir=root_path, recursive=True),
    )
    if absolute:
        images = map(lambda path: os.path.join(root_path, path), images)

    return sorted(images)
