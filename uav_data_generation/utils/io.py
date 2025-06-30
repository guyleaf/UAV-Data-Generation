import mimetypes
from pathlib import Path
from typing import Union


def collect_images(path: Union[str, Path]) -> list[Path]:
    mime_checker = mimetypes.MimeTypes()

    def validate_file_type(path: Path):
        mime_type = mime_checker.guess_type(path)[0]
        return mime_type is not None and mime_type.startswith("image")

    path = Path(path)
    if path.is_dir():
        images = filter(validate_file_type, path.rglob("*.*"))
    else:
        images = [path]

    # if as_posix:
    #     images = map(methodcaller("as_posix"), images)

    return sorted(images)


def collect_images_from_images(
    images: list[Union[str, Path]],
    root_path: Union[str, Path],
    target_path: Union[str, Path],
) -> list[Path]:
    target_path = Path(target_path)
    if target_path.is_file():
        return [target_path]

    paths = []
    exts = [
        ext
        for ext, mime_type in mimetypes.types_map.items()
        if mime_type.startswith("image")
    ]
    exts += [ext.upper() for ext in exts]
    for image in images:
        image = Path(image)
        rel_path = image.relative_to(root_path)

        for ext in exts:
            path = target_path / rel_path.with_suffix(ext)
            if path.exists():
                break
        else:
            raise RuntimeError(
                f"The corresponding background image is not found, {rel_path}."
            )

        paths.append(path)
    return paths
