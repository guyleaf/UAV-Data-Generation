import os

from utils import collect_images_from_dir

DEFAULT_PROMPT = "drones, natural lighting"


def prepare_examples(
    fg_root_path: str,
    bg_root_path: str,
    num_samples: int = 100,
    prompt: str = DEFAULT_PROMPT,
    seed: int = 12345,
):
    fg_examples = collect_images_from_dir(fg_root_path)[:num_samples]
    bg_examples = []
    for fg in fg_examples:
        path = os.path.relpath(fg, fg_root_path)
        path, _ = os.path.splitext(path)
        path = os.path.join(bg_root_path, f"{path}.jpg")
        bg_examples.append(path)

    # make sure all image pairs exist
    assert all(os.path.exists(bg) for bg in bg_examples)

    return list(
        map(
            list,
            zip(
                fg_examples,
                bg_examples,
                [prompt] * num_samples,
                ["Use Background Image"] * num_samples,
                [seed] * num_samples,
            ),
        )
    )


def prepare_prompts():
    prompts = [
        "drones",
        "drones, natural lighting",
    ]
    return [[x] for x in prompts]
