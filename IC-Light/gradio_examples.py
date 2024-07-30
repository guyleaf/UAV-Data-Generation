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
    bg_examples = collect_images_from_dir(bg_root_path)[:num_samples]

    # make sure all image pairs are the same relative path
    assert all(
        os.path.relpath(fg, fg_root_path) == os.path.relpath(bg, bg_root_path)
        for fg, bg in zip(fg_examples, bg_examples)
    )

    return list(
        zip(
            fg_examples,
            bg_examples,
            [prompt] * num_samples,
            ["Use Background Image"] * num_samples,
            [seed] * num_samples,
        )
    )


def prepare_prompts():
    prompts = [
        "drones",
        "drones, natural lighting",
    ]
    return [[x] for x in prompts]
