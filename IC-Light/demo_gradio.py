import argparse
import os

import gradio as gr
import numpy as np
from gradio_examples import prepare_examples, prepare_prompts
from gradio_utils import BGSource
from iclight import ICLightInferencer


def parse_args():
    parser = argparse.ArgumentParser(
        description="A demo script for IC-Light powered by gradio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fg", type=str, help="Root path of foreground images")
    parser.add_argument("bg", type=str, help="Root path of background images")
    parser.add_argument(
        "--model",
        type=str,
        default="stablediffusionapi/realistic-vision-v51",
        help="Name of the stable diffusion 1.5 model from huggingface hub",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint file of IC-Light",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples should be shown.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU device id will be used for demo",
    )
    args = parser.parse_args()
    return args


def handle_process_relight(inferencer: ICLightInferencer):
    def _process_relight(
        input_fg: np.ndarray,
        input_bg: np.ndarray,
        prompt: str,
        seed: int,
        steps: int,
        a_prompt: str,
        n_prompt: str,
        cfg: float,
        highres_scale: float,
        highres_denoise: float,
        bg_source: str,
    ):
        image_height, image_width = input_fg
        bg_source = BGSource(bg_source)

        if bg_source == BGSource.UPLOAD:
            pass
        elif bg_source == BGSource.UPLOAD_FLIP:
            input_bg = np.fliplr(input_bg)
        elif bg_source == BGSource.GREY:
            input_bg = (
                np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
            )
        elif bg_source == BGSource.LEFT:
            gradient = np.linspace(224, 32, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.RIGHT:
            gradient = np.linspace(32, 224, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.TOP:
            gradient = np.linspace(224, 32, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.BOTTOM:
            gradient = np.linspace(32, 224, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        else:
            raise "Wrong background source!"

        if input_fg.shape != input_bg.shape:
            raise gr.Error("The foreground and background image should be the same!")

        result, extra_images = inferencer(
            input_fg,
            input_bg,
            prompt,
            seed=seed,
            num_inference_steps=steps,
            postfix_positive_prompt=a_prompt,
            negative_prompt=n_prompt,
            guidance_scale=cfg,
            highres_scale=highres_scale,
            highres_denoise=highres_denoise,
        )[0]
        return [result] + extra_images

    return _process_relight


if __name__ == "__main__":
    args = parse_args()

    # initialize the inferencer
    # 'runwayml/stable-diffusion-v1-5'
    inferencer = ICLightInferencer(
        sd15_model_name=args.model, checkpoint=args.checkpoint, device=args.device
    )

    # prepare examples
    subsets = os.listdir(args.fg)
    examples: dict[str, list] = {}
    for subset in subsets:
        fg = os.path.join(args.fg, subset)
        bg = os.path.join(args.bg, subset)
        examples[subset] = prepare_examples(fg, bg, num_samples=args.num_examples)

    # prepare prompts
    quick_prompts = prepare_prompts()

    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown(
                "## IC-Light (Relighting with Foreground and Background Condition)"
            )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_fg = gr.Image(
                        source="upload", type="numpy", label="Foreground", height=480
                    )
                    input_bg = gr.Image(
                        source="upload", type="numpy", label="Background", height=480
                    )
                prompt = gr.Textbox(label="Prompt")
                bg_source = gr.Radio(
                    choices=[e.value for e in BGSource],
                    value=BGSource.UPLOAD.value,
                    label="Background Source",
                    type="value",
                )
                with gr.Group():
                    with gr.Row():
                        # num_samples = gr.Slider(
                        #     label="Images", minimum=1, maximum=12, value=1, step=1
                        # )
                        seed = gr.Number(label="Seed", value=12345, precision=0)
                    # with gr.Row():
                    #     image_width = gr.Slider(
                    #         label="Image Width",
                    #         minimum=256,
                    #         maximum=1024,
                    #         value=512,
                    #         step=64,
                    #     )
                    #     image_height = gr.Slider(
                    #         label="Image Height",
                    #         minimum=256,
                    #         maximum=1024,
                    #         value=640,
                    #         step=64,
                    #     )

                relight_button = gr.Button(value="Relight")
                example_prompts = gr.Dataset(
                    samples=quick_prompts,
                    label="Prompt Quick List",
                    components=[prompt],
                )

                with gr.Accordion("Advanced options", open=False):
                    steps = gr.Slider(
                        label="Steps", minimum=1, maximum=100, value=20, step=1
                    )
                    cfg = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0,
                        maximum=32.0,
                        value=7.0,
                        step=0.01,
                    )
                    highres_scale = gr.Slider(
                        label="Highres Scale",
                        minimum=1.0,
                        maximum=3.0,
                        value=1.5,
                        step=0.01,
                    )
                    highres_denoise = gr.Slider(
                        label="Highres Denoise",
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.01,
                    )
                    a_prompt = gr.Textbox(label="Added Prompt", value="best quality")
                    n_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="lowres, bad anatomy, bad hands, cropped, worst quality",
                    )

            with gr.Column():
                result_gallery = gr.Gallery(
                    height=832, object_fit="contain", label="Outputs"
                )

        with gr.Row():
            for subset, items in examples.items():
                gr.Examples(
                    examples=items,
                    label=f"{subset.capitalize()} examples",
                    inputs=[
                        input_fg,
                        input_bg,
                        prompt,
                        bg_source,
                        seed,
                    ],
                    examples_per_page=5,
                )

        ips = [
            input_fg,
            input_bg,
            prompt,
            seed,
            steps,
            a_prompt,
            n_prompt,
            cfg,
            highres_scale,
            highres_denoise,
            bg_source,
        ]
        relight_button.click(
            fn=handle_process_relight(inferencer), inputs=ips, outputs=[result_gallery]
        )
        example_prompts.click(
            lambda x: x[0],
            inputs=example_prompts,
            outputs=prompt,
            show_progress=False,
            queue=False,
        )

    block.launch(server_name="0.0.0.0")
