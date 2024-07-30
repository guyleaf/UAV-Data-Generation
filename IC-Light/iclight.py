import math
import os
from typing import Optional, Union

import numpy as np
import PIL.Image as Image
import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer
from utils import (
    collect_images_from_dir,
    pad_to_divisible,
)

IMAGE_TYPE = Union[str, np.ndarray, Image.Image]
IMAGES_TYPE = Union[list[str], list[np.ndarray], list[Image.Image]]


def _hook_unet_forward(unet: UNet2DConditionModel):
    unet_forward = unet.forward

    def _unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs["cross_attention_kwargs"] = {}
        return unet_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

    unet.forward = _unet_forward
    return unet


def _update_unet(unet: UNet2DConditionModel):
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(
            12,
            unet.conv_in.out_channels,
            unet.conv_in.kernel_size,
            unet.conv_in.stride,
            unet.conv_in.padding,
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

    unet = _hook_unet_forward(unet)
    return unet


# Reference: https://github.com/lllyasviel/IC-Light/blob/788687452a2bad59633a401281c8aee91bdd3750/gradio_demo_bg.py
class ICLightInferencer:
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        sd15_model_name: str = "stablediffusionapi/realistic-vision-v51",
        device: Union[str, torch.device] = "cuda:0",
    ) -> None:
        self._device = device
        # self.distributed_state = PartialState()

        if checkpoint is None:
            checkpoint = hf_hub_download(
                repo_id="lllyasviel/ic-light",
                filename="iclight_sd15_fbc.safetensors",
                revision="9cad1878695f546a7fb9eaca14e2a89131ba5ffe",
            )
        self._prepare_pipeline(sd15_model_name, checkpoint)

    @property
    def device(self):
        return self._device

    def _prepare_model(self, model_name: str, checkpoint: str):
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(model_name)
        text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(model_name)
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(model_name)
        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(model_name)

        # modify the network
        unet = _update_unet(unet)

        # SDP
        unet.set_attn_processor(AttnProcessor2_0())
        vae.set_attn_processor(AttnProcessor2_0())

        text_encoder = text_encoder.to(device=self.device)
        vae = vae.to(device=self.device)
        unet = unet.to(device=self.device)

        # load the checkpoint
        sd_offset = load_file(checkpoint, device=self.device)
        sd_origin = unet.state_dict()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        unet.load_state_dict(sd_merged)
        del sd_offset, sd_origin, sd_merged

        self.text_encoder = text_encoder.to(dtype=torch.float16)
        self.vae = vae.to(dtype=torch.bfloat16)
        self.unet = unet.to(dtype=torch.float16)

    def _prepare_sampler(self):
        self.dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1,
        )

    def _prepare_pipeline(self, model_name: str, checkpoint: str):
        self._prepare_model(model_name, checkpoint)
        self._prepare_sampler()

        self.t2i_pipe = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.dpmpp_2m_sde_karras_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None,
        )

        self.i2i_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.dpmpp_2m_sde_karras_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None,
        )

    @torch.inference_mode()
    def pytorch2numpy(self, imgs: torch.Tensor, quant: bool = True):
        results = []
        for x in imgs:
            y = x.movedim(0, -1)

            if quant:
                y = y * 127.5 + 127.5
                y = y.float().cpu().numpy().clip(0, 255).astype(np.uint8)
            else:
                y = y * 0.5 + 0.5
                y = y.float().cpu().numpy().clip(0, 1).astype(np.float32)

            results.append(y)
        return results

    @torch.inference_mode()
    def numpy2pytorch(self, imgs: np.ndarray):
        h = (
            torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
        )  # so that 127 must be strictly 0.0
        h = h.movedim(-1, 1)
        return h

    def _encode_prompt_inner(self, txt: str):
        max_length = self.tokenizer.model_max_length
        chunk_length = self.tokenizer.model_max_length - 2
        id_start = self.tokenizer.bos_token_id
        id_end = self.tokenizer.eos_token_id
        id_pad = id_end

        def pad(x, p, i):
            return x[:i] if len(x) >= i else x + [p] * (i - len(x))

        tokens = self.tokenizer(txt, truncation=False, add_special_tokens=False)[
            "input_ids"
        ]
        chunks = [
            [id_start] + tokens[i : i + chunk_length] + [id_end]
            for i in range(0, len(tokens), chunk_length)
        ]
        chunks = [pad(ck, id_pad, max_length) for ck in chunks]

        token_ids = torch.tensor(chunks).to(device=self.device, dtype=torch.int64)
        conds = self.text_encoder(token_ids).last_hidden_state

        return conds

    def _encode_prompt_pair(self, positive_prompt: str, negative_prompt: str):
        c = self._encode_prompt_inner(positive_prompt)
        uc = self._encode_prompt_inner(negative_prompt)

        c_len = float(len(c))
        uc_len = float(len(uc))
        max_count = max(c_len, uc_len)
        c_repeat = int(math.ceil(max_count / c_len))
        uc_repeat = int(math.ceil(max_count / uc_len))
        max_chunk = max(len(c), len(uc))

        c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
        uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

        c = torch.cat([p[None, ...] for p in c], dim=1)
        uc = torch.cat([p[None, ...] for p in uc], dim=1)

        return c, uc

    def _collect_images(
        self,
        fg: Union[IMAGE_TYPE, IMAGES_TYPE],
        bg: Union[IMAGE_TYPE, IMAGES_TYPE],
    ):
        print("Collecting images...")

        def _collect_images(data: Union[IMAGE_TYPE, IMAGES_TYPE]):
            if isinstance(data, str):
                return collect_images_from_dir(data)
            elif isinstance(data, (np.ndarray, Image.Image)):
                return [data]
            return data

        fg_images = _collect_images(fg)
        bg_images = _collect_images(bg)
        fg_bg_pairs = list(zip(fg_images, bg_images))

        if isinstance(fg, str) and isinstance(bg, str):
            # make sure all image pairs are the same relative path
            assert all(
                os.path.relpath(fg_, fg) == os.path.relpath(bg_, bg)
                for fg_, bg_ in fg_bg_pairs
            )
        return fg_bg_pairs

    def _preprocess(self, fg_bg_pairs: list[tuple[IMAGE_TYPE, IMAGE_TYPE]]):
        def _load_image(data: IMAGE_TYPE):
            if isinstance(data, str):
                return Image.open(data)
            elif isinstance(data, np.ndarray):
                return Image.fromarray(data)
            return data

        divisor: int = 2 ** (len(self.vae.config.block_out_channels) - 1)
        for fg, bg in fg_bg_pairs:
            fg = _load_image(fg)
            bg = _load_image(bg)
            assert fg.size == bg.size, "Support the same image size only."

            fg = pad_to_divisible(fg, divisor=divisor)
            bg = pad_to_divisible(bg, divisor=divisor)

            fg = np.array(fg)
            bg = np.array(bg)
            yield fg, bg

    @torch.inference_mode()
    def _forward(
        self,
        fg: np.ndarray,
        bg: np.ndarray,
        positive_prompt: str,
        negative_prompt: str,
        generator: torch.Generator,
        num_inference_steps: int,
        guidance_scale: float,
        highres_scale: float,
        highres_denoise: float,
    ):
        image_height, image_width = bg.shape[:2]

        # fg bg embeddings
        concat_conds = self.numpy2pytorch([fg, bg]).to(
            device=self.vae.device, dtype=self.vae.dtype
        )
        concat_conds = (
            self.vae.encode(concat_conds).latent_dist.mode()
            * self.vae.config.scaling_factor
        )
        concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

        # prompt embeddings
        conds, unconds = self._encode_prompt_pair(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
        )

        latents = (
            self.t2i_pipe(
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="latent",
                guidance_scale=guidance_scale,
                cross_attention_kwargs={"concat_conds": concat_conds.clone()},
            ).images.to(self.vae.dtype)
            / self.vae.config.scaling_factor
        )

        # pixels = self.vae.decode(latents).sample
        # pixels = self.pytorch2numpy(pixels)
        # pixels = [
        #     resize_without_crop(
        #         image=p,
        #         target_width=int(round(image_width * highres_scale / 64.0) * 64),
        #         target_height=int(round(image_height * highres_scale / 64.0) * 64),
        #     )
        #     for p in pixels
        # ]

        # pixels = self.numpy2pytorch(pixels).to(
        #     device=self.vae.device, dtype=self.vae.dtype
        # )
        # latents = (
        #     self.vae.encode(pixels).latent_dist.mode() * self.vae.config.scaling_factor
        # )
        # latents = latents.to(device=self.unet.device, dtype=self.unet.dtype)

        # image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

        # concat_conds = self.numpy2pytorch([fg, bg]).to(
        #     device=self.vae.device, dtype=self.vae.dtype
        # )
        # concat_conds = (
        #     self.vae.encode(concat_conds).latent_dist.mode()
        #     * self.vae.config.scaling_factor
        # )
        # concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

        # latents = (
        #     self.i2i_pipe(
        #         image=latents,
        #         strength=highres_denoise,
        #         prompt_embeds=conds,
        #         negative_prompt_embeds=unconds,
        #         width=image_width,
        #         height=image_height,
        #         num_inference_steps=int(round(num_inference_steps / highres_denoise)),
        #         num_images_per_prompt=num_samples,
        #         generator=self.generator,
        #         output_type="latent",
        #         guidance_scale=guidance_scale,
        #         cross_attention_kwargs={"concat_conds": concat_conds.clone()},
        #     ).images.to(self.vae.dtype)
        #     / self.vae.config.scaling_factor
        # )

        pixels = self.vae.decode(latents).sample
        pixels = self.pytorch2numpy(pixels)
        return pixels[0]

    def __call__(
        self,
        input_fg: Union[IMAGE_TYPE, IMAGES_TYPE],
        input_bg: Union[IMAGE_TYPE, IMAGES_TYPE],
        prompt: str,
        seed: int = 12345,
        num_inference_steps: int = 20,
        postfix_positive_prompt: str = "best quality",
        negative_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality",
        guidance_scale: float = 7.0,
        highres_scale: float = 1.5,
        highres_denoise: float = 0.5,
    ):
        fg_bg_pairs = self._collect_images(input_fg, input_bg)
        positive_prompt = prompt + ", " + postfix_positive_prompt

        iterator = self._preprocess(fg_bg_pairs)
        results = []
        for fg, bg in iterator:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            result = self._forward(
                fg,
                bg,
                positive_prompt,
                negative_prompt,
                generator,
                num_inference_steps,
                guidance_scale,
                highres_scale,
                highres_denoise,
            )
            results.append((result, [fg, bg]))
        return results
