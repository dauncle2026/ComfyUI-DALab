import os
import copy
import torch
import math
from types import SimpleNamespace

import folder_paths
import comfy.sd
import comfy.sample
import comfy.samplers
import node_helpers
import latent_preview
from comfy_api.latest import io

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.model_manager import ModelManager
from ..utils.logger import logger
from .da_flux2 import get_schedule
from ..utils.paths import get_config_file_path

_CONFIG_FILE_PATH = get_config_file_path("flux2_klein")

_CACHE = SimpleNamespace(
    positive = {},
    ref_latents = {},
)

class DAFlux2KleinConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        text_encoder_options = folder_paths.get_filename_list("text_encoders")
        vae_options = folder_paths.get_filename_list("vae")
        diffusion_options = folder_paths.get_filename_list("diffusion_models")
        lora_options = folder_paths.get_filename_list("loras")

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)

        loras = utils.dynamic_combo_loras(config, lora_options)
        easycache = utils.dynamic_combo_easycache(config)

        return io.Schema(
            node_id="DAFlux2KleinConfig",
            display_name="DA Flux2 Klein Config",
            category="DALab/Image/Flux2 Klein",
            description="Configure the Flux2 Klein model params. Run first to save config.",
            is_output_node=True,
            inputs=[
                io.Combo.Input(
                    "text_encoder_model",
                    default=config.get("text_encoder_model", "flux/qwen_3_8b_fp8mixed.safetensors"),
                    options=text_encoder_options,
                    display_name="text_encoder_model",
                    tooltip="The Flux2 Klein Text Encoder (Mistral). Default: models/text_encoders",
                ),
                io.Combo.Input(
                    "vae_model",
                    default=config.get("vae_model", "flux/flux2-vae.safetensors"),
                    options=vae_options,
                    display_name="vae_model",
                    tooltip="The Flux2 Klein VAE model. Default: models/vae",
                ),
                io.Combo.Input(
                    "diffusion_model",
                    default=config.get("diffusion_model", "flux-klein/flux-2-klein-9b-fp8.safetensors"),
                    options=diffusion_options,
                    display_name="diffusion_model",
                    tooltip="The Flux2 Klein Diffusion Model. Default: models/diffusion_models",
                ),
                io.Int.Input(
                    "steps",
                    default=config.get("step", 4),
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Sampling steps. Default 20 for Flux2 Klein.",
                    display_name="steps",
                ),
                io.Int.Input(
                    "batch_size",
                    default=config.get("batch_size", 1),
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Batch size for generation.",
                    display_name="batch_size",
                ),
                io.Combo.Input(
                    "sampler",
                    default=config.get("sampler", "euler"),
                    options=comfy.samplers.KSampler.SAMPLERS,
                    tooltip="Sampler algorithm. 'euler' is recommended.",
                    display_name="sampler",
                ),
                io.DynamicCombo.Input(
                    "easycache",
                    options=easycache,
                    display_name="easycache",
                    tooltip="Enable EasyCache to improve performance by caching model states.",
                ),
                io.DynamicCombo.Input(
                    "loras",
                    options=loras,
                    display_name="loras",
                    tooltip="Apply one or more LoRAs to the generation.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        text_encoder_model,
        vae_model,
        diffusion_model,
        steps,
        batch_size,
        sampler,
        easycache,
        loras
    ) -> io.NodeOutput:
        config_data = {
            "text_encoder_model": text_encoder_model,
            "vae_model": vae_model,
            "diffusion_model": diffusion_model,
            "steps": steps,
            "batch_size": batch_size,
            "sampler": sampler,
            "easycache": easycache,
            "loras": loras,
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()


class DAFlux2Klein(io.ComfyNode):
    @classmethod
    def reset_cache(cls):
        _CACHE.positive = {}
        _CACHE.ref_latents = {}

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DAFlux2Klein",
            display_name="DA Flux2 Klein",
            category="DALab/Image/Flux2 Klein",
            description="Generate images using the Flux2 Klein model.",
            is_input_list=True,
            inputs=[
                io.Int.Input(
                    "width",
                    default=1920,
                    min=16, 
                    max=4096, 
                    step=8,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Select the width of the images",
                    display_name="width",
                ),
                io.Int.Input(
                    "height",
                    default=1080,
                    min=16, 
                    max=4096, 
                    step=8,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Select the height of the images",
                    display_name="height",
                ),
                io.String.Input(
                    "prompts",
                    multiline=True,
                    default="A beautiful girl",
                    display_name="prompts",
                    tooltip="Enter the prompts for the images",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0, max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    display_name="seed",
                    tooltip="Select the seed for the images",
                ),
                io.Autogrow.Input(
                    id="images",
                    optional=True,
                    tooltip="Up to 10 reference images.",
                    display_name="images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input(
                            "ref_images",
                            optional=True,
                            tooltip="Up to 10 reference images."
                        ),
                        names=["image1", "image2", "image3", "image4", "image5", "image6", "image7", "image8", "image9", "image10"],
                        min=1,
                    ),
                )
            ],
            outputs=[
                io.Image.Output(
                    "images", 
                    is_output_list=True, 
                    tooltip="The generated images", 
                    display_name="images",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        width: list[int],
        height: list[int],
        prompts: list[str],
        seed: list[int],
        images=None,
    ) -> io.NodeOutput:
        scale_factor = 16
        target_height = height[0]
        target_width = width[0]
        latent_height = (target_height + scale_factor - 1) // scale_factor
        latent_width = (target_width + scale_factor - 1) // scale_factor
        aligned_height = latent_height * scale_factor
        aligned_width = latent_width * scale_factor

        seed = seed[0]

        batch_inputs = utils.inputs_to_batch(
            defaults={
                "prompt": "",
            },
            nested_inputs={
                "images": images,
            },
            prompt=prompts
        )

        if len(batch_inputs) == 0:
            raise Exception("[DALab] Flux2 Klein inputs is empty")

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)
        manager = ModelManager()

        text_encoder_path = folder_paths.get_full_path_or_raise(
            "text_encoders", config.get("text_encoder_model")
        )
        vae_path = folder_paths.get_full_path_or_raise(
            "vae", config.get("vae_model")
        )
        diffusion_model_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", config.get("diffusion_model")
        )

        steps = config.get("steps")
        sampler_name = config.get("sampler")

        seq_len = (aligned_width * aligned_height / (scale_factor * scale_factor))
        sigmas = get_schedule(steps, round(seq_len))
        sampler = comfy.samplers.sampler_object(sampler_name)

        cls.reset_cache()

        try:
            text_encoder = manager.get_text_encoder(
                paths=[text_encoder_path],
                clip_type=comfy.sd.CLIPType.FLUX2,
            )

            all_conditionings = []
            for idx, batch_input in enumerate(batch_inputs):
                prompt = batch_input["prompt"]
                ref_images = batch_input["images"]

                if prompt['cache_key'] in _CACHE.positive:
                    logger.info(f"Flux2 Klein prompt cache hit: {prompt['cache_key']}")
                    positive = copy.copy(_CACHE.positive[prompt['cache_key']])
                else:
                    if prompt['value'] is None:
                        raise Exception("[DALab] Flux2 Klein prompt is empty")

                    tokens = text_encoder.tokenize(prompt['value'])
                    positive = text_encoder.encode_from_tokens_scheduled(tokens)
                    _CACHE.positive[prompt['cache_key']] = positive

                all_conditionings.append((positive, ref_images))

            vae = manager.get_vae(vae_path)

            all_prepared_conds = []
            for positive, ref_images in all_conditionings:
                ref_latents = cls.encode_ref_images(vae, ref_images)
                if len(ref_latents) > 0:
                    positive = node_helpers.conditioning_set_values(
                        positive,
                        {"reference_latents": ref_latents},
                    )
                all_prepared_conds.append(positive)

            diffusion_model = manager.get_diffusion_model(diffusion_model_path)
            patched_model = utils.patch_model_with_loras(diffusion_model, config)
            patched_model = utils.patch_model_easycache(patched_model, config)

            all_samples = []
            for idx, positive in enumerate(all_prepared_conds):
                logger.info(f"Flux2 Klein processing diffusion {idx+1}/{len(all_prepared_conds)}")

                latent_image = torch.zeros(
                    [1, 128, latent_height, latent_width],
                    device=patched_model.load_device
                )
                noise = comfy.sample.prepare_noise(latent_image, seed)

                callback = latent_preview.prepare_callback(patched_model, steps)

                guider = comfy.samplers.CFGGuider(patched_model)
                guider.inner_set_conds({"positive": positive})

                samples = guider.sample(
                    noise,
                    latent_image,
                    sampler,
                    sigmas,
                    callback=callback,
                    seed=seed
                )
                all_samples.append(samples)

            output_images = []
            for idx, samples in enumerate(all_samples):
                logger.info(f"Flux2 Klein decoding VAE {idx+1}/{len(all_samples)}")

                decoded_images = vae.decode(samples)
                decoded_images = utils.scale_by_width_height(
                    decoded_images, target_width, target_height, "bilinear", "center"
                )
                output_images.append(decoded_images)

            return io.NodeOutput(output_images)

        finally:
            if manager.release_after_run:
                manager.release_all()
            elif manager.offload_after_run:
                manager.offload_all()

    @classmethod
    def encode_ref_images(cls, vae, ref_images):
        ref_latents = []
        latent_pixels_total = int(1024 * 1024)

        for image in ref_images.values():
            if image['cache_key'] in _CACHE.ref_latents:
                logger.info(f"Flux2 Klein ref image cache hit: {image['cache_key']}")
                ref_latents.append(_CACHE.ref_latents[image['cache_key']])
            else:
                if image['value'] is not None:
                    scaled_image = utils.scale_by_total_pixels(image['value'], latent_pixels_total, "lanczos", "center")
                    latent = vae.encode(scaled_image)

                    ref_latents.append(latent)
                    _CACHE.ref_latents[image['cache_key']] = latent

        return ref_latents

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            config_mtime = os.path.getmtime(_CONFIG_FILE_PATH)
            global_config_mtime = os.path.getmtime(get_config_file_path("global"))
        except:
            config_mtime = 0
            global_config_mtime = 0

        return hash((str(kwargs), str(config_mtime), str(global_config_mtime)))