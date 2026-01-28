import os
import torch
import copy
from types import SimpleNamespace

import folder_paths
import comfy.sd
import comfy.sample
import comfy.samplers
import latent_preview
import node_helpers
from comfy_api.latest import io

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.model_manager import ModelManager
from ..utils.logger import logger

_CONFIG_FILE_PATH = utils.get_config_file_path("qwen_image_edit")

_CACHE = SimpleNamespace(
    positive = {},
    negative = {},
    ref_images = {},
)

class DAQwenImageEditConfig(io.ComfyNode):
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
            node_id="DAQwenImageEditConfig",
            display_name="DA Qwen Image Edit Config",
            category="DALab/Image/Qwen Image Edit",
            description="Configure the Qwen Image Edit model params. Run first to save config.",
            is_output_node=True,
            inputs=[
                io.Combo.Input(
                    "text_encoder_model",
                    default=config.get("text_encoder_model", "qwen/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
                    options=text_encoder_options,
                    display_name="text_encoder_model",
                    tooltip="The Qwen VL text encoder model. Default: models/text_encoders",
                ),
                io.Combo.Input(
                    "vae_model",
                    default=config.get("vae_model", "qwen/qwen_image_vae.safetensors"),
                    options=vae_options,
                    display_name="vae_model",
                    tooltip="The Qwen Image VAE model. Default: models/vae",
                ),
                io.Combo.Input(
                    "diffusion_model",
                    default=config.get("diffusion_model", "qwen/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui.safetensors"),
                    options=diffusion_options,
                    display_name="diffusion_model",
                    tooltip="The Qwen Image Edit Diffusion Model. Default: models/diffusion_models",
                ),
                io.Int.Input(
                    "steps",
                    default=config.get("steps", 4),
                    min=1,
                    max=100,
                    tooltip="Sampling steps. Default 4 for Edit model.",
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
                io.Float.Input(
                    "cfg",
                    default=config.get("cfg", 1.0),
                    min=1.0,
                    max=20.0,
                    step=0.1,
                    tooltip="CFG Scale. Default is 1.0.",
                    display_name="cfg",
                ),
                io.Float.Input(
                    "shift",
                    default=config.get("shift", 3.10),
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Model sampling shift parameter. Default 3.10.",
                    display_name="shift",
                ),
                io.Combo.Input(
                    "sampler",
                    default=config.get("sampler", "euler"),
                    options=comfy.samplers.KSampler.SAMPLERS,
                    tooltip="Sampler algorithm. 'euler' is recommended.",
                    display_name="sampler",
                ),
                io.Combo.Input(
                    "scheduler",
                    default=config.get("scheduler", "simple"),
                    options=comfy.samplers.KSampler.SCHEDULERS,
                    tooltip="Noise scheduler. 'simple' is recommended.",
                    display_name="scheduler",
                ),
                io.Float.Input(
                    "cfg_norm_strength",
                    default=config.get("cfg_norm_strength", 1.0),
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="CFG Norm Strength. Default 1.0.",
                    display_name="cfg_norm_strength",
                ),
                io.String.Input(
                    "negative_prompt",
                    default=config.get("negative_prompt", ""),
                    tooltip="Negative prompt.",
                    display_name="negative_prompt",
                ),
                io.String.Input(
                    "template",
                    default="<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    multiline=True,
                    tooltip="Template for the prompt.",
                    display_name="template",
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
        cfg,
        shift,
        sampler,
        scheduler,
        batch_size,
        cfg_norm_strength,
        negative_prompt,
        template,
        easycache,
        loras,
    ) -> io.NodeOutput:
        config_data = {
            "text_encoder_model": text_encoder_model,
            "vae_model": vae_model,
            "diffusion_model": diffusion_model,
            "steps": steps,
            "cfg": cfg,
            "shift": shift,
            "sampler": sampler,
            "scheduler": scheduler,
            "negative_prompt": negative_prompt,
            "template": template,
            "batch_size": batch_size,
            "cfg_norm_strength": cfg_norm_strength,
            "easycache": easycache,
            "loras": loras
        }

        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()


class DAQwenImageEdit(io.ComfyNode):
    @classmethod
    def reset_cache(cls):
        _CACHE.positive = {}
        _CACHE.negative = {}
        _CACHE.ref_images = {}

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DAQwenImageEdit",
            display_name="DA Qwen Image Edit",
            category="DALab/Image/Qwen Image Edit",
            description="Edit images using the Qwen Image Edit model.",
            is_input_list=True,
            inputs=[
                io.Int.Input(
                    "width",
                    default=1920,
                    min=16, 
                    max=4096, 
                    step=8,
                    display_mode=io.NumberDisplay.number,
                    display_name="width",
                    tooltip="Select the width of the images",
                ),
                io.Int.Input(
                    "height",
                    default=1080,
                    min=16, 
                    max=4096, 
                    step=8,
                    display_mode=io.NumberDisplay.number,
                    display_name="height",
                    tooltip="Select the height of the images",
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
                    min=0,
                    step=1,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    display_name="seed",
                    tooltip="Select the seed for the images",
                ),
                io.Autogrow.Input(
                    "images",
                    display_name="images",
                    tooltip="Up to 10 reference images.",
                    optional=True,
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
                    display_name="images",
                    tooltip="The generated images",
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
        scale_factor = 8
        target_height = height[0]
        target_width = width[0]
        latent_height = (target_height + scale_factor - 1) // scale_factor
        latent_width = (target_width + scale_factor - 1) // scale_factor
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

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)
        manager = ModelManager()

        text_encoder_path = folder_paths.get_full_path_or_raise(
            "text_encoders", config.get("text_encoder_model")
        )
        vae_path = folder_paths.get_full_path_or_raise(
            "vae", config.get("vae_model")
        )
        diffusion_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", config.get("diffusion_model")
        )

        steps = config.get("steps")
        cfg = config.get("cfg")
        sampler_name = config.get("sampler")
        scheduler = config.get("scheduler")
        batch_size = config.get("batch_size")
        negative_prompt = config.get("negative_prompt")
        template = config.get("template")

        cls.reset_cache()

        try:
            vae = manager.get_vae(vae_path)

            all_ref_data = []
            for idx, batch_input in enumerate(batch_inputs):
                prompt = batch_input["prompt"]
                ref_images = batch_input["images"]
                ref_vl_images, ref_latents, ref_vl_image_prompts = cls.encode_ref_images(vae, ref_images)
                all_ref_data.append((prompt, ref_images, ref_vl_images, ref_latents, ref_vl_image_prompts))

            text_encoder = manager.get_text_encoder(
                paths=[text_encoder_path],
                clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
            )

            all_conditionings = []
            for prompt, ref_images, ref_vl_images, ref_latents, ref_vl_image_prompts in all_ref_data:
                cache_keys = [prompt['cache_key']] + [image['cache_key'] for image in ref_images.values()]
                cache_key = "_".join(cache_keys)

                if cache_key in _CACHE.positive:
                    logger.info(f"QwenImage Edit prompt cache hit: {cache_key}")
                    positive = copy.copy(_CACHE.positive[cache_key])
                else:
                    full_prompt = " ".join(ref_vl_image_prompts) + prompt["value"]
                    tokens = text_encoder.tokenize(
                        full_prompt,
                        images=None if len(ref_vl_images) == 0 else ref_vl_images,
                        llama_template=template
                    )
                    positive = text_encoder.encode_from_tokens_scheduled(tokens)
                    _CACHE.positive[cache_key] = positive

                if cfg > 1.0:
                    if cache_key in _CACHE.negative:
                        logger.info(f"QwenImage Edit negative prompt cache hit: {cache_key}")
                        negative = copy.copy(_CACHE.negative[cache_key])
                    else:
                        negative_tokens = text_encoder.tokenize(negative_prompt)
                        negative = text_encoder.encode_from_tokens_scheduled(negative_tokens)
                        _CACHE.negative[cache_key] = negative
                else:
                    negative = []

                if len(ref_latents) > 0:
                    positive = node_helpers.conditioning_set_values(
                        positive,
                        {"reference_latents": ref_latents},
                    )
                    if len(negative) > 0:
                        negative = node_helpers.conditioning_set_values(
                            negative,
                            {"reference_latents": ref_latents}
                        )

                positive = node_helpers.conditioning_set_values(
                    positive, {"reference_latents_method": "index_timestep_zero"}
                )
                if len(negative) > 0:
                    negative = node_helpers.conditioning_set_values(
                        negative, {"reference_latents_method": "index_timestep_zero"}
                    )

                all_conditionings.append((positive, negative))

            diffusion_model = manager.get_diffusion_model(diffusion_path)
            patched_model = utils.patch_model_with_loras(diffusion_model, config)
            patched_model = utils.patch_model_easycache(patched_model, config)
            patched_model = utils.patch_model_sampling(patched_model, shift=config.get("shift"), multiplier=1.0)
            patched_model = utils.patch_cfg_norm(patched_model, strength=config.get("cfg_norm_strength"))

            all_samples = []
            for idx, (positive, negative) in enumerate(all_conditionings):
                logger.info(f"QwenImage Edit processing diffusion {idx+1}/{len(all_conditionings)}")

                latent_image = torch.zeros(
                    [batch_size, 16, latent_height, latent_width],
                    device=patched_model.load_device
                )
                latent_image = comfy.sample.fix_empty_latent_channels(patched_model, latent_image)
                noise = comfy.sample.prepare_noise(latent_image, seed)

                callback = latent_preview.prepare_callback(patched_model, steps)
                samples = comfy.sample.sample(
                    patched_model,
                    noise,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent_image=latent_image,
                    callback=callback,
                    seed=seed
                )
                all_samples.append(samples)

            output_images = []
            for idx, samples in enumerate(all_samples):
                logger.info(f"QwenImage Edit decoding VAE {idx+1}/{len(all_samples)}")

                decoded_images = vae.decode(samples)
                if len(decoded_images.shape) == 5:
                    decoded_images = decoded_images.reshape(
                        -1, decoded_images.shape[2], decoded_images.shape[3], decoded_images.shape[4]
                    )

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
        ref_vl_images = []
        ref_latents = []
        ref_vl_image_prompts = []

        latent_pixels_total = int(1024 * 1024)
        vl_pixels_total = int(384 * 384)

        for image_idx, image in enumerate(ref_images.values()):
            if image['cache_key'] in _CACHE.ref_images:
                logger.info(f"QwenImage Edit ref image cache hit: {image['cache_key']}")

                ref_vl_images.append(_CACHE.ref_images[image['cache_key']]['vl_image'])
                ref_latents.append(_CACHE.ref_images[image['cache_key']]['latent'])
                ref_vl_image_prompts.append(_CACHE.ref_images[image['cache_key']]['vl_image_prompt'])
            else:
                if image['value'] is not None:
                    vl_scaled_image = utils.scale_by_total_pixels(image['value'], vl_pixels_total, "area", "disabled")
                    ref_vl_images.append(vl_scaled_image)

                    latent_scaled_image = utils.scale_by_total_pixels(image['value'], latent_pixels_total, "area", "disabled")
                    latent = vae.encode(latent_scaled_image)
                    ref_latents.append(latent)

                    image_prompt = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(image_idx + 1)
                    ref_vl_image_prompts.append(image_prompt)

                    _CACHE.ref_images[image['cache_key']] = {
                        'vl_image': vl_scaled_image,
                        'latent': latent,
                        'vl_image_prompt': image_prompt
                    }

        return ref_vl_images, ref_latents, ref_vl_image_prompts

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            config_mtime = os.path.getmtime(_CONFIG_FILE_PATH)
            global_config_mtime = os.path.getmtime(utils.get_config_file_path("global"))
        except:
            config_mtime = 0
            global_config_mtime = 0

        return hash((str(kwargs), str(config_mtime), str(global_config_mtime)))