import os
import torch

import folder_paths
import comfy.sd
import comfy.samplers
import comfy.sample
import latent_preview
from comfy_api.latest import io

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.model_manager import ModelManager
from ..utils.logger import logger
from ..utils.paths import get_config_file_path

_CONFIG_FILE_PATH = get_config_file_path("z_image")

class DAZImageConfig(io.ComfyNode):
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
            node_id="DAZImageConfig",
            display_name="DA ZImage Config",
            category="DALab/Image/ZImage",
            description="Configure the ZImage model params. Run first to save config.",
            is_output_node=True,
            inputs=[
                io.Combo.Input(
                    "text_encoder_model",
                    default=config.get("text_encoder_model", "z-image/qwen_3_4b.safetensors"),
                    options=text_encoder_options,
                    display_name="text_encoder_model",
                    tooltip="The Qwen text encoder model. Default: models/text_encoders",
                ),
                io.Combo.Input(
                    "vae_model",
                    default=config.get("vae_model", "flux/flux1-vae.safetensors"),
                    options=vae_options,
                    display_name="vae_model",
                    tooltip="The VAE model. ZImage works well with Flux VAE.",
                ),
                io.Combo.Input(
                    "diffusion_model",
                    default=config.get("diffusion_model", "z-image/z_image_turbo_bf16.safetensors"),
                    options=diffusion_options,
                    display_name="diffusion_model",
                    tooltip="The ZImage Turbo Diffusion Model. Default: models/diffusion_models",
                ),
                io.Int.Input(
                    "steps",
                    default=config.get("steps", 9),
                    min=1,
                    max=100,
                    tooltip="Sampling steps. ZImage Turbo usually needs fewer steps (e.g., 9).",
                    display_name="steps",
                ),
                io.Int.Input(
                    "batch_size",
                    default=config.get("batch_size", 1),
                    min=1,
                    max=64,
                    tooltip="Batch size for generation.",
                    display_name="batch_size",
                ),
                io.Float.Input(
                    "shift",
                    default=config.get("shift", 3.0),
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    tooltip="Model sampling shift parameter. Default 3.0 for ZImage.",
                    display_name="Shift",
                ),
                io.Combo.Input(
                    "sampler",
                    default=config.get("sampler", "res_multistep"),
                    options=comfy.samplers.KSampler.SAMPLERS,
                    tooltip="Sampler algorithm. 'res_multistep' is recommended.",
                    display_name="sampler",
                ),
                io.Combo.Input(
                    "scheduler",
                    default=config.get("scheduler", "simple"),
                    options=comfy.samplers.KSampler.SCHEDULERS,
                    tooltip="Noise scheduler. 'simple' is recommended.",
                    display_name="scheduler",
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
        shift,
        sampler,
        scheduler,
        easycache,
        loras
    ) -> io.NodeOutput:
        config_data = {
            "text_encoder_model": text_encoder_model,
            "vae_model": vae_model,
            "diffusion_model": diffusion_model,
            "steps": steps,
            "batch_size": batch_size,
            "shift": shift,
            "sampler": sampler,
            "scheduler": scheduler,
            "easycache": easycache,
            "loras": loras,
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()


class DAZImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DAZImage",
            display_name="DA ZImage",
            category="DALab/Image/ZImage",
            description="Generate images using ZImage model with config.",
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
                    tooltip="The prompts to generate images",
                    display_name="prompts",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0, max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for creating the noise.",
                    display_name="seed",
                ),
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
        seed: list[int]
    ) -> io.NodeOutput:
        scale_factor = 8
        target_height = height[0]
        target_width = width[0]
        latent_height = (target_height + scale_factor - 1) // scale_factor
        latent_width = (target_width + scale_factor - 1) // scale_factor
        seed = seed[0]

        if len(prompts) == 0:
            raise ValueError("no prompts provided, please check the prompts input")

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
        sampler = config.get("sampler")
        scheduler = config.get("scheduler")
        batch_size = config.get("batch_size")

        try:
            text_encoder = manager.get_text_encoder(
                paths=[text_encoder_path],
                clip_type=comfy.sd.CLIPType.LUMINA2,
            )

            all_conditionings = []
            for idx, prompt in enumerate(prompts):
                if prompt.strip() == "":
                    logger.info(f"Prompt is empty, skipping: {idx+1}")
                    continue

                tokens = text_encoder.tokenize(prompt)
                cond = text_encoder.encode_from_tokens_scheduled(tokens)
                all_conditionings.append(cond)

            if len(all_conditionings) == 0:
                raise ValueError("No valid prompts provided")

            diffusion_model = manager.get_diffusion_model(diffusion_path)
            patched_model = utils.patch_model_with_loras(diffusion_model, config)
            patched_model = utils.patch_model_easycache(patched_model, config)
            patched_model = utils.patch_model_sampling(patched_model, shift=config.get("shift"), multiplier=1.0)

            all_samples = []
            for idx, cond in enumerate(all_conditionings):
                logger.info(f"Processing diffusion {idx+1}/{len(all_conditionings)}")

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
                    cfg=1,
                    sampler_name=sampler,
                    scheduler=scheduler,
                    positive=cond,
                    negative=[],
                    latent_image=latent_image,
                    callback=callback,
                    seed=seed
                )
                all_samples.append(samples)

            vae = manager.get_vae(vae_path)

            output_images = []
            for idx, samples in enumerate(all_samples):
                logger.info(f"Decoding VAE {idx+1}/{len(all_samples)}")

                images = vae.decode(samples)
                if len(images.shape) == 5:
                    images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])

                images = utils.scale_by_width_height(
                    images, target_width, target_height, "bilinear", "center"
                )

                output_images.append(images)

            return io.NodeOutput(output_images)

        finally:
            if manager.release_after_run:
                manager.release_all()
            elif manager.offload_after_run:
                manager.offload_all()

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            config_mtime = os.path.getmtime(_CONFIG_FILE_PATH)
            global_config_mtime = os.path.getmtime(get_config_file_path("global"))
        except:
            config_mtime = 0
            global_config_mtime = 0
            
        return hash((str(kwargs),str(config_mtime),str(global_config_mtime)))