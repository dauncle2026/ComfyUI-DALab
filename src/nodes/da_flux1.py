import torch
import os

import folder_paths
import comfy.sd
import comfy.samplers
import comfy.sample
import node_helpers
import latent_preview
from comfy_api.latest import io

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.model_manager import ModelManager
from ..utils.logger import logger
from ..utils.paths import get_config_file_path

_CONFIG_FILE_PATH = get_config_file_path("flux1")

class DAFlux1Config(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        text_encoder_options = folder_paths.get_filename_list("text_encoders")
        vae_options = folder_paths.get_filename_list("vae")
        diffusion_options = folder_paths.get_filename_list("diffusion_models")
        lora_options = folder_paths.get_filename_list("loras")

        config = ConfigLoader(_CONFIG_FILE_PATH,strict=False)

        easycache = utils.dynamic_combo_easycache(config)
        loras = utils.dynamic_combo_loras(config, lora_options)

        return io.Schema(
            node_id="DAFlux1Config",
            display_name="DA Flux1 Config",
            category="DALab/Image/Flux1",
            description="Configure the Flux1 model,Run first to save the config",
            is_output_node=True,
            inputs=[
                io.Combo.Input(
                    "clip_model",
                    default=config.get("clip_model", "flux/clip_l.safetensors"),
                    options=text_encoder_options,
                    display_name="clip_model",
                    tooltip="The CLIP text encoder model. Default path: models/text_encoders",
                ),
                io.Combo.Input(
                    "t5_model",
                    default=config.get("t5_model", "flux/t5xxl_fp8_e4m3fn_scaled.safetensors"),
                    options=text_encoder_options,
                    display_name="t5_model",
                    tooltip="The T5 text encoder model (fp8 recommended). Default path: models/text_encoders",
                ),
                io.Combo.Input(
                    "vae_model",
                    default=config.get("vae_model", "flux/flux1-vae.safetensors"),
                    options=vae_options,
                    display_name="vae_model",
                    tooltip="The VAE model used to decode latents into pixels. Default path: models/vae",
                ),
                io.Combo.Input(
                    "diffusion_model",
                    default=config.get("diffusion_model", "flux/flux1-dev-fp8.safetensors"),
                    options=diffusion_options,
                    display_name="diffusion_model",
                    tooltip="The core Flux diffusion model (UNet/Transformer). Default path: models/diffusion_models",
                ),
                io.Int.Input(
                    "steps",
                    default=config.get("steps", 20),
                    min=1,
                    max=100,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Number of sampling steps. Higher values improve detail but slow down generation. Flux works well around 20.",
                    display_name="steps",
                ),
                io.Int.Input(
                    "batch_size",
                    default=config.get("batch_size", 1),
                    min=1,
                    max=24,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Number of images to generate in a single batch.",
                    display_name="batch_size",
                ),
                io.Float.Input(
                    "guidance",
                    default=config.get("guidance", 3.5),
                    min=1,
                    max=10,
                    step=0.1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Guidance scale. Controls how closely the image adheres to the prompt. Default 3.5 is recommended for Flux.",
                    display_name="guidance",
                ),
                io.Combo.Input(
                    "sampler",
                    default=config.get("sampler", "euler"),
                    options=comfy.samplers.KSampler.SAMPLERS,
                    tooltip="The sampling algorithm (e.g., euler, dpm++).",
                    display_name="sampler",
                ),
                io.Combo.Input(
                    "scheduler",
                    default=config.get("scheduler", "simple"),
                    options=comfy.samplers.KSampler.SCHEDULERS,
                    tooltip="The noise schedule strategy. 'Simple' or 'Beta' are common for Flux.",
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
        clip_model: str,
        t5_model: str,
        vae_model: str,
        diffusion_model: str,
        steps: int,
        batch_size: int,
        guidance: float,
        sampler: str,
        scheduler: str,
        easycache: dict,
        loras: dict,
    ) -> io.NodeOutput:
        config_data = {
            "clip_model": clip_model,
            "t5_model": t5_model,
            "vae_model": vae_model,
            "diffusion_model": diffusion_model,
            "steps": steps,
            "batch_size": batch_size,
            "guidance": guidance,
            "sampler": sampler,
            "scheduler": scheduler,
            "easycache": easycache,
            "loras": loras,
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()

class DAFlux1(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DAFlux1",
            display_name="DA Flux1",
            category="DALab/Image/Flux1",
            description="Generate images using the Flux1 model",
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
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
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
        seed: list[int],
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

        clip_path = folder_paths.get_full_path_or_raise(
            "text_encoders",
            config.get("clip_model")
        )
        t5_path = folder_paths.get_full_path_or_raise(
            "text_encoders",
            config.get("t5_model")
        )
        vae_path = folder_paths.get_full_path_or_raise(
            "vae",
            config.get("vae_model")
        )
        diffusion_path = folder_paths.get_full_path_or_raise(
            "diffusion_models",
            config.get("diffusion_model")
        )

        steps = config.get("steps")
        sampler = config.get("sampler")
        scheduler = config.get("scheduler")
        batch_size = config.get("batch_size")
        guidance = config.get("guidance")

        try:
            text_encoder = manager.get_text_encoder(
                paths=[clip_path, t5_path],
                clip_type=comfy.sd.CLIPType.FLUX,
            )

            all_conditionings = []
            for idx, prompt in enumerate(prompts):
                if prompt.strip() == "":
                    logger.info(f"Prompt is empty, skipping: {idx+1}")
                    continue

                tokens = text_encoder.tokenize(prompt)
                cond = text_encoder.encode_from_tokens_scheduled(tokens)
                node_helpers.conditioning_set_values(cond, {"guidance": guidance})
                all_conditionings.append(cond)

            if len(all_conditionings) == 0:
                raise ValueError("No valid prompts provided")

            diffusion_model = manager.get_diffusion_model(diffusion_path)
            patched_model = utils.patch_model_with_loras(diffusion_model, config)
            patched_model = utils.patch_model_easycache(patched_model, config)

            all_samples = []
            for idx, cond in enumerate(all_conditionings):
                logger.info(f"Processing diffusion {idx+1}/{len(all_conditionings)}")

                latent_image = torch.zeros(
                    [batch_size, 16, latent_height, latent_width],
                    device=patched_model.load_device
                )
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
                    disable_pbar=False,
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
            global_config_mtime = os.path.getmtime(utils.get_config_file_path("global"))
        except:
            config_mtime = 0
            global_config_mtime = 0
            
        return hash((str(kwargs),str(config_mtime),str(global_config_mtime)))