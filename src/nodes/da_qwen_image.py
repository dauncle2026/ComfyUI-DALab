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

_CONFIG_FILE_PATH = get_config_file_path("qwen_image")

class DAQwenImageConfig(io.ComfyNode):
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
            node_id="DAQwenImageConfig",
            display_name="DA Qwen Image Config",
            category="DALab/Image/Qwen Image",
            description="Configure the Qwen Image model params. Run first to save config.",
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
                    default=config.get("diffusion_model", "qwen/qwen-image-2512-fp8.safetensors"),
                    options=diffusion_options,
                    display_name="diffusion_model",
                    tooltip="The Qwen Image Diffusion Model (fp8). Default: models/diffusion_models",
                ),
                io.Int.Input(
                    "steps",
                    default=config.get("steps", 5),
                    min=1,
                    max=100,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Sampling steps. Default 5 for Lightning LoRA.",
                    display_name="steps",
                ),
                io.Int.Input(
                    "batch_size",
                    default=config.get("batch_size", 1),
                    min=1,
                    max=64,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Batch size for generation.",
                    display_name="batch_size",
                ),
                io.Float.Input(
                    "cfg",
                    default=config.get("cfg", 1.0),
                    min=1.0,
                    max=20.0,
                    step=0.1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="CFG Scale. Default is 1.0.",
                    display_name="cfg",
                ),
                io.Float.Input(
                    "shift",
                    default=config.get("shift", 3.10),
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Model sampling shift parameter. Default 3.10 for Qwen Image.",
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
                io.String.Input(
                    "negative_prompt",
                    default=config.get(
                        "negative_prompt", 
                        "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲"
                    ),
                    multiline=True,
                    tooltip="The negative prompt to generate images",
                    display_name="negative_prompt",
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
                    tooltip="Select the number of LoRAs to apply to the generation.",
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
        cfg,
        shift,
        sampler, scheduler,
        negative_prompt,
        easycache, 
        loras
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
            "batch_size": batch_size,
            "negative_prompt": negative_prompt,
            "easycache": easycache,
            "loras": loras,
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()


class DAQwenImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DAQwenImage",
            display_name="DA Qwen Image",
            category="DALab/Image/Qwen Image",
            description="Generate images using the Qwen Image model.",
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
        cfg = config.get("cfg")
        sampler = config.get("sampler")
        scheduler = config.get("scheduler")
        batch_size = config.get("batch_size")
        negative_prompt = config.get("negative_prompt")

        try:
            text_encoder = manager.get_text_encoder(
                paths=[text_encoder_path],
                clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
            )

            all_conditionings = []
            for idx, prompt in enumerate(prompts):
                if prompt.strip() == "":
                    logger.info(f"Prompt is empty, skipping: {idx+1}")
                    continue

                tokens = text_encoder.tokenize(prompt)
                positive_cond = text_encoder.encode_from_tokens_scheduled(tokens)

                if cfg > 1.0:
                    negative_tokens = text_encoder.tokenize(negative_prompt)
                    negative_cond = text_encoder.encode_from_tokens_scheduled(negative_tokens)
                else:
                    negative_cond = []

                all_conditionings.append((positive_cond, negative_cond))

            if len(all_conditionings) == 0:
                raise ValueError("No valid prompts provided")

            diffusion_model = manager.get_diffusion_model(
                diffusion_path,
                model_options={"dtype": torch.float8_e4m3fn}
            )
            patched_model = utils.patch_model_with_loras(diffusion_model, config)
            patched_model = utils.patch_model_easycache(patched_model, config)
            patched_model = utils.patch_model_sampling(patched_model, shift=config.get("shift"), multiplier=1.0)

            all_samples = []
            for idx, (positive_cond, negative_cond) in enumerate(all_conditionings):
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
                    cfg=cfg,
                    sampler_name=sampler,
                    scheduler=scheduler,
                    positive=positive_cond,
                    negative=negative_cond,
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
                images = images.flatten(0, 1)

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