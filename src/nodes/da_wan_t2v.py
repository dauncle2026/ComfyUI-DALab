import os
import torch
import math
from fractions import Fraction

import folder_paths
import comfy.sd
import comfy.sample
import latent_preview
from comfy_api.latest import io, InputImpl, Types

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.model_manager import ModelManager
from ..utils.logger import logger

_CONFIG_FILE_PATH = utils.get_config_file_path("wan_t2v")

class DAWanT2VConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        text_encoder_options = folder_paths.get_filename_list("text_encoders")
        vae_options = folder_paths.get_filename_list("vae")
        diffusion_options = folder_paths.get_filename_list("diffusion_models")
        lora_options = folder_paths.get_filename_list("loras")

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)
        low_loras = utils.dynamic_combo_loras(config, lora_options, key_name="low_loras")
        high_loras = utils.dynamic_combo_loras(config, lora_options, key_name="high_loras")

        return io.Schema(
            node_id="DAWanT2VConfig",
            display_name="DA Wan2.2 T2V Config",
            category="DALab/Video/Wan2.2 T2V",
            description="Configure the Wan2.2 T2V model params. Run first to save config.",
            is_output_node=True,
            inputs=[
                io.Combo.Input(
                    "text_encoder_model",
                    default=config.get("text_encoder_model", "wan/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
                    options=text_encoder_options,
                    display_name="text_encoder_model",
                    tooltip="The Wan T5 Text Encoder. Default: models/text_encoders",
                ),
                io.Combo.Input(
                    "vae_model",
                    default=config.get("vae_model", "wan/wan_2.1_vae.safetensors"),
                    options=vae_options,
                    display_name="vae_model",
                    tooltip="The Wan VAE model. Default: models/vae",
                ),
                io.Combo.Input(
                    "high_model",
                    default=config.get("high_model", "wan2.2/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"),
                    options=diffusion_options,
                    display_name="high_model",
                    tooltip="The Wan2.2 High Noise Diffusion Model (14B).",
                ),
                io.Combo.Input(
                    "low_model",
                    default=config.get("low_model", "wan2.2/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"),
                    options=diffusion_options,
                    display_name="low_model",
                    tooltip="The Wan2.2 Low Noise Diffusion Model (14B).",
                ),
                io.Int.Input(
                    "steps",
                    default=config.get("steps", 4),
                    min=1,
                    max=100,
                    tooltip="Sampling steps. Default 4 for Wan2.2 Lightning.",
                    display_name="steps",
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
                    default=config.get("shift", 5.0),
                    min=0.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Model sampling shift parameter. Default 5.0 for Wan.",
                    display_name="shift",
                ),
                io.Combo.Input(
                    "scheduler",
                    default=config.get("scheduler", "simple"),
                    options=comfy.samplers.KSampler.SCHEDULERS,
                    tooltip="Noise scheduler. 'simple' is recommended.",
                    display_name="scheduler",
                ),
                io.Combo.Input(
                    "sampler",
                    default=config.get("sampler", "euler"),
                    options=comfy.samplers.KSampler.SAMPLERS,
                    tooltip="Sampler algorithm. 'euler' is recommended.",
                    display_name="sampler",
                ),
                io.String.Input(
                    "negative_prompt",
                    default=config.get("negative_prompt", "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，裸露，NSFW"),
                    tooltip="Negative prompt.",
                    display_name="negative_prompt",
                    multiline=True,
                ),
                io.DynamicCombo.Input(
                    "high_loras",
                    options=high_loras,
                    display_name="high_loras",
                    tooltip="Apply one or more LoRAs to the generation.",
                ),
                io.DynamicCombo.Input(
                    "low_loras",
                    options=low_loras,
                    display_name="low_loras",
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
        high_model,
        low_model,
        steps,
        cfg,
        shift,
        scheduler,
        sampler,
        negative_prompt,
        high_loras,
        low_loras,
    ) -> io.NodeOutput:
        config_data = {
            "text_encoder_model": text_encoder_model,
            "vae_model": vae_model,
            "high_model": high_model,
            "low_model": low_model,
            "steps": steps,
            "cfg": cfg,
            "shift": shift,
            "scheduler": scheduler,
            "sampler": sampler,
            "negative_prompt": negative_prompt,
            "high_loras": high_loras,
            "low_loras": low_loras
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()

class DAWanT2V(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DAWanT2V",
            display_name="DA Wan2.2 T2V",
            category="DALab/Video/Wan2.2 T2V",
            description="Generate videos using Wan2.2 T2V model with config.",
            is_input_list=True,
            inputs=[
                io.Int.Input(
                    "width",
                    default=640,
                    min=16, 
                    max=2560, 
                    step=8,
                    display_name="width",
                    tooltip="Output video width in pixels.",
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    "height",
                    default=360,
                    min=16, 
                    max=2560, 
                    step=8,
                    display_name="height",
                    tooltip="Output video height in pixels.",
                    display_mode=io.NumberDisplay.number,
                ),
                io.Int.Input(
                    "frame_count",
                    default=81,
                    min=1, 
                    max=200, 
                    step=1,
                    display_name="frame_count",
                    tooltip="Number of frames to generate. Default 81.",
                    display_mode=io.NumberDisplay.number,
                ),
                io.Float.Input(
                    "fps",
                    default=16.0,
                    min=1, 
                    max=200, 
                    step=1,
                    display_name="fps",
                    tooltip="Frames per second for video output. Default 16.0.",
                    display_mode=io.NumberDisplay.number,
                ),
                io.String.Input(
                    "prompts",
                    multiline=True,
                    default="A beautiful girl",
                    display_name="prompts",
                    tooltip="Text description for video generation.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0, max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    display_name="seed",
                    tooltip="Random seed for reproducible generation.",
                ),
            ],
            outputs=[
                io.Video.Output(
                    "videos",
                    is_output_list=True,
                    display_name="videos",
                    tooltip="Generated videos.",
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
        frame_count: list[int],
        fps: list[float],
    ) -> io.NodeOutput:
        scale_factor = 8
        target_height = height[0]
        target_width = width[0]
        latent_height = (target_height + scale_factor - 1) // scale_factor
        latent_width = (target_width + scale_factor - 1) // scale_factor
        seed = seed[0]
        fps = int(fps[0])

        batch_inputs = utils.inputs_to_batch(
            defaults={"prompt": ""},
            prompt=prompts,
            frame_count=frame_count
        )

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)

        text_encoder_path = folder_paths.get_full_path_or_raise(
            "text_encoders", config.get("text_encoder_model")
        )
        vae_path = folder_paths.get_full_path_or_raise(
            "vae", config.get("vae_model")
        )
        high_model_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", config.get("high_model")
        )
        low_model_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", config.get("low_model")
        )

        steps = config.get("steps")
        cfg = config.get("cfg")
        scheduler = config.get("scheduler")
        sampler = config.get("sampler")
        negative_prompt = config.get("negative_prompt")

        manager = ModelManager()

        try:
            text_encoder = manager.get_text_encoder(
                paths=[text_encoder_path],
                clip_type=comfy.sd.CLIPType.WAN
            )

            all_conditionings = []
            for idx, batch_input in enumerate(batch_inputs):
                prompt = batch_input["prompt"]["value"]
                frame_cnt = batch_input["frame_count"]["value"]

                if prompt.strip() == "":
                    logger.info(f"WanT2V Prompt is empty, skipping: {idx+1}")
                    continue

                if frame_cnt is None:
                    logger.warning(f"Frame is None in index {idx}")
                    continue

                tokens = text_encoder.tokenize(prompt)
                positive = text_encoder.encode_from_tokens_scheduled(tokens)
                if cfg > 1.0:
                    negative_tokens = text_encoder.tokenize(negative_prompt)
                    negative = text_encoder.encode_from_tokens_scheduled(negative_tokens)
                else:
                    negative = []

                all_conditionings.append((positive, negative, frame_cnt))

            high_model = manager.get_diffusion_model(high_model_path)
            patched_high = utils.patch_model_with_loras(high_model, config, key_name="high_loras")
            patched_high = utils.patch_model_sampling(patched_high, shift=config.get("shift"), multiplier=1000.0)

            all_high_samples = []
            for idx, (positive, negative, frame_cnt) in enumerate(all_conditionings):
                logger.info(f"High diffusion processing {idx+1}/{len(all_conditionings)}")

                latent_image = torch.zeros(
                    [1, 16, ((frame_cnt - 1) // 4) + 1, latent_height, latent_width],
                    device=patched_high.load_device
                )

                high_noise = comfy.sample.prepare_noise(latent_image, seed)
                high_callback = latent_preview.prepare_callback(patched_high, steps)

                high_samples = comfy.sample.sample(
                    patched_high,
                    high_noise,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent_image=latent_image,
                    callback=high_callback,
                    seed=seed,
                    start_step=0,
                    last_step=math.ceil(steps / 2),
                    force_full_denoise=False,
                    disable_noise=False,
                )
                all_high_samples.append((high_samples, positive, negative, frame_cnt))

            low_model = manager.get_diffusion_model(low_model_path)
            patched_low = utils.patch_model_with_loras(low_model, config, key_name="low_loras")
            patched_low = utils.patch_model_sampling(patched_low, shift=config.get("shift"), multiplier=1000.0)

            all_low_samples = []
            for idx, (high_samples, positive, negative, frame_cnt) in enumerate(all_high_samples):
                logger.info(f"Low diffusion processing {idx+1}/{len(all_high_samples)}")

                low_noise = torch.zeros(
                    high_samples.size(),
                    dtype=high_samples.dtype,
                    layout=high_samples.layout,
                    device="cpu"
                )
                low_callback = latent_preview.prepare_callback(patched_low, steps)

                low_samples = comfy.sample.sample(
                    patched_low,
                    low_noise,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent_image=high_samples,
                    callback=low_callback,
                    seed=seed,
                    start_step=math.ceil(steps / 2),
                    last_step=steps,
                    force_full_denoise=True,
                    disable_noise=True,
                )
                all_low_samples.append((low_samples, frame_cnt))

            vae = manager.get_vae(vae_path)

            output_videos = []
            for idx, (low_samples, frame_cnt) in enumerate(all_low_samples):
                logger.info(f"VAE decoding {idx+1}/{len(all_low_samples)}")

                images = vae.decode(low_samples)
                images = images.flatten(0, 1)

                video = InputImpl.VideoFromComponents(
                    Types.VideoComponents(images=images, audio=None, frame_rate=Fraction(fps))
                )
                output_videos.append(video)

        finally:
            if manager.release_after_run:
                manager.release_all()
            elif manager.offload_after_run:
                manager.offload_all()

        return io.NodeOutput(output_videos)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            config_mtime = os.path.getmtime(_CONFIG_FILE_PATH)
            global_config_mtime = os.path.getmtime(utils.get_config_file_path("global"))
        except:
            config_mtime = 0
            global_config_mtime = 0

        return hash((str(kwargs), str(config_mtime), str(global_config_mtime)))