import os
import torch
import av
import numpy as np
import copy
import json
from io import BytesIO
from fractions import Fraction
from types import SimpleNamespace

import folder_paths
import node_helpers
import latent_preview
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.nested_tensor
from comfy_api.latest import io, InputImpl, Types, Input
from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.model_manager import ModelManager
from ..utils.logger import logger
from ..utils.paths import get_config_file_path

_CONFIG_FILE_PATH = get_config_file_path("ltx2")

_CACHE = SimpleNamespace(
    positive={},
    negative={},
    video_latents={},
    audio_latents={},
)

class DALTX2Config(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        text_encoder_options = folder_paths.get_filename_list("text_encoders")
        diffusion_options = folder_paths.get_filename_list("diffusion_models")
        vae_options = folder_paths.get_filename_list("vae")
        upscale_model_options = folder_paths.get_filename_list("latent_upscale_models")
        lora_options = folder_paths.get_filename_list("loras")

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)

        step1_loras = utils.dynamic_combo_loras(config, lora_options, "step1_loras")
        step2_loras = utils.dynamic_combo_loras(config, lora_options, "step2_loras")

        return io.Schema(
            node_id="DALTX2Config",
            display_name="DA LTX2 Config",
            category="DALab/Video/LTX2",
            description="Configure the LTX2 model params. Run first to save config.",
            is_output_node=True,
            inputs=[
                io.Combo.Input(
                    "text_encoder_model",
                    default=config.get("text_encoder_model", "ltx/gemma_3_12B_it.safetensors"),
                    options=text_encoder_options,
                    display_name="text_encoder_model",
                    tooltip="The LTX Text Encoder (Gemma). Default: models/text_encoders",
                ),
                io.Combo.Input(
                    "text_proj",
                    default=config.get("text_proj", "ltx2/ltx2_text_proj.safetensors"),
                    options=text_encoder_options,
                    display_name="text_proj",
                    tooltip="The LTX2 Text Projection. Default: models/text_encoders",
                ),
                io.Combo.Input(
                    "diffusion_model",
                    default=config.get("diffusion_model", "ltx2_split/ltx2_diffusion.safetensors"),
                    options=diffusion_options,
                    display_name="diffusion_model",
                    tooltip="The LTX2 Diffusion Model (UNet). Default: models/diffusion_models",
                ),
                io.Combo.Input(
                    "video_vae_model",
                    default=config.get("video_vae_model", "ltx2_split/ltx2_video_vae.safetensors"),
                    options=vae_options,
                    display_name="video_vae_model",
                    tooltip="The LTX2 Video VAE. Default: models/vae",
                ),
                io.Combo.Input(
                    "audio_vae_model",
                    default=config.get("audio_vae_model", "ltx2_split/ltx2_audio_vae.safetensors"),
                    options=vae_options,
                    display_name="audio_vae_model",
                    tooltip="The LTX2 Audio VAE. Default: models/vae",
                ),
                io.Combo.Input(
                    "upscale_model",
                    default=config.get("upscale_model", "ltx/ltx-2-spatial-upscaler-x2-1.0.safetensors"),
                    options=upscale_model_options,
                    display_name="upscale_model",
                    tooltip="The LTX Upscale Model.",
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
                io.Combo.Input(
                    "sampler1",
                    default=config.get("sampler1", "euler"),
                    options=comfy.samplers.KSampler.SAMPLERS,
                    tooltip="Sampler algorithm. 'euler' is recommended.",
                    display_name="sampler1",
                ),
                io.Combo.Input(
                    "sampler2",
                    default=config.get("sampler2", "gradient_estimation"),
                    options=comfy.samplers.KSampler.SAMPLERS,
                    tooltip="Sampler algorithm. 'euler' is recommended.",
                    display_name="sampler2",
                ),
                io.Int.Input(
                    "image_crf1",
                    default=config.get("image_crf1", 33),
                    min=0,
                    max=100,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Image CRF. Default 33.",
                    display_name="image_crf1",
                ),
                io.Int.Input(
                    "image_crf2",
                    default=config.get("image_crf2", 33),
                    min=0,
                    max=100,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Image CRF. Default 33.",
                    display_name="image_crf2",
                ),
                io.Float.Input(
                    "image_strength1",
                    default=config.get("image_strength1", 1),
                    min=0,
                    max=10,
                    step=0.1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Image Strength. Default 1.",
                    display_name="image_strength1",
                ),
                io.Float.Input(
                    "image_strength2",
                    default=config.get("image_strength2", 1),
                    min=0,
                    max=10,
                    step=0.1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Image Strength. Default 1.",
                    display_name="image_strength2",
                ),
                io.String.Input(
                    "sigmas1",
                    default=config.get("sigmas1", "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"),
                    tooltip="Sigmas. with distilled: 1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0 with dev: 1.0000, 0.9915, 0.9821, 0.9718, 0.9605, 0.9478, 0.9337, 0.9178, 0.8998,0.8792, 0.8554, 0.8276, 0.7948, 0.7553, 0.7071, 0.6467, 0.5690, 0.4652,0.3194, 0.1000, 0.0000",
                    display_name="sigmas1",
                    multiline=True,
                ),
                io.String.Input(
                    "sigmas2",
                    default=config.get("sigmas2", "0.909375, 0.725, 0.421875, 0.0"),
                    tooltip="Sigmas. Default 0.909375, 0.725, 0.421875, 0.0.",
                    display_name="sigmas2",
                    multiline=True,
                ),
                io.String.Input(
                    "negative_prompt",
                    default=config.get("negative_prompt", ""),
                    tooltip="Negative prompt for the video.",
                    display_name="negative_prompt",
                    multiline=True,
                ),
                io.DynamicCombo.Input(
                    "step1_loras",
                    options=step1_loras,
                    display_name="step1_loras",
                    tooltip="Additional Style LoRAs.",
                ),
                io.DynamicCombo.Input(
                    "step2_loras",
                    options=step2_loras,
                    display_name="step2_loras",
                    tooltip="Additional Style LoRAs.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        text_encoder_model,
        text_proj,
        diffusion_model,
        video_vae_model,
        audio_vae_model,
        upscale_model,
        cfg,
        sampler1,
        sampler2,
        image_crf1,
        image_crf2,
        image_strength1,
        image_strength2,
        sigmas1,
        sigmas2,
        negative_prompt,
        step1_loras,
        step2_loras
    ) -> io.NodeOutput:
        config_data = {
            "text_encoder_model": text_encoder_model,
            "text_proj": text_proj,
            "diffusion_model": diffusion_model,
            "video_vae_model": video_vae_model,
            "audio_vae_model": audio_vae_model,
            "upscale_model": upscale_model,
            "cfg": cfg,
            "sampler1": sampler1,
            "sampler2": sampler2,
            "image_crf1": image_crf1,
            "image_crf2": image_crf2,
            "image_strength1": image_strength1,
            "image_strength2": image_strength2,
            "sigmas1": sigmas1,
            "sigmas2": sigmas2,
            "negative_prompt": negative_prompt,
            "step1_loras": step1_loras,
            "step2_loras": step2_loras
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()

class DALTX2(io.ComfyNode):
    @classmethod
    def reset_cache(cls):
        _CACHE.positive.clear()
        _CACHE.negative.clear()
        _CACHE.video_latents.clear()
        _CACHE.audio_latents.clear()

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DALTX2",
            display_name="DA LTX2",
            category="DALab/Video/LTX2",
            description="Generate videos using the LTX2 model.",
            is_input_list=True,
            inputs=[
                io.Int.Input(
                    "width",
                    default=1280,
                    min=16, 
                    max=2560, 
                    step=8,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Output video width in pixels.",
                    display_name="width",
                ),
                io.Int.Input(
                    "height",
                    default=720,
                    min=16, 
                    max=2560, 
                    step=8,
                    display_name="height",
                    display_mode=io.NumberDisplay.number,
                    tooltip="Output video height in pixels.",
                ),
                io.Int.Input(
                    "frame_count",
                    default=151,
                    min=1, 
                    max=200, 
                    step=1,
                    display_name="frame_count",
                    display_mode=io.NumberDisplay.number,
                    tooltip="Number of frames to generate. Default 151.",
                ),
                io.Int.Input(
                    "fps",
                    default=25,
                    min=1, 
                    max=120, 
                    step=1,
                    display_name="fps",
                    display_mode=io.NumberDisplay.number,
                    tooltip="FPS of the video. Default 25.",
                ),
                io.String.Input(
                    "prompts",
                    multiline=True,
                    default="A beautiful girl",
                    tooltip="Prompts for the video.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0, 
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed for the video.",
                ),
                io.MultiType.Input(
                    "first_frames",
                    optional=True,
                    types=[io.Image, io.Video],
                    tooltip="First frames of the video. Can be image or video.",
                ),
                io.MultiType.Input(
                    "last_frames",
                    optional=True,
                    types=[io.Image, io.Video],
                    tooltip="Last frames of the video. Can be image or video.",
                ),
                io.Audio.Input(
                    "audios",
                    optional=True,
                    tooltip="Audio for the video.",
                ),
            ],
            outputs=[
                io.Video.Output(
                    "videos", 
                    is_output_list=True, 
                    tooltip="Generated videos.",
                    display_name="videos",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        width: list[int],
        height: list[int],
        frame_count: list[int],
        fps: list[int],
        prompts: list[str],
        seed: list[int],
        audios=None,
        first_frames=None,
        last_frames=None,
    ) -> io.NodeOutput:
        manager = ModelManager()
        cls.reset_cache()

        scale_factor = 32
        target_height = height[0]
        target_width = width[0]
        latent_height = ((target_height + (scale_factor * 4) - 1) // (scale_factor * 4)) * 4
        latent_width =  ((target_width  + (scale_factor * 4) - 1) // (scale_factor * 4)) * 4
        half_latent_height = latent_height // 2
        half_latent_width = latent_width // 2
        seed = seed[0]
        fps = fps[0]

        batch_inputs = utils.inputs_to_batch(
            defaults={"prompt": ""},
            first_frame=first_frames,
            last_frame=last_frames,
            prompt=prompts,
            frame_count=frame_count,
            audio=audios,
        )

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)

        text_encoder_path = folder_paths.get_full_path_or_raise(
            "text_encoders",
            config.get("text_encoder_model")
        )
        text_proj_path = folder_paths.get_full_path_or_raise(
            "text_encoders",
            config.get("text_proj")
        )
        diffusion_model_path = folder_paths.get_full_path_or_raise(
            "diffusion_models",
            config.get("diffusion_model")
        )
        video_vae_path = folder_paths.get_full_path_or_raise(
            "vae",
            config.get("video_vae_model")
        )
        audio_vae_path = folder_paths.get_full_path_or_raise(
            "vae",
            config.get("audio_vae_model")
        )
        upscale_model_path = folder_paths.get_full_path_or_raise(
            "latent_upscale_models",
            config.get("upscale_model")
        )

        cfg = config.get("cfg")
        sampler1 = config.get("sampler1")
        sampler2 = config.get("sampler2")
        image_crf1 = config.get("image_crf1")
        image_crf2 = config.get("image_crf2")
        image_strength1 = config.get("image_strength1")
        image_strength2 = config.get("image_strength2")
        sigmas1 = torch.FloatTensor([float(sigma) for sigma in config.get("sigmas1").split(",")])
        sigmas2 = torch.FloatTensor([float(sigma) for sigma in config.get("sigmas2").split(",")])
        negative_prompt = config.get("negative_prompt")

        sampler1 = comfy.samplers.sampler_object(sampler1)
        sampler2 = comfy.samplers.sampler_object(sampler2)

        try:
            text_encoder = manager.get_text_encoder(
                paths=[text_encoder_path, text_proj_path],
                clip_type=comfy.sd.CLIPType.LTXV,
            )

            all_conditionings = []
            for idx, batch_input in enumerate(batch_inputs):
                logger.info(f"Processing text encoding input {idx+1}/{len(batch_inputs)}")

                prompt = batch_input["prompt"]
                first_frame = batch_input["first_frame"]
                last_frame = batch_input["last_frame"]
                frame_count_val = batch_input["frame_count"]["value"]
                audio = batch_input["audio"]

                if frame_count_val is None:
                    logger.warning(f"Frame is None in index {idx+1}")
                    continue

                if prompt["cache_key"] in _CACHE.positive:
                    logger.info(f"Positive prompt cache hit: {prompt['cache_key']} in index {idx+1}")
                    positive = copy.copy(_CACHE.positive[prompt["cache_key"]])
                else:
                    tokens = text_encoder.tokenize(prompt["value"])
                    positive = text_encoder.encode_from_tokens_scheduled(tokens)
                    _CACHE.positive[prompt["cache_key"]] = positive

                if cfg > 1.0:
                    if prompt["cache_key"] in _CACHE.negative:
                        logger.info(f"Negative prompt cache hit: {prompt['cache_key']} in index {idx+1}")
                        negative = copy.copy(_CACHE.negative[prompt["cache_key"]])
                    else:
                        negative_tokens = text_encoder.tokenize(negative_prompt)
                        negative = text_encoder.encode_from_tokens_scheduled(negative_tokens)
                        _CACHE.negative[prompt["cache_key"]] = negative
                else:
                    negative = []

                positive = node_helpers.conditioning_set_values(
                    positive, {"frame_rate": fps}
                )
                if cfg > 1.0:
                    negative = node_helpers.conditioning_set_values(
                        negative, {"frame_rate": fps}
                    )

                all_conditionings.append({
                    "positive": positive,
                    "negative": negative,
                    "first_frame": first_frame,
                    "last_frame": last_frame,
                    "frame_count": frame_count_val,
                    "audio": audio,
                })

            audio_vae = manager.get_audio_vae(audio_vae_path)

            for idx, item in enumerate(all_conditionings):
                logger.info(f"Processing step1 Audio VAE encoding input {idx+1}/{len(all_conditionings)}")

                frame_count_val = item["frame_count"]
                audio = item["audio"]

                step1_audio_latent, step1_audio_latent_mask = cls.create_empty_audio_latent(
                    audio_vae, frame_count_val, fps
                )
                step1_audio_latent, step1_audio_latent_mask = cls.encode_audio(
                    audio_vae=audio_vae,
                    audio=audio,
                    audio_latent=step1_audio_latent,
                    audio_latent_mask=step1_audio_latent_mask,
                )

                item["step1_audio_latent"] = step1_audio_latent
                item["step1_audio_latent_mask"] = step1_audio_latent_mask

            video_vae = manager.get_vae(video_vae_path)

            for idx, item in enumerate(all_conditionings):
                logger.info(f"Processing step1 VAE encoding input {idx+1}/{len(all_conditionings)}")

                first_frame = item["first_frame"]
                last_frame = item["last_frame"]
                frame_count_val = item["frame_count"]

                step1_video_latent, step1_video_latent_mask = cls.create_empty_video_latent(
                    frame_count_val, half_latent_height, half_latent_width
                )
                step1_video_latent, step1_video_latent_mask = cls.encode_frame(
                    video_vae=video_vae,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    video_latent=step1_video_latent,
                    video_latent_mask=step1_video_latent_mask,
                    image_crf=image_crf1,
                    image_strength=image_strength1
                )

                item["step1_video_latent"] = step1_video_latent
                item["step1_video_latent_mask"] = step1_video_latent_mask

            diffusion_model = manager.get_diffusion_model(diffusion_model_path)
            step1_diffusion_model = utils.patch_model_with_loras(diffusion_model, config, "step1_loras")

            for idx, item in enumerate(all_conditionings):
                logger.info(f"Processing step1 diffusion sampling input {idx+1}/{len(all_conditionings)}")

                positive = item["positive"]
                negative = item["negative"]
                step1_video_latent = item["step1_video_latent"]
                step1_video_latent_mask = item["step1_video_latent_mask"]
                step1_audio_latent = item["step1_audio_latent"]
                step1_audio_latent_mask = item["step1_audio_latent_mask"]

                step1_concat_latent = comfy.nested_tensor.NestedTensor(
                    (step1_video_latent, step1_audio_latent.to(step1_video_latent.device))
                )
                step1_concat_latent_mask = comfy.nested_tensor.NestedTensor(
                    (step1_video_latent_mask, step1_audio_latent_mask.to(step1_video_latent_mask.device))
                )

                step1_noise = comfy.sample.prepare_noise(step1_concat_latent, seed)
                step1_callback = latent_preview.prepare_callback(step1_diffusion_model, sigmas1.shape[-1] - 1)

                guider = comfy.samplers.CFGGuider(step1_diffusion_model)
                guider.set_conds(positive, negative)
                guider.set_cfg(cfg)

                step1_samples = guider.sample(
                    step1_noise,
                    step1_concat_latent,
                    sampler1,
                    sigmas1,
                    denoise_mask=step1_concat_latent_mask,
                    callback=step1_callback,
                    seed=seed
                )

                step1_samples = step1_samples.unbind()

                item['step2_video_latent'] = step1_samples[0]
                item['step2_video_latent_mask'] = step1_video_latent_mask

                if audio["value"] is not None:
                    item['step2_audio_latent'] = step1_audio_latent
                    item['step2_audio_latent_mask'] = step1_audio_latent_mask
                else:
                    item['step2_audio_latent'] = step1_samples[1]
                    item['step2_audio_latent_mask'] = step1_audio_latent_mask
            
            if manager.model_switch_offload:
                manager.offload_all()

            upscale_model = get_latent_upscale(upscale_model_path)

            for idx, item in enumerate(all_conditionings):
                logger.info(f"Processing latent upscaling input {idx+1}/{len(all_conditionings)}")

                step2_video_latent = item['step2_video_latent']
                step2_video_latent = cls.latent_upscale(upscale_model, video_vae, step2_video_latent)
                item['step2_video_latent'] = step2_video_latent
            
            if manager.model_switch_offload:
                upscale_model.to(comfy.model_management.intermediate_device())

            for idx, item in enumerate(all_conditionings):
                logger.info(f"Processing step2 VAE encoding input {idx+1}/{len(all_conditionings)}")

                positive = item["positive"]
                negative = item["negative"]
                first_frame = item["first_frame"]
                last_frame = item["last_frame"]
                audio = item["audio"]
                step2_video_latent = item["step2_video_latent"]
                step2_video_latent_mask = item["step2_video_latent_mask"]

                step2_video_latent, step2_video_latent_mask = cls.encode_frame(
                    video_vae=video_vae,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    video_latent=step2_video_latent,
                    video_latent_mask=step2_video_latent_mask,
                    image_crf=image_crf2,
                    image_strength=image_strength2
                )

                item['step2_video_latent'] = step2_video_latent
                item['step2_video_latent_mask'] = step2_video_latent_mask

            if manager.model_switch_offload:
                manager.offload_all()
                
            step2_diffusion_model = utils.patch_model_with_loras(diffusion_model, config, "step2_loras")

            for idx, item in enumerate(all_conditionings):
                logger.info(f"Processing step2 diffusion sampling input {idx+1}/{len(all_conditionings)}")

                positive = item["positive"]
                negative = item["negative"]
                first_frame = item["first_frame"]
                last_frame = item["last_frame"]
                audio = item["audio"]
                step2_video_latent = item["step2_video_latent"]
                step2_video_latent_mask = item["step2_video_latent_mask"]
                step2_audio_latent = item["step2_audio_latent"]
                step2_audio_latent_mask = item["step2_audio_latent_mask"]
                
                step2_concat_latent = comfy.nested_tensor.NestedTensor(
                    (step2_video_latent, step2_audio_latent.to(step2_video_latent.device))
                )
                step2_concat_latent_mask = comfy.nested_tensor.NestedTensor(
                    (step2_video_latent_mask, step2_audio_latent_mask.to(step2_video_latent_mask.device))
                )

                step2_noise = comfy.sample.prepare_noise(step2_concat_latent, seed)
                step2_callback = latent_preview.prepare_callback(step2_diffusion_model, sigmas2.shape[-1] - 1)

                guider = comfy.samplers.CFGGuider(step2_diffusion_model)
                guider.set_conds(positive, negative)
                guider.set_cfg(cfg)

                step2_samples = guider.sample(
                    step2_noise,
                    step2_concat_latent,
                    sampler2,
                    sigmas2,
                    denoise_mask=step2_concat_latent_mask,
                    callback=step2_callback,
                    seed=seed
                )

                step2_samples = step2_samples.unbind()
                item['output_video_latent'] = step2_samples[0]
                item['output_audio_latent'] = step2_samples[1]
            
            if manager.model_switch_offload:
                manager.offload_all()

            output_videos = []
            for idx, item in enumerate(all_conditionings):
                logger.info(f"Processing output video decoding {idx+1}/{len(all_conditionings)}")

                output_video_latent = item['output_video_latent']
                output_audio_latent = item['output_audio_latent']
                decoded_video = video_vae.decode(output_video_latent)
                decoded_video = decoded_video.reshape(
                    -1, decoded_video.shape[-3], decoded_video.shape[-2], decoded_video.shape[-1]
                )

                decoded_video = utils.scale_by_width_height(
                    decoded_video, target_width, target_height, "bilinear", "center"
                )

                decoded_audio = audio_vae.decode(output_audio_latent).cpu()

                video = InputImpl.VideoFromComponents(
                    Types.VideoComponents(
                        images=decoded_video,
                        audio={
                            "waveform": decoded_audio,
                            "sample_rate": audio_vae.output_sample_rate,
                        },
                        frame_rate=Fraction(fps)
                    )
                )

                output_videos.append(video)
        finally:
            global _LATENT_UPSCALE_CACHE
            if manager.release_after_run:
                manager.release_all()
                if _LATENT_UPSCALE_CACHE is not None:
                    _LATENT_UPSCALE_CACHE = None
            elif manager.offload_after_run:
                manager.offload_all()
                if _LATENT_UPSCALE_CACHE is not None:
                    _LATENT_UPSCALE_CACHE.to(comfy.model_management.intermediate_device())

        return io.NodeOutput(output_videos)

    @classmethod
    def create_empty_video_latent(cls, frame_count, height, width):
        empty_video_latent = torch.zeros(
            [1, 128, ((frame_count - 1) // 8) + 1, height, width], 
            device=comfy.model_management.intermediate_device()
        )
        empty_video_latent_mask = torch.ones(
            [1, 1, ((frame_count - 1) // 8) + 1, 1, 1], 
            device=comfy.model_management.intermediate_device()
        )
        return empty_video_latent, empty_video_latent_mask

    @classmethod
    def create_empty_audio_latent(cls, audio_vae, frame_count, fps):
        z_channels = audio_vae.latent_channels
        audio_freq = audio_vae.latent_frequency_bins
        num_audio_latents = audio_vae.num_of_latents_from_frames(frame_count, fps)

        empty_audio_latent = torch.zeros(
            (1, z_channels, num_audio_latents, audio_freq),
            device=comfy.model_management.intermediate_device(),
        )
        empty_audio_latent_mask = torch.ones(
            (1, 1, num_audio_latents, 1), 
            device=comfy.model_management.intermediate_device()
        )

        return empty_audio_latent, empty_audio_latent_mask

    @classmethod
    def encode_frame(
        cls, 
        video_vae,
        first_frame,
        last_frame,
        video_latent,
        video_latent_mask,
        image_crf,
        image_strength
    ):
        _, _, _, latent_height, latent_width = video_latent.shape

        if first_frame["value"] is not None:
            key = f"{first_frame['cache_key']}_{image_crf}_{image_strength}_{latent_height}_{latent_width}"

            if key in _CACHE.video_latents:
                logger.info(f"LTX2 video first frame latent cache hit: {key}")
                first_frame_latent = _CACHE.video_latents[key]
            else:
                first = first_frame["value"]
                if isinstance(first, Input.Video):
                    first = first_frame["value"].get_components().images[-1:]

                first = image_to_video(first, image_crf)
                scaled_first_frame = utils.scale_by_width_height(
                    first, latent_width * 32, latent_height * 32, "bilinear", "center"
                )

                first_frame_latent = video_vae.encode(scaled_first_frame)
                _CACHE.video_latents[key] = first_frame_latent

            video_latent[:, :, :first_frame_latent.shape[2]] = first_frame_latent
            video_latent_mask[:, :, :first_frame_latent.shape[2]] = 1.0 - image_strength

        if last_frame["value"] is not None:
            key = f"{last_frame['cache_key']}_{image_crf}_{image_strength}_{latent_height}_{latent_width}"

            if key in _CACHE.video_latents:
                logger.info(f"LTX2 video last frame latent cache hit: {key}")
                last_frame_latent = _CACHE.video_latents[key]
            else:
                last = last_frame["value"]
                if isinstance(last, Input.Video):
                    last = last_frame["value"].get_components().images[:1]

                last = image_to_video(last, image_crf)
                scaled_last_frame = utils.scale_by_width_height(
                    last, latent_width * 32, latent_height * 32, "bilinear", "center"
                )

                last_frame_latent = video_vae.encode(scaled_last_frame)
                _CACHE.video_latents[key] = last_frame_latent

            video_latent[:, :, -last_frame_latent.shape[2]:] = last_frame_latent
            video_latent_mask[:, :, -last_frame_latent.shape[2]:] = 1.0 - image_strength

        return video_latent, video_latent_mask

    @classmethod
    def encode_audio(
        cls,
        audio_vae,
        audio,
        audio_latent,
        audio_latent_mask,
    ):
        if audio["cache_key"] in _CACHE.audio_latents:
            logger.info(f"LTX2 audio latent cache hit: {audio['cache_key']}")

            audio_latent = _CACHE.audio_latents[audio["cache_key"]]["latent"]
            audio_latent_mask = _CACHE.audio_latents[audio["cache_key"]]["mask"]
        else:
            if audio["value"] is not None:
                if audio["value"]["waveform"].shape[1] == 1:
                    audio["value"]["waveform"] = audio["value"]["waveform"].repeat(1, 2, 1)

                ref_audio_latent = audio_vae.encode(audio["value"])

                if audio_latent.shape[2] < ref_audio_latent.shape[2]:
                    audio_latent = ref_audio_latent[:, :, :audio_latent.shape[2]]
                    audio_latent_mask[:] = 0.0
                else:
                    audio_latent[:, :, :ref_audio_latent.shape[2]] = ref_audio_latent
                    audio_latent_mask[:, :, :ref_audio_latent.shape[2]] = 0.0
                
                _CACHE.audio_latents[audio["cache_key"]] = {
                    "latent": audio_latent,
                    "mask": audio_latent_mask
                }

        return audio_latent, audio_latent_mask

    @classmethod
    def latent_upscale(cls, upscale_model, video_vae, video_latent):
        model_dtype = next(upscale_model.parameters()).dtype
        upscale_model.to(comfy.model_management.get_torch_device())

        video_latent = video_vae.first_stage_model.per_channel_statistics.un_normalize(
            video_latent.to(model_dtype)
        )
        upsampled_latents = upscale_model(video_latent)

        upsampled_latents = video_vae.first_stage_model.per_channel_statistics.normalize(
            upsampled_latents
        )

        return upsampled_latents

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            config_mtime = os.path.getmtime(_CONFIG_FILE_PATH)
            global_config_mtime = os.path.getmtime(get_config_file_path("global"))
        except:
            config_mtime = 0
            global_config_mtime = 0
            
        return hash((str(kwargs),str(config_mtime),str(global_config_mtime)))

_LATENT_UPSCALE_CACHE = None

def get_latent_upscale(path):
    global _LATENT_UPSCALE_CACHE
    if path in _LATENT_UPSCALE_CACHE:
        logger.info(f"Using cached latent upscaler model: {path}")
        return _LATENT_UPSCALE_CACHE

    sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    config = json.loads(metadata["config"])
    upscale_model = LatentUpsampler.from_config(config).to(
        dtype=comfy.model_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float32]),
        device=comfy.model_management.get_torch_device()
    )
    upscale_model.load_state_dict(sd)

    _LATENT_UPSCALE_CACHE[path] = upscale_model
    
    return upscale_model

def image_to_video(image: torch.Tensor, crf=29):
    if crf == 0:
        return image

    tensors = []
    for i in range(image.shape[0]):
        image_i = image[i]
        image_array = (image_i[:(image_i.shape[0] // 2) * 2, :(image_i.shape[1] // 2) * 2] * 255.0).byte().cpu().numpy()
        with BytesIO() as output_file:
            encode_single_frame(output_file, image_array, crf)
            video_bytes = output_file.getvalue()
        with BytesIO(video_bytes) as video_file:
            image_array = decode_single_frame(video_file)
        tensor = torch.tensor(image_array, dtype=image.dtype, device=image.device) / 255.0
        tensors.append(tensor)

    return torch.stack(tensors)

def encode_single_frame(output_file, image_array: np.ndarray, crf):
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height = image_array.shape[0]
        stream.width = image_array.shape[1]
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()

def decode_single_frame(video_file):
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")