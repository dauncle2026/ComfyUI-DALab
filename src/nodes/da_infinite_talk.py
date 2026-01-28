import os
import torch
import torchaudio
import copy
from fractions import Fraction
from types import SimpleNamespace
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

import folder_paths
import node_helpers
import latent_preview
import comfy.sd
import comfy.utils
import comfy.sample
import comfy.patcher_extension
import comfy.ops
import comfy.model_management
from comfy_api.latest import io, Input, InputImpl, Types
from comfy_api.torch_helpers import set_torch_compile_wrapper
from comfy_extras.nodes_model_patch import MultiTalkModelPatch
from comfy.ldm.wan.model_multitalk import (
    InfiniteTalkOuterSampleWrapper,
    MultiTalkCrossAttnPatch,
    MultiTalkGetAttnMapPatch,
    project_audio_features,
)

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.model_manager import ModelManager
from ..utils.logger import logger

_CONFIG_FILE_PATH = utils.get_config_file_path("infinite_talk")

_CACHE = SimpleNamespace(
    positive={},
    negative={},
    audio_embed={},
    clip_vision_embeds={},
    image_latents={},
)

_WAV2VEC_CACHE = None
_INFINITE_MODEL_CACHE = {}

class DAInfiniteTalkConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        diffusion_model_options = folder_paths.get_filename_list("diffusion_models")
        infinite_model_options = folder_paths.get_filename_list("model_patches")
        vae_options = folder_paths.get_filename_list("vae")
        text_encoder_options = folder_paths.get_filename_list("text_encoders")
        clip_vision_options = folder_paths.get_filename_list("clip_vision")
        audio_encoder_path = os.path.join(folder_paths.get_folder_paths("audio_encoders")[0], "TencentGameMate/chinese-wav2vec2-base")
        lora_options = folder_paths.get_filename_list("loras")

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)
        loras = utils.dynamic_combo_loras(config, lora_options)

        return io.Schema(
            node_id="DAInfiniteTalkConfig",
            display_name="DA InfiniteTalk Config",
            category="DALab/Video/Infinite Talk",
            description="Configure Infinite Talk model paths and parameters.",
            is_output_node=True,
            inputs=[
                io.Combo.Input(
                    "base_model",
                    default=config.get("base_model_name", "wan2.1/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors"),
                    options=diffusion_model_options,
                    display_name="base_model",
                    tooltip="The base Wan2.1 I2V model.",
                ),
                io.Combo.Input(
                    "single_infinite_model",
                    default=config.get("single_infinite_model", "Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"),
                    options=infinite_model_options,
                    display_name="single_infinite_model",
                    tooltip="The InfiniteTalk checkpoint to merge.",
                ),
                io.Combo.Input(
                    "multi_infinite_model",
                    default=config.get("multi_infinite_model", "Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors"),
                    options=infinite_model_options,
                    display_name="multi_infinite_model",
                    tooltip="The InfiniteTalk checkpoint to merge.",
                ),
                io.Combo.Input(
                    "vae_model",
                    default=config.get("vae_model", "wan/wan_2.1_vae.safetensors"),
                    options=vae_options,
                    display_name="vae_model",
                ),
                io.Combo.Input(
                    "text_encoder_model",
                    default=config.get("text_encoder_model", "wan/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
                    options=text_encoder_options,
                    display_name="text_encoder_model",
                ),
                io.Combo.Input(
                    "clip_vision_model",
                    default=config.get("clip_vision_model", "infinite_talk/clip_vision_h.safetensors"),
                    options=clip_vision_options,
                    display_name="clip_vision_model",
                ),
                io.String.Input(
                    "audio_encoder_path",
                    default=config.get("audio_encoder_path", audio_encoder_path),
                    display_name="audio_encoder_path",
                    tooltip="The audio encoder path. Default: TencentGameMate/chinese-wav2vec2-base",
                ),
                io.Boolean.Input(
                    "torch_compile",
                    default=config.get("torch_compile", False),
                    display_name="torch_compile",
                ),
                io.Int.Input(
                    "steps",
                    default=config.get("steps", 4),
                    min=1, 
                    max=100,
                    display_name="steps",
                ),
                io.Float.Input(
                    "cfg",
                    default=config.get("cfg", 1.0),
                    min=1.0, 
                    max=20.0, 
                    step=0.1,
                    display_name="cfg",
                ),
                io.Float.Input(
                    "shift",
                    default=config.get("shift", 3.0),
                    min=0.0, 
                    max=20.0, 
                    step=0.1,
                    display_name="shift",
                ),
                io.Int.Input(
                    "chunk_frame_count",
                    default=config.get("chunk_frame_count", 81),
                    min=1, 
                    max=200,
                    display_name="chunk_frame_count",
                    tooltip="Total frames per generation window.",
                ),
                io.Int.Input(
                    "chunk_motion_frame_count",
                    default=config.get("chunk_motion_frame_count", 9),
                    min=0, 
                    max=50,
                    display_name="chunk_motion_frame_count",
                    tooltip="Overlap frames for continuity.",
                ),
                io.Float.Input(
                    "prefix_silence_seconds",
                    default=config.get("prefix_silence_seconds", 0.5),
                    min=0.0,
                    max=2.0, 
                    step=0.01, 
                    display_name="prefix_silence_seconds",
                    tooltip="Silence seconds at the beginning of the audio.",
                ),
                io.String.Input(
                    "sampler",
                    default=config.get("sampler", "euler"),
                    display_name="sampler",
                    tooltip="The sampler name.",
                ),
                io.String.Input(
                    "scheduler",
                    default=config.get("scheduler", "simple"),
                    display_name="scheduler",
                    tooltip="The scheduler.",
                ),
                io.String.Input(
                    "negative_prompt",
                    default=config.get("negative_prompt", "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"),
                    multiline=True,
                    display_name="negative_prompt",
                ),
                io.DynamicCombo.Input(
                    "loras",
                    options=loras,
                    display_name="loras",
                    tooltip="Additional Style LoRAs for generation.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        base_model,
        single_infinite_model,
        multi_infinite_model,
        vae_model,
        text_encoder_model,
        clip_vision_model,
        audio_encoder_path,
        torch_compile,
        steps,
        cfg,
        shift,
        prefix_silence_seconds,
        chunk_frame_count,
        chunk_motion_frame_count,
        sampler,
        scheduler,
        negative_prompt,
        loras
    ) -> io.NodeOutput:
        config_data = {
            "base_model": base_model,
            "single_infinite_model": single_infinite_model,
            "multi_infinite_model": multi_infinite_model,
            "vae_model": vae_model,
            "text_encoder_model": text_encoder_model,
            "clip_vision_model": clip_vision_model,
            "audio_encoder_path": audio_encoder_path,
            "torch_compile": torch_compile,
            "steps": steps,
            "cfg": cfg,
            "shift": shift,
            "prefix_silence_seconds": prefix_silence_seconds,
            "chunk_frame_count": chunk_frame_count,
            "chunk_motion_frame_count": chunk_motion_frame_count,
            "sampler": sampler,
            "scheduler": scheduler,
            "negative_prompt": negative_prompt,
            "loras": loras,
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()

class DAInfiniteTalk(io.ComfyNode):
    @classmethod
    def reset_cache(cls):
        _CACHE.positive.clear()
        _CACHE.negative.clear()
        _CACHE.audio_embed.clear()
        _CACHE.clip_vision_embeds.clear()
        _CACHE.image_latents.clear()

    @classmethod
    def define_schema(cls) -> io.Schema:
        audio_options = [
            io.DynamicCombo.Option("one_person", [
                io.Audio.Input(
                    "audios",
                    tooltip="Person1 driving audio.",
                    optional=True,
                    display_name="audios",
                ),
                io.Int.Input(
                    "fps",
                    default=16,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Frames per second of the output video. Default 16.",
                    display_name="fps",
                ),
            ]),
            io.DynamicCombo.Option("two_person", [
                io.Audio.Input(
                    "person1_audios",
                    tooltip="Person1 driving audio.",
                    optional=True,
                    display_name="person1_audios",
                ),
                io.Audio.Input(
                    "person2_audios",
                    tooltip="Person2 driving audio.",
                    optional=True,
                    display_name="person2_audios",
                ),
                io.Combo.Input(
                    "who_first",
                    default="person1",
                    options=["person1", "person2"],
                    tooltip="Who will speak first.",
                    display_name="who_first",
                ),
                io.Float.Input(
                    "split_position",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Split position. 0.0 means person1 will speak first, 1.0 means person2 will speak first. Default: 0.5",
                    display_mode=io.NumberDisplay.slider,
                    display_name="split_position",
                ),
                io.Int.Input(
                    "fps",
                    default=16,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Frames per second of the output video. Default 16.",
                    display_name="fps",
                ),
            ]),
        ]

        return io.Schema(
            node_id="DAInfiniteTalk",
            display_name="DA InfiniteTalk",
            category="DALab/Video/Infinite Talk",
            description="Generate talking head videos using InfiniteTalk.",
            is_input_list=True,
            inputs=[
                io.Int.Input(
                    "width", 
                    default=640, 
                    min=16, 
                    max=2048, 
                    step=8, 
                    display_mode=io.NumberDisplay.number,
                    tooltip="The width of the video. Default: 640",
                    display_name="width",
                ),
                io.Int.Input(
                    "height", 
                    default=360, 
                    min=16, 
                    max=2048, 
                    step=8, 
                    display_mode=io.NumberDisplay.number,
                    tooltip="The height of the video. Default: 640",
                    display_name="height",
                ),
                io.MultiType.Input(
                    "first_frames",
                    optional=True,
                    types=[io.Image, io.Video],
                    tooltip="First frames of the video. Can be image or video.If video use last frame",
                    display_name="first_frames",
                ),
                io.Video.Input( 
                    "from_videos",
                    tooltip="Generate from video.",
                    optional=True,
                    display_name="from_videos",
                ),
                io.Int.Input(
                    "max_frame_count",
                    default=-1,
                    min=-1,
                    max=2048,
                    step=1,
                    tooltip="The maximum frame count of the video. Default: -1 (no limit)",
                    display_name="max_frame_count",
                ),
                io.DynamicCombo.Input(
                    "audio_options",
                    options=audio_options,
                    tooltip="The audio options. Default: one person",
                    display_name="audio_options",
                ),
                io.String.Input(
                    "prompts", 
                    multiline=True, 
                    default="一位激情演讲的男士",
                    tooltip="The prompts of the generation. Default: 一位激情演讲的男士",
                    display_name="prompts",
                ),
                io.Int.Input(
                    "seed", 
                    default=0, 
                    min=0, 
                    max=0xFFFFFFFFFFFFFFFF, 
                    control_after_generate=True,
                    tooltip="The seed of the generation. Default: 0",
                    display_name="seed",
                ),
            ],
            outputs=[
                io.Video.Output(
                    "videos", 
                    is_output_list=True,
                    display_name="videos",
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
        max_frame_count: list[int],
        first_frames = None,
        audio_options = None,
        from_videos = None,
    ) -> io.NodeOutput:
        global _WAV2VEC_CACHE, _INFINITE_MODEL_CACHE
        manager = ModelManager()
        cls.reset_cache()

        scale_factor = 8
        target_height = height[0]
        target_width = width[0]
        latent_height = ((target_height + (scale_factor * 2) - 1) // (scale_factor * 2)) * 2
        latent_width =  ((target_width  + (scale_factor * 2) - 1) // (scale_factor * 2)) * 2

        seed = seed[0]
        max_frame_count = max_frame_count[0]

        batch_inputs = utils.inputs_to_batch(
            defaults={"prompt": ""},
            prompt=prompts,
            first_frame=first_frames,
            nested_inputs={
                "audio_option": audio_options,
            },
            from_video=from_videos
        )

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)

        base_model_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", config.get("base_model")
        )
        single_infinite_model_path = folder_paths.get_full_path_or_raise(
            "model_patches", config.get("single_infinite_model")
        )
        multi_infinite_model_path = folder_paths.get_full_path_or_raise(
            "model_patches", config.get("multi_infinite_model")
        )
        vae_path = folder_paths.get_full_path_or_raise(
            "vae", config.get("vae_model")
        )
        text_encoder_path = folder_paths.get_full_path_or_raise(
            "text_encoders", config.get("text_encoder_model")
        )
        clip_vision_path = folder_paths.get_full_path_or_raise(
            "clip_vision", config.get("clip_vision_model")
        )
        audio_encoder_path = config.get("audio_encoder_path")

        if not os.path.exists(audio_encoder_path):
            raise ValueError(f"Audio encoder path {audio_encoder_path} not found")

        cfg = config.get("cfg")
        steps = config.get("steps")
        shift = config.get("shift")
        sampler = config.get("sampler")
        scheduler = config.get("scheduler")
        torch_compile = config.get("torch_compile")
        negative_prompt = config.get("negative_prompt")
        chunk_frame_count = config.get("chunk_frame_count")
        prefix_silence_seconds = config.get("prefix_silence_seconds")
        chunk_motion_frame_count = config.get("chunk_motion_frame_count")

        latent_t = (chunk_frame_count - 1) // 4 + 1

        try:
            text_encoder = manager.get_text_encoder(
                paths=[text_encoder_path], clip_type=comfy.sd.CLIPType.WAN
            )

            all_conditionings = []
            for idx, batch_input in enumerate(batch_inputs):
                logger.info(f"Processing text encoding {idx+1}/{len(batch_inputs)}")

                prompt = batch_input["prompt"]
                first_frame = batch_input["first_frame"]
                audio_option = batch_input["audio_option"]
                from_video = batch_input["from_video"]
                fps = batch_input["audio_option"]["fps"]["value"]

                if fps is None:
                    logger.warning(f"FPS is None in index {idx+1}, skipping")
                    continue

                if first_frame["value"] is None and from_video["value"] is None:
                    logger.warning(f"First frame and from video are None in index {idx+1}, skipping")
                    continue

                if audio_option["audio_options"]["value"] == "one_person":
                    if audio_option["audios"]["value"] is None:
                        logger.warning(f"Audio is None in index {idx+1}, skipping")
                        continue
                else:
                    if audio_option["person1_audios"]["value"] is None and audio_option["person2_audios"]["value"] is None:
                        logger.warning(f"Audio is None in index {idx+1}, skipping")
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

                all_conditionings.append({
                    "positive": positive,
                    "negative": negative,
                    "first_frame": first_frame,
                    "audio_option": audio_option,
                    "from_video": from_video,
                    "fps": fps,
                })

            if manager.model_switch_offload:
                manager.offload_all()

            device = comfy.model_management.get_torch_device()
            wav2vec, wav2vec_feature_extractor = get_wav2vec_model(audio_encoder_path, device)

            for idx, item in enumerate(all_conditionings):
                logger.info(f"Processing audio encoding {idx+1}/{len(all_conditionings)}")

                audio_option = item["audio_option"]
                fps = item["fps"]

                person1_audio_embed, person2_audio_embed, combine_audio = cls.encode_audio_options(
                    wav2vec, wav2vec_feature_extractor, audio_option, fps, prefix_silence_seconds
                )

                encoded_audio_list = []
                if person1_audio_embed is not None:
                    encoded_audio_list.append(person1_audio_embed)
                if person2_audio_embed is not None:
                    encoded_audio_list.append(person2_audio_embed)

                token_ref_target_masks = None
                if person1_audio_embed is not None and person2_audio_embed is not None:
                    split_position = audio_option["split_position"]["value"]
                    ref_target_masks = cls.create_audio_mask(latent_height, latent_width, split_position)
                    token_ref_target_masks = torch.nn.functional.interpolate(
                        ref_target_masks.unsqueeze(0),
                        size=(latent_height // 2, latent_width // 2),
                        mode='nearest'
                    )[0]
                    token_ref_target_masks = (token_ref_target_masks > 0).view(token_ref_target_masks.shape[0], -1)
                    total_audio_frames = min(person1_audio_embed.shape[0], person2_audio_embed.shape[0])
                else:
                    total_audio_frames = person1_audio_embed.shape[0]

                if max_frame_count > 0 and total_audio_frames > max_frame_count:
                    total_audio_frames = max_frame_count
                    logger.info(f"Limiting frames to {max_frame_count} in index {idx+1}")

                item["person1_audio_embed"] = person1_audio_embed
                item["person2_audio_embed"] = person2_audio_embed
                item["combine_audio"] = combine_audio
                item["encoded_audio_list"] = encoded_audio_list
                item["token_ref_target_masks"] = token_ref_target_masks
                item["total_audio_frames"] = total_audio_frames

            if manager.model_switch_offload:
                if _WAV2VEC_CACHE is not None:
                    _WAV2VEC_CACHE["model"].to(comfy.model_management.intermediate_device())

            clip_vision = manager.get_clip_vision(clip_vision_path)
            for idx, item in enumerate(all_conditionings):
                logger.info(f"Processing clip encoding {idx+1}/{len(all_conditionings)}")

                first_frame = item["first_frame"]
                from_video = item["from_video"]
                total_audio_frames = item["total_audio_frames"]

                if from_video["value"] is not None:
                    current_audio_idx = 0
                    from_video_clip_vision_embeds = {}
                    while current_audio_idx < total_audio_frames:
                        current_frame = cls.get_current_frame(from_video, current_audio_idx)
                        clip_vision_embed = cls.clip_vision_encode(clip_vision, current_frame, latent_width, latent_height)
                        
                        from_video_clip_vision_embeds[current_audio_idx] = clip_vision_embed

                        current_audio_idx += chunk_frame_count - chunk_motion_frame_count
                    
                    item["from_video_clip_vision_embeds"] = from_video_clip_vision_embeds
                else:
                    clip_vision_embed = cls.clip_vision_encode(clip_vision, first_frame, latent_width, latent_height)
                    item["first_frame_clip_vision_embed"] = clip_vision_embed

            vae = manager.get_vae(vae_path)
            for idx, item in enumerate(all_conditionings):
                logger.info(f"Processing vae encoding {idx+1}/{len(all_conditionings)}")

                first_frame = item["first_frame"]
                from_video = item["from_video"]

                if from_video["value"] is not None:
                    current_audio_idx = 0
                    from_video_concat_latent_images = {}
                    from_video_concat_latent_masks = {}
                    while current_audio_idx < total_audio_frames:
                        current_frame = cls.get_current_frame(from_video, current_audio_idx)
                        concat_latent_image, concat_latent_mask = cls.encode_latent_frame(
                            vae, current_frame, latent_t, latent_width, latent_height
                        )
                        
                        from_video_concat_latent_images[current_audio_idx] = concat_latent_image
                        from_video_concat_latent_masks[current_audio_idx] = concat_latent_mask

                        current_audio_idx += chunk_frame_count - chunk_motion_frame_count
                    
                    item["from_video_concat_latent_images"] = from_video_concat_latent_images
                    item["from_video_concat_latent_masks"] = from_video_concat_latent_masks
                else:
                    concat_latent_image, concat_latent_mask = cls.encode_latent_frame(
                        vae, first_frame, latent_t, latent_width, latent_height
                    )
                    item["first_frame_concat_latent_image"] = concat_latent_image
                    item["first_frame_concat_latent_mask"] = concat_latent_mask

            base_model = manager.get_diffusion_model(base_model_path)

            if audio_options["audio_options"][0] == "one_person":
                infinite_model = get_infinite_model(single_infinite_model_path)
            else:
                infinite_model = get_infinite_model(multi_infinite_model_path)

            base_model = utils.patch_model_with_loras(base_model, config)
            base_model = utils.patch_model_sampling(base_model, shift, 1000.0)

            if torch_compile:
                set_torch_compile_wrapper(
                    model=base_model, backend="inductor", options={"guard_filter_fn": skip_torch_compile_dict}
                )

            output_videos = []
            for idx, item in enumerate(all_conditionings):
                logger.info(f"Processing sampling {idx+1}/{len(all_conditionings)}")

                positive = item["positive"]
                negative = item["negative"]
                first_frame = item["first_frame"]
                from_video = item["from_video"]
                fps = item["fps"]
                encoded_audio_list = item["encoded_audio_list"]
                token_ref_target_masks = item["token_ref_target_masks"]
                total_audio_frames = item["total_audio_frames"]
                combine_audio = item["combine_audio"]

                positive_input = copy.copy(positive)
                negative_input = copy.copy(negative)

                current_audio_idx = 0
                generated_frames_list = []
                previous_latent_motion = None

                needs_dynamic_frame_encode = from_video["value"] is not None
                if needs_dynamic_frame_encode:
                    clip_vision = manager.get_clip_vision(clip_vision_path)
                    vae = manager.get_vae(vae_path)

                while current_audio_idx < total_audio_frames:
                    logger.info(f"Processing sampling {idx+1} audio index: {current_audio_idx}/{total_audio_frames}")

                    if from_video["value"] is not None:
                        current_frame_clip_embed = item["from_video_clip_vision_embeds"][current_audio_idx]
                        concat_latent_image = item["from_video_concat_latent_images"][current_audio_idx]
                        concat_latent_mask = item["from_video_concat_latent_masks"][current_audio_idx]
                    else:
                        current_frame_clip_embed = item["first_frame_clip_vision_embed"]
                        concat_latent_image = item["first_frame_concat_latent_image"]
                        concat_latent_mask = item["first_frame_concat_latent_mask"]

                    positive_input = node_helpers.conditioning_set_values(
                        positive_input, {"clip_vision_output": current_frame_clip_embed}
                    )
                    positive_input = node_helpers.conditioning_set_values(
                        positive_input, {"concat_latent_image": concat_latent_image, "concat_mask": concat_latent_mask}
                    )

                    work_model = base_model.clone()

                    if current_audio_idx == 0:
                        motion_lat_t = 1
                        motion_frames_latent = concat_latent_image[:, :, :1]
                    else:
                        motion_lat_t = ((chunk_motion_frame_count - 1) // 4) + 1
                        inject_len = min(previous_latent_motion.shape[2], motion_lat_t)
                        motion_frames_latent = previous_latent_motion[:, :, :inject_len, :, :]

                    audio_start = current_audio_idx
                    audio_end = current_audio_idx + chunk_frame_count
                    audio_embed = project_audio_features(
                        infinite_model.model.audio_proj,
                        encoded_audio_list,
                        audio_start,
                        audio_end
                    ).to(work_model.model_dtype())

                    work_model.model_options["transformer_options"]["audio_embeds"] = audio_embed

                    work_model.add_wrapper_with_key(
                        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                        "infinite_talk_outer_sample",
                        InfiniteTalkOuterSampleWrapper(
                            motion_frames_latent,
                            infinite_model,
                            is_extend=True,
                        )
                    )

                    work_model.set_model_patch(
                        MultiTalkCrossAttnPatch(infinite_model, audio_scale=1.0),
                        "attn2_patch"
                    )

                    if token_ref_target_masks is not None:
                        work_model.set_model_patch(
                            MultiTalkGetAttnMapPatch(token_ref_target_masks),
                            "attn1_patch"
                        )

                    empty_latent = torch.zeros((1, 16, latent_t, latent_height, latent_width))
                    noise = comfy.sample.prepare_noise(empty_latent, seed)
                    callback = latent_preview.prepare_callback(work_model, steps)

                    samples = comfy.sample.sample(
                        work_model,
                        noise,
                        steps=steps,
                        cfg=cfg,
                        sampler_name=sampler,
                        scheduler=scheduler,
                        positive=positive_input,
                        negative=negative_input,
                        latent_image=empty_latent,
                        callback=callback,
                        seed=seed,
                    )

                    decoded_images = vae.decode(samples)
                    decoded_images = decoded_images.flatten(0, 1)

                    output_decoded_images = utils.scale_by_width_height(decoded_images, target_width, target_height, "bilinear", "center")

                    if current_audio_idx == 0:
                        generated_frames_list.append(output_decoded_images)
                    else:
                        generated_frames_list.append(output_decoded_images[chunk_motion_frame_count:])

                    previous_latent_motion = vae.encode(decoded_images[-chunk_motion_frame_count:])
                    current_audio_idx += chunk_frame_count - chunk_motion_frame_count

                    generated_frames_len = sum(t.shape[0] for t in generated_frames_list)
                    if generated_frames_len >= total_audio_frames:
                        break

                final_video_frames = torch.cat(generated_frames_list, dim=0)
                final_video_frames = final_video_frames[:int(total_audio_frames)]

                video_output = InputImpl.VideoFromComponents(
                    Types.VideoComponents(
                        images=final_video_frames,
                        audio=combine_audio,
                        frame_rate=Fraction(fps)
                    )
                )
                output_videos.append(video_output)
        finally:
            if manager.release_after_run:
                manager.release_all()
                _WAV2VEC_CACHE = None
                _INFINITE_MODEL_CACHE.clear()
            elif manager.offload_after_run:
                manager.offload_all()
                if _WAV2VEC_CACHE is not None:
                    _WAV2VEC_CACHE["model"].to(comfy.model_management.intermediate_device())
                for model in _INFINITE_MODEL_CACHE.values():
                    model.model.to(comfy.model_management.unet_offload_device())

        return io.NodeOutput(output_videos)

    @classmethod
    def add_silence(cls, audio, add_audio, add_side = "front", prefix_silence_seconds = 0):
        audio_waveform = audio["waveform"]
        audio_sample_rate = audio["sample_rate"]

        if add_audio is not None:   
            add_audio_waveform = add_audio["waveform"]
            add_audio_sample_rate = add_audio["sample_rate"]

            silence_length = int((add_audio_waveform.shape[2] / add_audio_sample_rate) * audio_sample_rate)
            silence_tensor = torch.zeros(
                (audio_waveform.shape[0], audio_waveform.shape[1], silence_length),
                dtype=audio_waveform.dtype, 
                device=audio_waveform.device
            )

            if add_side == "front":
                audio_waveform = torch.cat((silence_tensor, audio_waveform), dim=2)
            elif add_side == "back":
                audio_waveform = torch.cat((audio_waveform,silence_tensor), dim=2)
        
        if prefix_silence_seconds > 0.01:
            prefix_silence_frames = int(prefix_silence_seconds * audio_sample_rate)
            silence_tensor = torch.zeros(
                (audio_waveform.shape[0], audio_waveform.shape[1], prefix_silence_frames),
                dtype=audio_waveform.dtype, 
                device=audio_waveform.device
            )
            audio_waveform = torch.cat((silence_tensor, audio_waveform), dim=2)

        return {
            'waveform': audio_waveform,
            'sample_rate': audio_sample_rate
        }

    @classmethod
    def resample_audio(cls, audio, model_sample_rate):
        audio_waveform = audio["waveform"]
        audio_sample_rate = audio["sample_rate"]

        if len(audio_waveform.shape) != 3:
            raise ValueError("[DALab] InfiniteTalk Audio waveform must be a 3D tensor")

        if audio_waveform.shape[1] != 1:
            audio_waveform = torch.mean(audio_waveform,1,keepdim=True)
        
        if audio_sample_rate != model_sample_rate:
            audio_waveform = torchaudio.functional.resample(audio_waveform, audio_sample_rate, model_sample_rate)

        audio_waveform = audio_waveform.repeat(1,2,1)

        audio = {
            'waveform': audio_waveform,
            'sample_rate': model_sample_rate
        }
        
        return audio

    @classmethod
    def encode_audio_options(cls, wav2vec2, wav2vec2_feature_extractor, audio_option, fps,prefix_silence_seconds = 0):
        model_sample_rate = 16000

        if audio_option["audio_options"]["value"] == "one_person":
            original_person1_audio = audio_option["audios"]["value"]
            silent_person1_audio = cls.add_silence(original_person1_audio, None, "none", prefix_silence_seconds)
            person1_audio = cls.resample_audio(silent_person1_audio, model_sample_rate)
            person1_embedding = cls.encode_audio(wav2vec2, wav2vec2_feature_extractor, person1_audio, fps)
            person2_embedding = None
            combine_audio = silent_person1_audio
        elif audio_option["audio_options"]["value"] == "two_person":
            who_first = audio_option["who_first"]["value"]
            original_person1_audio = audio_option["person1_audios"]["value"]
            original_person2_audio = audio_option["person2_audios"]["value"]

            if who_first == "person1":
                silent_person1_audio = cls.add_silence(original_person1_audio, original_person2_audio, "back", prefix_silence_seconds)
                silent_person2_audio = cls.add_silence(original_person2_audio, original_person1_audio, "front", prefix_silence_seconds)
            else:
                silent_person1_audio = cls.add_silence(original_person1_audio, original_person2_audio, "front", prefix_silence_seconds)
                silent_person2_audio = cls.add_silence(original_person2_audio, original_person1_audio, "back", prefix_silence_seconds)

            person1_audio = cls.resample_audio(silent_person1_audio, model_sample_rate)
            person2_audio = cls.resample_audio(silent_person2_audio, model_sample_rate)
            
            person1_embedding = cls.encode_audio(wav2vec2, wav2vec2_feature_extractor, person1_audio, fps)
            person2_embedding = cls.encode_audio(wav2vec2, wav2vec2_feature_extractor, person2_audio, fps)

            combine_audio_waveform = person1_audio["waveform"] + person2_audio["waveform"]
            combine_audio = {
                'waveform': combine_audio_waveform,
                'sample_rate': model_sample_rate
            }
        else:
            raise ValueError("[DALab] InfiniteTalk Invalid audio option")
        
        return person1_embedding, person2_embedding, combine_audio

    @classmethod
    def create_audio_mask(cls, latent_height, latent_width, split_position=0.5):
        face_scale = 0.05

        x_min = int(latent_height * face_scale)
        x_max = int(latent_height * (1 - face_scale))

        split_x = int(latent_width * split_position)

        human_mask1 = torch.zeros([latent_height, latent_width])
        human_mask2 = torch.zeros([latent_height, latent_width])

        left_margin = int(split_x * face_scale)

        lefty_min = left_margin
        lefty_max = split_x - left_margin

        right_panel_width = latent_width - split_x
        right_margin = int(right_panel_width * face_scale)

        righty_min = split_x + right_margin
        righty_max = latent_width - right_margin

        human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
        human_mask2[x_min:x_max, righty_min:righty_max] = 1

        ref_target_masks = torch.stack([human_mask1, human_mask2], dim=0)

        return ref_target_masks

    @classmethod
    def get_current_frame(cls, video, frame_index):
        if video["value"] is None:
            raise ValueError("Video is None")
        
        if not isinstance(video["value"], Input.Video):
            raise ValueError("Video is not a video input")

        video_value = video["value"].get_components()

        if frame_index >= video_value.images.shape[0]:
            current_frame_value = video_value.images[-1:]
            current_frame_cache_key = video["cache_key"] + f"_frame_{video_value.images.shape[0]}"
        else:
            current_frame_value = video_value.images[frame_index:frame_index+1]
            current_frame_cache_key = video["cache_key"] + f"_frame_{frame_index}"

        current_frame = {
            "value": current_frame_value,
            "cache_key": current_frame_cache_key
        }

        return current_frame

    @classmethod
    def clip_vision_encode(cls, clip_vision, frame, latent_width, latent_height):
        if frame["value"] is None:
            raise ValueError("Frame is None")
    
        if frame["cache_key"] in _CACHE.clip_vision_embeds:
            logger.info(f"[DALab] InfiniteTalk clip vision embed cache hit: {frame['cache_key']}")
            clip_vision_embed = _CACHE.clip_vision_embeds[frame["cache_key"]]
        else:
            frame_value = frame["value"]
            if isinstance(frame_value, Input.Video):
                frame_value = frame_value.get_components().images[-1:]
            
            frame_value_scaled = utils.scale_by_width_height(
                frame_value, latent_width * 8, latent_height * 8, "bilinear", "center"
            )

            clip_vision_embed = clip_vision.encode_image(frame_value_scaled)
            _CACHE.clip_vision_embeds[frame["cache_key"]] = clip_vision_embed

        return clip_vision_embed

    @classmethod
    def encode_latent_frame(cls, vae, frame, latent_t, latent_width, latent_height):
        if frame["value"] is None:
            raise ValueError("Frame is None")
        
        init_image = torch.ones(((latent_t - 1)*4+1, latent_height * 8, latent_width * 8, 3)) * 0.5
        init_mask = torch.ones(
            (1, 1, latent_t * 4, latent_height, latent_width)
        )

        if frame["cache_key"] in _CACHE.image_latents:
            logger.info(f"[DALab] InfiniteTalk image latent cache hit: {frame['cache_key']}")
            concat_latent_image = _CACHE.image_latents[frame["cache_key"]]['image']
            concat_latent_mask = _CACHE.image_latents[frame["cache_key"]]['mask']
        else:
            frame_value = frame["value"]
            if isinstance(frame_value, Input.Video):
                frame_value = frame_value.get_components().images[-1:]
            
            frame_value_scaled = utils.scale_by_width_height(
                frame_value, latent_width * 8, latent_height * 8, "bilinear", "center"
            )

            init_image[:frame_value_scaled.shape[0]] = frame_value_scaled
            init_mask[:, :, :frame_value_scaled.shape[0] + 3] = 0.0
        
            concat_latent_image = vae.encode(init_image[:, :, :, :3])
            concat_latent_mask = init_mask.view(1, init_mask.shape[2] // 4, 4, init_mask.shape[3], init_mask.shape[4]).transpose(1, 2)
            _CACHE.image_latents[frame["cache_key"]] = {
                'image': concat_latent_image, 
                'mask': concat_latent_mask
            }

        return concat_latent_image, concat_latent_mask

    @classmethod
    def encode_audio(cls, wav2vec2, wav2vec2_feature_extractor, audio, fps):
        audio_input = audio["waveform"]
        sample_rate = audio["sample_rate"]
        model_sample_rate = 16000

        audio_input = audio_input[0][0]

        if sample_rate != model_sample_rate:
            audio_input = torchaudio.functional.resample(audio_input, sample_rate, model_sample_rate)

        audio_features = wav2vec2_feature_extractor(
            audio_input,
            sampling_rate=model_sample_rate,
            return_tensors="pt"
        ).input_values

        audio_duration = len(audio_input) / model_sample_rate
        video_length = int(audio_duration * fps)

        audio_features = audio_features.to(dtype=wav2vec2.dtype, device=wav2vec2.device)

        with torch.no_grad():
            extract_features = wav2vec2.feature_extractor(audio_features)
            extract_features = extract_features.transpose(1, 2)

            extract_features = torch.nn.functional.interpolate(
                extract_features.transpose(1, 2),  # [1, 512, T_raw]
                size=video_length,
                mode='linear',
                align_corners=True
            ).transpose(1, 2)  # [1, video_length, 512]

            hidden_states, _ = wav2vec2.feature_projection(extract_features)

            encoder_outputs = wav2vec2.encoder(
                hidden_states,
                output_hidden_states=True,
                return_dict=True
            )

        stacked = torch.stack(encoder_outputs.hidden_states[1:], dim=1).squeeze(0)

        final_emb = stacked.movedim(0, 1)

        return final_emb
    
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            config_mtime = os.path.getmtime(_CONFIG_FILE_PATH)
            global_config_mtime = os.path.getmtime(utils.get_config_file_path("global"))
        except:
            config_mtime = 0
            global_config_mtime = 0

        return hash((str(kwargs), str(config_mtime), str(global_config_mtime)))

def skip_torch_compile_dict(guard_entries):
    return [("transformer_options" not in entry.name) for entry in guard_entries]

def get_wav2vec_model(audio_encoder_path, device):
    global _WAV2VEC_CACHE
    if _WAV2VEC_CACHE is not None and _WAV2VEC_CACHE["path"] == audio_encoder_path:
        _WAV2VEC_CACHE["model"].to(device)
        return _WAV2VEC_CACHE["model"], _WAV2VEC_CACHE["feature_extractor"]

    logger.info(f"Loading wav2vec model from {audio_encoder_path}")
    wav2vec = Wav2Vec2Model.from_pretrained(audio_encoder_path).float().to(device).eval()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        audio_encoder_path, local_files_only=True
    )

    _WAV2VEC_CACHE = {
        "path": audio_encoder_path,
        "model": wav2vec,
        "feature_extractor": wav2vec_feature_extractor
    }
    return wav2vec, wav2vec_feature_extractor

def get_infinite_model(model_patch_path):
    global _INFINITE_MODEL_CACHE
    if model_patch_path in _INFINITE_MODEL_CACHE:
        logger.info(f"Using cached infinite model: {model_patch_path}")
        return _INFINITE_MODEL_CACHE[model_patch_path]

    logger.info(f"Loading infinite model: {model_patch_path}")
    sd, metadata = comfy.utils.load_torch_file(model_patch_path, safe_load=True, return_metadata=True)
    sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        None,
        comfy.model_management.get_torch_device(),
        [torch.float16, torch.bfloat16, torch.float32]
    )

    model = MultiTalkModelPatch(
        audio_window=5, context_tokens=32, vae_scale=4,
        in_dim=sd["blocks.0.audio_cross_attn.proj.weight"].shape[0],
        intermediate_dim=sd["audio_proj.proj1.weight"].shape[0],
        out_dim=sd["audio_proj.norm.weight"].shape[0],
        device=comfy.model_management.unet_offload_device(),
        operations=comfy.ops.mixed_precision_ops({"mixed_ops": True}, manual_cast_dtype)
    )

    model.load_state_dict(sd)
    model = comfy.model_patcher.ModelPatcher(model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.unet_offload_device())

    _INFINITE_MODEL_CACHE[model_patch_path] = model
    return model