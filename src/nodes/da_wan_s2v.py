import os
import torch
import math
import copy
import numpy as np
from fractions import Fraction
from types import SimpleNamespace

import folder_paths
import node_helpers
import latent_preview
import comfy.sd
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.latent_formats
from comfy_api.latest import io, InputImpl, Types, Input

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.model_manager import ModelManager
from ..utils.logger import logger

_CONFIG_FILE_PATH = utils.get_config_file_path("wan_s2v")

_CACHE = SimpleNamespace(
    positive={},
    negative={},
    motion_latents={},
    image_latents={},
    audio_embed={},
    control_video_buckets={},
)

class DAWanS2VConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        text_encoder_options = folder_paths.get_filename_list("text_encoders")
        vae_options = folder_paths.get_filename_list("vae")
        diffusion_options = folder_paths.get_filename_list("diffusion_models")
        lora_options = folder_paths.get_filename_list("loras")
        audio_encoder_options = folder_paths.get_filename_list("audio_encoders")

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)
        loras = utils.dynamic_combo_loras(config, lora_options)

        return io.Schema(
            node_id="DAWanS2VConfig",
            display_name="DA Wan2.2 S2V Config",
            category="DALab/Video/Wan2.2 S2V",
            description="Configure the Wan2.2 Sound-to-Video model params. Run first to save config.",
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
                    "diffusion_model",
                    default=config.get("diffusion_model", "wan2.2/wan2.2_s2v_14B_fp8_scaled.safetensors"),
                    options=diffusion_options,
                    display_name="diffusion_model",
                    tooltip="The Wan2.2 Sound-to-Video Diffusion Model (14B).",
                ),
                io.Combo.Input(
                    "audio_encoder_model",
                    default=config.get("audio_encoder_model", "wan/wav2vec2_large_english_fp16.safetensors"),
                    options=audio_encoder_options,
                    display_name="audio_encoder_model",
                    tooltip="The Wav2Vec2 audio encoder for sound feature extraction.",
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
                io.Int.Input(
                    "chunk_frame_count",
                    default=config.get("chunk_frame_count", 77),
                    min=1,
                    max=100,
                    step=4,
                    tooltip="Number of frames per generation segment. Recommended: 77 (4.8s at 16fps).",
                    display_name="chunk_frame_count",
                ),
                io.Int.Input(
                    "chunk_motion_frame_count",
                    default=config.get("chunk_motion_frame_count", 73),
                    min=1,
                    max=100,
                    step=4,
                    tooltip="Number of motion frames per generation segment. Recommended: 73 (4.5s at 16fps).",
                    display_name="chunk_motion_frame_count",
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
                    default=config.get("sampler", "uni_pc"),
                    options=comfy.samplers.KSampler.SAMPLERS,
                    tooltip="Sampler algorithm. 'uni_pc' is recommended for S2V.",
                    display_name="sampler",
                ),
                io.String.Input(
                    "negative_prompt",
                    default=config.get("negative_prompt", "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"),
                    tooltip="Negative prompt for generation.",
                    display_name="negative_prompt",
                    multiline=True,
                ),
                io.DynamicCombo.Input(
                    "loras",
                    options=loras,
                    display_name="LoRAs",
                    tooltip="Additional Style LoRAs for generation.",
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
        audio_encoder_model,
        chunk_frame_count,
        chunk_motion_frame_count,
        steps,
        cfg,
        shift,
        scheduler,
        sampler,
        negative_prompt,
        loras
    ) -> io.NodeOutput:
        config_data = {
            "text_encoder_model": text_encoder_model,
            "vae_model": vae_model,
            "diffusion_model": diffusion_model,
            "audio_encoder_model": audio_encoder_model,
            "chunk_frame_count": chunk_frame_count,
            "chunk_motion_frame_count": chunk_motion_frame_count,
            "steps": steps,
            "cfg": cfg,
            "shift": shift,
            "scheduler": scheduler,
            "sampler": sampler,
            "negative_prompt": negative_prompt,
            "loras": loras
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()


class DAWanS2V(io.ComfyNode):
    @classmethod
    def reset_cache(cls):
        _CACHE.positive.clear()
        _CACHE.negative.clear()
        _CACHE.motion_latents.clear()
        _CACHE.image_latents.clear()
        _CACHE.audio_embed.clear()
        _CACHE.control_video_buckets.clear()

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DAWanS2V",
            display_name="DA Wan2.2 S2V",
            category="DALab/Video/Wan2.2 S2V",
            description="Generate lip-sync videos from audio using the Wan2.2 Sound-to-Video model.",
            is_input_list=True,
            inputs=[
                io.Int.Input(
                    "width",
                    default=640,
                    min=16,
                    max=1280,
                    step=8,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Width of the output video. Recommended: 640 for S2V.",
                    display_name="width",
                ),
                io.Int.Input(
                    "height",
                    default=360,
                    min=16,
                    max=1280,
                    step=8,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Height of the output video. Recommended: 360 for S2V.",
                    display_name="height",
                ),
                io.Int.Input(
                    "max_frame_count",
                    default=-1,
                    min=-1,
                    max=2048,
                    step=1,
                    tooltip="Max frame count of the output video. Default -1 (no limit).",
                    display_name="max_frame_count",
                ),
                io.Float.Input(
                    "fps",
                    default=16.0,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Frames per second of the output video. Default 16.0.",
                    display_name="fps",
                ),
                io.String.Input(
                    "prompts",
                    multiline=True,
                    default="一位激情演讲的男士",
                    tooltip="Text prompts describing the scene/person.",
                    display_name="prompts",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Random seed for generation. Same seed = same result.",
                    display_name="seed",
                ),
                io.MultiType.Input(
                    "first_frames",
                    optional=True,
                    types=[io.Image, io.Video],
                    tooltip="First frames of the video. Can be image or video.",
                    display_name="first_frames",
                ),
                io.Audio.Input(
                    "audios",
                    optional=True,
                    tooltip="Audio for the video.",
                    display_name="audios",
                ),
                io.Video.Input(
                    "control_videos",
                    optional=True,
                    tooltip="Control video for the video.",
                    display_name="control_videos",
                ),
            ],
            outputs=[
                io.Video.Output(
                    "videos",
                    is_output_list=True,
                    tooltip="Generated lip-sync videos.",
                    display_name="videos",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        width: list[int],
        height: list[int],
        fps: list[float],
        prompts: list[str],
        seed: list[int],
        max_frame_count: list[int],
        first_frames = None,
        audios = None,
        control_videos = None,
    ) -> io.NodeOutput:
        manager = ModelManager()
        cls.reset_cache()

        scale_factor = 8
        target_height = height[0]
        target_width = width[0]
        latent_height = ((target_height + (scale_factor * 2) - 1) // (scale_factor * 2)) * 2
        latent_width =  ((target_width  + (scale_factor * 2) - 1) // (scale_factor * 2)) * 2

        seed = seed[0]
        fps = int(fps[0])
        max_frame_count = max_frame_count[0]

        batch_inputs = utils.inputs_to_batch(
            defaults={"prompt": ""},
            prompt=prompts,
            first_frame=first_frames,
            audio=audios,
            control_video=control_videos,
        )

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)

        text_encoder_path = folder_paths.get_full_path_or_raise(
            "text_encoders",
            config.get("text_encoder_model")
        )
        vae_path = folder_paths.get_full_path_or_raise(
            "vae",
            config.get("vae_model")
        )
        diffusion_path = folder_paths.get_full_path_or_raise(
            "diffusion_models",
            config.get("diffusion_model")
        )
        audio_encoder_path = folder_paths.get_full_path_or_raise(
            "audio_encoders",
            config.get("audio_encoder_model")
        )

        steps = config.get("steps")
        cfg = config.get("cfg")
        shift = config.get("shift")
        scheduler = config.get("scheduler")
        sampler = config.get("sampler")
        negative_prompt = config.get("negative_prompt")

        chunk_frame_count = config.get("chunk_frame_count", 77)
        chunk_motion_frame_count = config.get("chunk_motion_frame_count", 73)
        batch_latent_t = (chunk_frame_count - 1) // 4 + 1
        batch_frame_count = batch_latent_t * 4
        motion_latent_t = (chunk_motion_frame_count + 3) // 4

        try:
            text_encoder = manager.get_text_encoder(
                paths=[text_encoder_path],
                clip_type=comfy.sd.CLIPType.WAN
            )

            all_conditionings = []
            for idx, batch_input in enumerate(batch_inputs):
                prompt = batch_input["prompt"]

                if prompt["cache_key"] in _CACHE.positive:
                    logger.info(f"Positive prompt cache hit: {prompt['cache_key']}")
                    positive = copy.copy(_CACHE.positive[prompt["cache_key"]])
                else:
                    tokens = text_encoder.tokenize(prompt["value"])
                    positive = text_encoder.encode_from_tokens_scheduled(tokens)
                    _CACHE.positive[prompt["cache_key"]] = positive

                if cfg > 1.0:
                    if prompt["cache_key"] in _CACHE.negative:
                        logger.info(f"Negative prompt cache hit: {prompt['cache_key']}")
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
                    "first_frame": batch_input["first_frame"],
                    "audio": batch_input["audio"],
                    "control_video": batch_input["control_video"],
                })

            audio_encoder = manager.get_audio_encoder(audio_encoder_path)

            for item in all_conditionings:
                audio = item["audio"]
                if audio["value"] is None:
                    item["audio_embed"] = None
                    continue

                if audio["cache_key"] in _CACHE.audio_embed:
                    logger.info(f"Audio embed cache hit: {audio['cache_key']}")
                    item["audio_embed"] = _CACHE.audio_embed[audio["cache_key"]]
                else:
                    audio_embed = audio_encoder.encode_audio(
                        audio["value"]["waveform"], audio["value"]["sample_rate"]
                    )
                    _CACHE.audio_embed[audio["cache_key"]] = audio_embed
                    item["audio_embed"] = audio_embed

            vae = manager.get_vae(vae_path)
            diffusion_model = manager.get_diffusion_model(diffusion_path)
            diffusion_model = utils.patch_model_with_loras(diffusion_model, config)
            diffusion_model = utils.patch_model_sampling(diffusion_model, shift=shift, multiplier=1000.0)

            output_videos = []
            for idx, item in enumerate(all_conditionings):
                if item["audio_embed"] is None:
                    logger.info(f"Audio is None, skipping input {idx+1}")
                    continue

                logger.info(f"Processing input {idx + 1}/{len(all_conditionings)}")

                positive = item["positive"]
                negative = item["negative"]
                first_frame = item["first_frame"]
                audio = item["audio"]
                control_video = item["control_video"]
                audio_embed = item["audio_embed"]

                audio_buckets, num_repeat = cls.bucket_audio_embeds(audio_embed, fps, batch_frame_count)

                audio_duration = audio["value"]["waveform"].shape[2] / audio["value"]["sample_rate"]
                total_target_frames = int(audio_duration * fps)

                if max_frame_count > 0 and total_target_frames > max_frame_count:
                    total_target_frames = max_frame_count
                    num_repeat = (total_target_frames + batch_frame_count - 1) // batch_frame_count
                    logger.info(f"Limiting frames to {max_frame_count}, num_repeat adjusted to {num_repeat}")

                control_videos_buckets = cls.bucket_control_videos(
                    vae, control_video, fps, num_repeat, batch_frame_count, latent_width, latent_height
                )

                logger.info(
                    f"Target frames: {total_target_frames} ({total_target_frames/fps:.2f}s) (num_repeat: {num_repeat})"
                )

                generated_latent_list = []
                ref_motion_latent = None

                for i in range(num_repeat):
                    logger.info(f"Processing input {idx+1} segment {i+1}/{num_repeat}")

                    positive_input = copy.copy(positive)
                    negative_input = copy.copy(negative)

                    latent_empty = torch.zeros(
                        [1, 16, batch_latent_t, latent_height, latent_width],
                        device=comfy.model_management.intermediate_device()
                    )

                    start_index = i * batch_frame_count
                    end_index = i * batch_frame_count + batch_frame_count
                    audio_bucket_embed = audio_buckets[:,:,:,start_index:end_index]
                    positive_input = node_helpers.conditioning_set_values(
                        positive_input, {"audio_embed": audio_bucket_embed}
                    )

                    if first_frame["value"] is not None:
                        first_frame_latent = cls.encode_frame(
                            vae, first_frame, latent_width, latent_height
                        )
                        positive_input = node_helpers.conditioning_set_values(
                            positive_input, {"reference_latents": [first_frame_latent]}
                        )

                    if ref_motion_latent is not None:
                        # i > 0
                        ref_motion_latent = ref_motion_latent[:,:,-motion_latent_t:]
                        positive_input = node_helpers.conditioning_set_values(
                            positive_input, {"reference_motion": ref_motion_latent}
                        )
                    elif first_frame["value"] is not None and isinstance(first_frame["value"], Input.Video):
                        # i = 0
                        ref_motion_latent = cls.encode_motion(
                            vae, first_frame, latent_width, latent_height, chunk_motion_frame_count
                        )
                        positive_input = node_helpers.conditioning_set_values(
                            positive_input, {"reference_motion": ref_motion_latent}
                        )

                    if control_video["value"] is not None:
                        control_video_bucket_latent = control_videos_buckets[i]
                        positive_input = node_helpers.conditioning_set_values(
                            positive_input, {"control_video": control_video_bucket_latent}
                        )
                    else:
                        control_video_bucket_latent = control_videos_buckets[0]
                        positive_input = node_helpers.conditioning_set_values(
                            positive_input, {"control_video": control_video_bucket_latent}
                        )

                    noise = comfy.sample.prepare_noise(latent_empty, seed)
                    callback = latent_preview.prepare_callback(diffusion_model, steps)

                    samples = comfy.sample.sample(
                        diffusion_model,
                        noise,
                        steps=steps,
                        cfg=cfg,
                        sampler_name=sampler,
                        scheduler=scheduler,
                        positive=positive_input,
                        negative=negative_input,
                        latent_image=latent_empty,
                        callback=callback,
                        seed=seed,
                    )

                    if i == 0:
                        # concat video from previous node
                        if ref_motion_latent is not None:
                            decode_latents = torch.cat([
                                ref_motion_latent,
                                samples
                            ], dim=2)
                        else:
                            decode_latents = torch.cat([
                                first_frame_latent,
                                samples
                            ], dim=2)
                    else:
                        decode_latents = torch.cat([ref_motion_latent, samples], dim=2)

                    concat_image = vae.decode(decode_latents)
                    concat_image = concat_image.flatten(0, 1)
                    concat_image = concat_image[-batch_frame_count:]
                    if i == 0:
                        concat_image = concat_image[3:]

                    overlop_frames_num = min(chunk_motion_frame_count, concat_image.shape[0])
                    overlop_concat_image = concat_image[-overlop_frames_num:]
                    ref_motion_latent = vae.encode(overlop_concat_image)

                    output_concat_image = utils.scale_by_width_height(
                        concat_image, target_width, target_height, "bilinear", "center",
                    )
                    generated_latent_list.append(output_concat_image)

                images = torch.cat(generated_latent_list, dim=0)
                images = images[:total_target_frames]
                video = InputImpl.VideoFromComponents(
                    Types.VideoComponents(
                        images=images,
                        audio=audio["value"],
                        frame_rate=Fraction(fps)
                    )
                )
                output_videos.append(video)

        finally:
            if manager.release_after_run:
                manager.release_all()
            elif manager.offload_after_run:
                manager.offload_all()

        return io.NodeOutput(output_videos)

    @classmethod
    def sample_video_frames(cls, video_images, original_fps, target_fps, target_frame_count):
        total_frames = len(video_images)
        interval = max(1, round(original_fps / target_fps))

        sampled_indices = []
        for i in range(target_frame_count):
            idx = i * interval
            if idx < total_frames:
                sampled_indices.append(idx)
            else:
                break

        sampled_frames = [video_images[i] for i in sampled_indices]

        if len(sampled_frames) < target_frame_count:
            sampled_frames = utils.pingpong_video_padding(sampled_frames, target_frame_count)

        return sampled_frames

    @classmethod
    def bucket_control_videos(
        cls,
        vae,
        control_video,
        fps,
        num_repeat,
        batch_frame_count,
        latent_width,
        latent_height
    ):
        control_videos_buckets = []

        if control_video["value"] is None:
            empty_control_video = comfy.latent_formats.Wan21().process_out(torch.zeros(
                [1, 16, batch_frame_count // 4, latent_height, latent_width]
            ))
            control_videos_buckets.append(empty_control_video)
            return control_videos_buckets

        if control_video["cache_key"] in _CACHE.control_video_buckets:
            logger.info(f"Control video buckets cache hit: {control_video['cache_key']}")
            control_videos_buckets = _CACHE.control_video_buckets[control_video["cache_key"]]
            return control_videos_buckets

        video_images = control_video["value"].get_components().images
        original_fps = control_video["value"].get_components().frame_rate

        video_images = utils.scale_by_width_height(
            video_images, latent_width * 8, latent_height * 8, "bilinear", "center"
        )

        n_frames = num_repeat * batch_frame_count
        final_images = cls.sample_video_frames(video_images, original_fps, fps, n_frames)

        cond_tensor = torch.stack(final_images, dim=0)
        cond_tensors = torch.chunk(cond_tensor, num_repeat, dim=0)

        for r in range(len(cond_tensors)):
            cond = cond_tensors[r]
            cond = torch.cat([cond[0:1], cond], dim=0)
            cond_lat = vae.encode(cond)[:, :, 1:].cpu()
            control_videos_buckets.append(cond_lat)

        _CACHE.control_video_buckets[control_video["cache_key"]] = control_videos_buckets

        return control_videos_buckets

    @classmethod
    def encode_motion(cls, vae, frame, latent_width, latent_height, chunk_motion_frame_count):
        if frame["value"] is None and not isinstance(frame["value"], Input.Video):
            raise ValueError("[DALab] WanS2V frame is None or not a video")

        if frame["cache_key"] in _CACHE.motion_latents:
            logger.info(f"Video motion cache hit: {frame['cache_key']}")
            motion_latent = _CACHE.motion_latents[frame["cache_key"]]
        else:
            motion = frame["value"].get_components().images

            if motion.shape[0] > chunk_motion_frame_count:
                motion = motion[-chunk_motion_frame_count:]

            motion = utils.scale_by_width_height(
                motion, latent_width * 8, latent_height * 8, "bilinear", "center"
            )

            if motion.shape[0] < chunk_motion_frame_count:
                r = torch.ones([chunk_motion_frame_count, latent_height * 8, latent_width * 8, 3]) * 0.5
                r[-motion.shape[0]:] = motion
                motion = r

            motion_latent = vae.encode(motion)
            _CACHE.motion_latents[frame["cache_key"]] = motion_latent

        return motion_latent

    @classmethod
    def encode_frame(cls, vae, frame, latent_width, latent_height):
        if frame["value"] is None:
            raise ValueError("[DALab] WanS2V frame is None")
    
        if frame["cache_key"] in _CACHE.image_latents:
            logger.info(f"Image latent cache hit: {frame['cache_key']}")
            frame_latent = _CACHE.image_latents[frame["cache_key"]]
        else:
            frame_value = frame["value"]
            if isinstance(frame_value, Input.Video):
                frame_value = frame_value.get_components().images[-1:]
            
            frame_value_scaled = utils.scale_by_width_height(
                frame_value, latent_width * 8, latent_height * 8, "bilinear", "center"
            )
            frame_latent = vae.encode(frame_value_scaled)
            _CACHE.image_latents[frame["cache_key"]] = frame_latent
        
        return frame_latent

    @classmethod
    def bucket_audio_embeds(cls, audio_embed, fps, batch_frame_count):
        feat = torch.cat(audio_embed["encoded_audio_all_layers"])

        model_video_rate = 30
        
        feat = linear_interpolation(feat, input_fps=50, output_fps=model_video_rate)

        audio_embed_bucket, num_repeat = get_audio_embed_bucket_fps(
            feat, fps=fps, batch_frames=batch_frame_count, m=0, video_rate=model_video_rate
        )
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)

        return audio_embed_bucket,num_repeat

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            config_mtime = os.path.getmtime(_CONFIG_FILE_PATH)
            global_config_mtime = os.path.getmtime(utils.get_config_file_path("global"))
        except:
            config_mtime = 0
            global_config_mtime = 0

        return hash((str(kwargs), str(config_mtime), str(global_config_mtime)))


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = torch.nn.functional.interpolate(
        features, size=output_len, align_corners=True, mode='linear'
    )
    return output_features.transpose(1, 2)


def get_sample_indices(original_fps, total_frames, target_fps, num_sample, fixed_start=None):
    required_duration = num_sample / target_fps
    required_origin_frames = int(np.ceil(required_duration * original_fps))
    if required_duration > total_frames / original_fps:
        raise ValueError("required_duration must be less than video length")

    if fixed_start is not None and fixed_start >= 0:
        start_frame = fixed_start
    else:
        max_start = total_frames - required_origin_frames
        if max_start < 0:
            raise ValueError("video length is too short")
        start_frame = np.random.randint(0, max_start + 1)
    
    start_time = start_frame / original_fps
    end_time = start_time + required_duration
    time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)

    frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices


def get_audio_embed_bucket_fps(audio_embed, fps=16, batch_frames=80, m=0, video_rate=30):
    num_layers, audio_frame_num, audio_dim = audio_embed.shape

    return_all_layers = num_layers > 1
    scale = video_rate / fps

    min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1
    bucket_num = min_batch_num * batch_frames
    padd_audio_num = math.ceil(min_batch_num * batch_frames / fps * video_rate) - audio_frame_num
    
    batch_idx = get_sample_indices(
        original_fps=video_rate,
        total_frames=audio_frame_num + padd_audio_num,
        target_fps=fps,
        num_sample=bucket_num,
        fixed_start=0
    )
    
    batch_audio_eb = []
    audio_sample_stride = int(video_rate / fps)
    
    for bi in batch_idx:
        if bi < audio_frame_num:
            chosen_idx = list(
                range(bi - m * audio_sample_stride, bi + (m + 1) * audio_sample_stride, audio_sample_stride)
            )
            chosen_idx = [0 if c < 0 else c for c in chosen_idx]
            chosen_idx = [audio_frame_num - 1 if c >= audio_frame_num else c for c in chosen_idx]

            if return_all_layers:
                frame_audio_embed = audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
            else:
                frame_audio_embed = audio_embed[0][chosen_idx].flatten()
        else:
            if not return_all_layers:
                frame_audio_embed = torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device)
            else:
                frame_audio_embed = torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
        batch_audio_eb.append(frame_audio_embed)
    
    batch_audio_eb = torch.cat([c.unsqueeze(0) for c in batch_audio_eb], dim=0)
    
    return batch_audio_eb, min_batch_num
