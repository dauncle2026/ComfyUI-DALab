import os
import types
import torch
import torchaudio
import logging
import copy
from fractions import Fraction
from types import SimpleNamespace
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor

import folder_paths
import node_helpers
import latent_preview
import comfy.sd
import comfy.utils
import comfy.sample
import comfy.clip_vision
import comfy.model_management
import comfy.ops
import comfy.model_patcher
import comfy.model_detection
from comfy_api.latest import io, Input, InputImpl, Types
from comfy_api.torch_helpers import set_torch_compile_wrapper

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..libs.infinite_talk.wav2vec2 import Wav2Vec2Model
from ..libs.infinite_talk.infinite_model import (
    AudioProjModel,
    SingleStreamMutiAttention,
    infinite_block_forward,
    infinite_block_self_attn_forward,
    infinite_forward
)

_CONFIG_FILE_PATH = utils.get_config_file_path("infinite_talk")

_CACHE = SimpleNamespace(
    positive={},
    negative={},
    audio_embed={},
    clip_vision_embeds={},
    image_latents={},
    first_latents={},
)

class DAInfiniteTalkConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        diffusion_model_options = folder_paths.get_filename_list("diffusion_models")
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
                    "infinite_model",
                    default=config.get("infinite_model_name", "infinite-talk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"),
                    options=diffusion_model_options,
                    display_name="infinite_model",
                    tooltip="The InfiniteTalk checkpoint to merge.",
                ),
                io.Combo.Input(
                    "vae_model",
                    default=config.get("vae_model", "wan/wan_2.1_vae.safetensors"),
                    options=vae_options,
                    display_name="vae_model",
                ),
                io.Combo.Input(
                    "text_encoder",
                    default=config.get("clip_name", "wan/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
                    options=text_encoder_options,
                    display_name="text_encoder",
                ),
                io.Combo.Input(
                    "clip_vision_model",
                    default=config.get("clip_vision_model", "infinite_talk/clip_vision_h.safetensors"),
                    options=clip_vision_options,
                    display_name="clip_vision_model",
                ),
                io.String.Input(
                    "audio_encoder",
                    default=config.get("audio_encoder_path", audio_encoder_path),
                    display_name="audio_encoder",
                    tooltip="The audio encoder path. Default: TencentGameMate/chinese-wav2vec2-base",
                ),
                io.Boolean.Input(
                    "auto_offload",
                    default=config.get("auto_unload", False),
                    display_name="auto_offload",
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
                    "frame_count",
                    default=config.get("frame_count", 81),
                    min=1, 
                    max=200,
                    display_name="frame_count",
                    tooltip="Total frames per generation window.",
                ),
                io.Int.Input(
                    "motion_frame_count",
                    default=config.get("motion_frame_count", 9),
                    min=0, 
                    max=50,
                    display_name="motion_frame_count",
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
        infinite_model, 
        vae_model, 
        text_encoder, 
        clip_vision_model, 
        audio_encoder,
        auto_offload, 
        torch_compile, 
        steps, 
        cfg, 
        shift, 
        prefix_silence_seconds, 
        frame_count, 
        motion_frame_count, 
        sampler, 
        scheduler, 
        negative_prompt, 
        loras
    ) -> io.NodeOutput:
        config_data = {
            "base_model": base_model,
            "infinite_model": infinite_model,
            "vae_model": vae_model,
            "text_encoder": text_encoder,
            "clip_vision_model": clip_vision_model,
            "audio_encoder": audio_encoder,
            "auto_offload": auto_offload,
            "torch_compile": torch_compile,
            "steps": steps,
            "cfg": cfg,
            "shift": shift,
            "prefix_silence_seconds": prefix_silence_seconds,
            "frame_count": frame_count,
            "motion_frame_count": motion_frame_count,
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
        _CACHE.first_latents.clear()

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
                    step=16, 
                    display_mode=io.NumberDisplay.number,
                    tooltip="The width of the video. Default: 640",
                    display_name="width",
                ),
                io.Int.Input(
                    "height", 
                    default=640, 
                    min=16, 
                    max=2048, 
                    step=16, 
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
        first_frames = None,
        audio_options = None,
        from_videos = None,
    ) -> io.NodeOutput:
        scale_factor = 8
        target_height = height[0]
        target_width = width[0]
        latent_height = (target_height + scale_factor - 1) // scale_factor
        latent_width = (target_width + scale_factor - 1) // scale_factor

        seed = seed[0]

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
        
        base_model_path = folder_paths.get_full_path_or_raise("diffusion_models", config.get("base_model"))
        infinite_model_path = folder_paths.get_full_path_or_raise("diffusion_models", config.get("infinite_model"))
        vae_path = folder_paths.get_full_path_or_raise("vae", config.get("vae_model"))
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", config.get("text_encoder"))
        clip_vision_path = folder_paths.get_full_path_or_raise("clip_vision", config.get("clip_vision_model"))
        audio_encoder_path = config.get("audio_encoder")

        if not os.path.exists(audio_encoder_path):
            raise ValueError(f"Audio encoder path {audio_encoder_path} not found")
        
        auto_unload = config.get("auto_offload")
        torch_compile = config.get("torch_compile")
        steps = config.get("steps")
        cfg = config.get("cfg")
        shift = config.get("shift")
        negative_prompt = config.get("negative_prompt")
        sampler = config.get("sampler")
        scheduler = config.get("scheduler")
        prefix_silence_seconds = config.get("prefix_silence_seconds")
        frame_count = config.get("frame_count")
        motion_frame_count = config.get("motion_frame_count")

        latent_t = (frame_count - 1) // 4 + 1
        
        text_encoder = comfy.sd.load_clip(
            ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.WAN
        )
        clip_vision = comfy.clip_vision.load(clip_vision_path)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
        diffusion_model = create_infinite_model(base_model_path, infinite_model_path)

        device = diffusion_model.load_device

        wav2vec = Wav2Vec2Model.from_pretrained(audio_encoder_path).float().to(device).eval()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_encoder_path, local_files_only=True)
        
        diffusion_model = utils.patch_model_with_loras(diffusion_model, config)
        diffusion_model = utils.patch_model_sampling(diffusion_model, shift, 1000.0)

        if torch_compile:
             set_torch_compile_wrapper(
                model=diffusion_model, backend="inductor", options={"guard_filter_fn": skip_torch_compile_dict}
            )

        cls.reset_cache()
        output_videos = []
        for idx, input in enumerate(batch_inputs):
            logging.info(f"[DALab] InfiniteTalk processing {idx + 1}/{len(batch_inputs)}")
            
            prompt = input["prompt"]
            first_frame = input["first_frame"]
            audio_option = input["audio_option"]
            from_video = input["from_video"]
            fps = input["audio_option"]["fps"]["value"]

            if fps is None:
                raise ValueError("[DALab] InfiniteTalk FPS is not set")

            if first_frame["value"] is None and from_video["value"] is None:
                logging.info(f"[DALab] InfiniteTalk first frame and from video are None, skipping : {idx+1}")
                continue
            
            if prompt["cache_key"] in _CACHE.positive:
                logging.info(f"[DALab] InfiniteTalk positive prompt cache hit: {prompt['cache_key']}")
                positive = copy.copy(_CACHE.positive[prompt["cache_key"]])
            else:
                tokens = text_encoder.tokenize(prompt["value"])
                positive = text_encoder.encode_from_tokens_scheduled(tokens)
                _CACHE.positive[prompt["cache_key"]] = positive

            if cfg > 1.0:
                if prompt["cache_key"] in _CACHE.negative:
                    logging.info(f"[DALab] InfiniteTalk negative prompt cache hit: {prompt['cache_key']}")
                    negative = copy.copy(_CACHE.negative[prompt["cache_key"]])
                else:
                    negative_tokens = text_encoder.tokenize(negative_prompt)
                    negative = text_encoder.encode_from_tokens_scheduled(negative_tokens)
                    _CACHE.negative[prompt["cache_key"]] = negative
            else:
                negative = []

            positive_input = copy.copy(positive)
            negative_input = copy.copy(negative)
            
            if auto_unload:
                utils.unload_model(text_encoder)
            
            if audio_option["audio_options"]["value"] == "one_person":
                if audio_option["audios"]["value"] is None:
                    logging.info(f"[DALab] InfiniteTalk audio is None, skipping : {idx+1}")
                    continue
            else:
                if audio_option["person1_audios"]["value"] is None and audio_option["person2_audios"]["value"] is None:
                    logging.info(f"[DALab] InfiniteTalk audio is None, skipping : {idx+1}")
                    continue

            person1_audio_embed, person2_audio_embed, combine_audio = cls.encode_audio_options(
                wav2vec, wav2vec_feature_extractor, audio_option, fps, prefix_silence_seconds
            )
            
            work_model = diffusion_model

            if person1_audio_embed is not None and person2_audio_embed is not None:
                split_position = audio_option["split_position"]["value"]
                ref_target_masks = cls.create_audio_mask(latent_height, latent_width, split_position)
                if ref_target_masks.isnan().any():
                    raise ValueError("Ref target masks is NaN")

                work_model.model_options["transformer_options"]["human_num"] = 2
                work_model.model_options["transformer_options"]["ref_target_masks"] = ref_target_masks.to(device)
                total_audio_frames = min(person1_audio_embed.shape[0], person2_audio_embed.shape[0])
            else:
                work_model.model_options["transformer_options"]["human_num"] = 1
                work_model.model_options["transformer_options"]["ref_target_masks"] = None
                total_audio_frames = person1_audio_embed.shape[0]

            current_audio_idx = 0
            generated_frames_list = []
            previous_latent_motion = None

            while current_audio_idx < total_audio_frames:
                logging.info(f"[DALab] InfiniteTalk processing prompt:{idx+1} audio index: {current_audio_idx}/{total_audio_frames}")

                current_frame = first_frame
                if from_video["value"] is not None:
                    logging.info(f"[DALab] InfiniteTalk use video at frame {current_audio_idx}")
                    current_frame = cls.get_current_frame(from_video,current_audio_idx)

                current_frame_clip_embed = cls.clip_vision_encode(clip_vision, current_frame, latent_width, latent_height)
                positive_input = node_helpers.conditioning_set_values(
                    positive_input, {"clip_vision_output": current_frame_clip_embed}
                )

                concat_latent_image, concat_latent_mask = cls.encode_latent_frame(
                    vae, current_frame, latent_t, latent_width, latent_height
                )
                positive_input = node_helpers.conditioning_set_values(
                    positive_input, {"concat_latent_image": concat_latent_image, "concat_mask": concat_latent_mask}
                )

                indices = (torch.arange(2 * 2 + 1) - 2) * 1 
                center_indices = torch.arange(
                    current_audio_idx, 
                    current_audio_idx + frame_count, 
                    1
                ).unsqueeze(1) + indices.unsqueeze(0)
                
                center_indices = torch.clamp(center_indices, min=0, max=total_audio_frames - 1)

                if person1_audio_embed is not None and person2_audio_embed is not None:
                    current_audio_emb = torch.cat([
                        person1_audio_embed[center_indices].unsqueeze(0), 
                        person2_audio_embed[center_indices].unsqueeze(0)
                    ],dim=0).to(device)
                elif person1_audio_embed is not None:
                    current_audio_emb = person1_audio_embed[center_indices].unsqueeze(0).to(device)
                else:
                    raise ValueError("[DALab] InfiniteTalk No audio embed found")

                current_audio_emb = current_audio_emb.to(work_model.model.manual_cast_dtype)
                work_model.model_options["transformer_options"]["audio_cond"] = current_audio_emb

                motion_lat_t = ((motion_frame_count - 1) // 4) + 1 if current_audio_idx != 0 else 1
                empty_latent = torch.zeros((1, 16, latent_t, latent_height, latent_width))
                
                if current_audio_idx == 0:
                    first_latent = cls.encode_first_latent(vae, current_frame, latent_width, latent_height)
                    first_latent = diffusion_model.model.process_latent_in(first_latent)
                else:
                    inject_len = min(previous_latent_motion.shape[2], motion_lat_t)
                    first_latent = previous_latent_motion[:, :, :inject_len, :, :]
                    first_latent = diffusion_model.model.process_latent_in(first_latent)

                work_model.model_options["transformer_options"]["first_latent"] = first_latent

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

                samples[:,:,:motion_lat_t] = diffusion_model.model.process_latent_out(first_latent)
                
                decoded_images = vae.decode(samples)
                decoded_images = decoded_images.flatten(0, 1)

                if decoded_images.shape[2] != target_height or decoded_images.shape[3] != target_width:
                    output_decoded_images = utils.scale_by_width_height(decoded_images, target_width, target_height, "bilinear", "center")
                else:
                    output_decoded_images = decoded_images

                if current_audio_idx == 0:
                    generated_frames_list.append(output_decoded_images)
                else:
                    generated_frames_list.append(output_decoded_images[motion_frame_count:])
                
                previous_latent_motion = vae.encode(decoded_images[-motion_frame_count:])
                current_audio_idx += frame_count - motion_frame_count

                if auto_unload:
                    utils.unload_model(vae)

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
        background_mask = torch.zeros([latent_height, latent_width])

        left_margin = int(split_x * face_scale)
        
        lefty_min = left_margin
        lefty_max = split_x - left_margin

        right_panel_width = latent_width - split_x
        right_margin = int(right_panel_width * face_scale)

        righty_min = split_x + right_margin
        righty_max = latent_width - right_margin

        human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
        human_mask2[x_min:x_max, righty_min:righty_max] = 1

        background_mask += human_mask1
        background_mask += human_mask2

        human_masks = [human_mask1, human_mask2]
        background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
        human_masks.append(background_mask)

        ref_target_masks = torch.stack(human_masks, dim=0)

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
            logging.info(f"[DALab] InfiniteTalk clip vision embed cache hit: {frame['cache_key']}")
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
    def encode_first_latent(cls, vae, frame, latent_width, latent_height):
        if frame["value"] is None:
            raise ValueError("Frame is None")
        
        if frame["cache_key"] in _CACHE.first_latents:
            logging.info(f"[DALab] InfiniteTalk first frame latent cache hit: {frame['cache_key']}")
            first_frame_latent = _CACHE.first_latents[frame["cache_key"]]
        else:
            frame_value = frame["value"]
            if isinstance(frame_value, Input.Video):
                frame_value = frame_value.get_components().images[-1:]
            
            frame_value_scaled = utils.scale_by_width_height(
                frame_value, latent_width * 8, latent_height * 8, "bilinear", "center"
            )

            first_frame_latent = vae.encode(frame_value_scaled)
            _CACHE.first_latents[frame["cache_key"]] = first_frame_latent

        return first_frame_latent
        
    @classmethod
    def encode_latent_frame(cls, vae, frame, latent_t, latent_width, latent_height):
        if frame["value"] is None:
            raise ValueError("Frame is None")
        
        init_image = torch.ones(((latent_t - 1)*4+1, latent_height * 8, latent_width * 8, 3)) * 0.5
        init_mask = torch.ones(
            (1, 1, latent_t * 4, latent_height, latent_width)
        )

        if frame["cache_key"] in _CACHE.image_latents:
            logging.info(f"[DALab] InfiniteTalk image latent cache hit: {frame['cache_key']}")
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

        audio_input = audio_input[0][0] # batch=1, channels=1
        
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

        audio_embeddings = wav2vec2(
            audio_features, seq_len=video_length, output_hidden_states=True
        )
        
        stacked = torch.stack(audio_embeddings.hidden_states[1:], dim=1).squeeze(0) # [Layers, SeqLen, Dim]
        final_emb = rearrange(stacked, "l s d -> s l d") # [SeqLen, Layers, Dim]
        
        return final_emb
    
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            config_mtime = os.path.getmtime(_CONFIG_FILE_PATH)
        except:
            config_mtime = 0
            
        return hash((str(kwargs),str(config_mtime)))

def create_infinite_model(base_model_path, infinite_model_path):
    sd, metadata = comfy.utils.load_torch_file(base_model_path, return_metadata=True)
    infinite_sd = comfy.utils.load_torch_file(infinite_model_path)
    sd.update(infinite_sd)

    sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
    weight_dtype = comfy.utils.weight_dtype(sd)
    parameters = comfy.utils.calculate_parameters(sd)

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    model_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    unet_dtype = comfy.model_management.unet_dtype(
        model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype
    )

    dtype = unet_dtype
    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(dtype=dtype, manual_cast_dtype=manual_cast_dtype)
    model = model_config.get_model(sd, "")

    operations = comfy.ops.pick_operations(
        weight_dtype, manual_cast_dtype, model_config=model_config
    )
    multitalk_proj_model = create_multitalk_proj_model(operations=operations)

    dim = model.diffusion_model.dim
    num_heads = model.diffusion_model.num_heads
    eps = model.diffusion_model.eps

    for block in model.diffusion_model.blocks:
        block.norm_x = operations.LayerNorm(dim, eps, elementwise_affine=True)
        block.audio_cross_attn = SingleStreamMutiAttention(
            dim=dim,
            encoder_hidden_states_dim=768,
            num_heads=num_heads,
            qkv_bias=True,
            class_range=24,
            class_interval=4,
            operations=operations
        )
        block.forward = types.MethodType(infinite_block_forward, block)
        block.self_attn.forward = types.MethodType(infinite_block_self_attn_forward, block.self_attn)

    model.diffusion_model.audio_proj = multitalk_proj_model
    model.diffusion_model.forward_orig = types.MethodType(infinite_forward, model.diffusion_model)

    model = model.to(offload_device)
    model.load_model_weights(sd, "")

    return comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)

def create_multitalk_proj_model(operations=None):
    audio_window=5
    intermediate_dim=512
    output_dim=768
    context_tokens=32
    vae_scale=4
    norm_output_audio = True

    multitalk_proj_model = AudioProjModel(
        seq_len=audio_window,
        seq_len_vf=audio_window+vae_scale-1,
        intermediate_dim=intermediate_dim,
        output_dim=output_dim,
        context_tokens=context_tokens,
        norm_output_audio=norm_output_audio,
        operations=operations,
    )
    return multitalk_proj_model

def skip_torch_compile_dict(guard_entries):
    return [("transformer_options" not in entry.name) for entry in guard_entries]