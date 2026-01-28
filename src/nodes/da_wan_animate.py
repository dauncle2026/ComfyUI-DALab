'''
@author: 
@date: 2026-01-25
@description: This node is used to generate animated videos using the Wan2.2 Animate model with pose/face control.

required:
- cv2
'''
import os
import torch
import copy
import json
import numpy as np
from tqdm import tqdm
from fractions import Fraction
from types import SimpleNamespace

import folder_paths
import node_helpers
import latent_preview
import comfy.sd
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.model_management
from comfy_api.latest import io, InputImpl, Types

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.model_manager import ModelManager
from ..utils.logger import logger
from .da_florence2 import DAFlorence2
from .da_sam2 import DASAM2
from .da_dwpose import DADWPose
from ..utils.paths import get_config_file_path

_CONFIG_FILE_PATH = get_config_file_path("wan_animate")

_CACHE = SimpleNamespace(
    positive={},
    negative={},
    clip_vision_output={},
    concat_latents={},
)

FACE_SCALE_WIDTH = 512
FACE_SCALE_HEIGHT = 512

class DAWanAnimateConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        text_encoder_options = folder_paths.get_filename_list("text_encoders")
        vae_options = folder_paths.get_filename_list("vae")
        diffusion_model_options = folder_paths.get_filename_list("diffusion_models")
        lora_options = folder_paths.get_filename_list("loras")
        clip_vision_options = folder_paths.get_filename_list("clip_vision")

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)
        loras = utils.dynamic_combo_loras(config, lora_options)

        return io.Schema(
            node_id="DAWanAnimateConfig",
            display_name="DA Wan2.2 Animate Config",
            category="DALab/Video/Wan2.2 Animate",
            description="Configure the Wan2.2 Animate model params. Run first to save config.",
            is_output_node=True,
            inputs=[
                io.Combo.Input(
                    "text_encoder",
                    default=config.get("text_encoder", "wan/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
                    options=text_encoder_options,
                    display_name="text_encoder",
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
                    default=config.get("diffusion_model", "wan2.2/wan2.2_animate_14B_fp8_scaled.safetensors"),
                    options=diffusion_model_options,
                    display_name="diffusion_model",
                    tooltip="The Wan2.2 Animate Diffusion Model (14B).",
                ),
                io.Combo.Input(
                    "clip_vision_model",
                    default=config.get("clip_vision_model", "clip_vision_h.safetensors"),
                    options=clip_vision_options,
                    display_name="clip_vision_model",
                    tooltip="The CLIP Vision model for reference image encoding.",
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
                io.Int.Input(
                    "chunk_frame_count",
                    default=config.get("chunk_frame_count", 77),
                    min=1,
                    max=1000,
                    step=1,
                    tooltip="Number of frames for each round. Default 77.",
                    display_name="chunk_frame_count",
                ),
                io.Int.Input(
                    "chunk_motion_frame_count",
                    default=config.get("chunk_motion_frame_count", 5),
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Number of frames for motion. Default 5.",
                    display_name="chunk_motion_frame_count",
                ),
                io.Int.Input(
                    "sam_mask_grow_ratio",
                    default=config.get("sam_mask_grow_ratio", 5),
                    min=1,
                    max=1000,
                    step=1,
                    tooltip="SAM mask grow ratio. Default 5.",
                    display_name="sam_mask_grow_ratio",
                ),
                io.Int.Input(
                    "lego_mask_w_div",
                    default=config.get("lego_mask_w_div", 10),
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Lego mask box size in width.越大越精细 Default 10.",
                    display_name="lego_mask_w_div",
                ),
                io.Int.Input(
                    "lego_mask_h_div",
                    default=config.get("lego_mask_h_div", 20),
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Lego mask height division. Default 20.",
                    display_name="lego_mask_h_div",
                ),
                io.Float.Input(
                    "face_crop_scale",
                    default=config.get("face_crop_scale", 2.0),
                    min=1.0,
                    max=5.0,
                    step=0.1,
                    tooltip="Face crop box scale multiplier. Larger = bigger crop area. Default 2.0.",
                    display_name="face_crop_scale",
                ),
                io.Float.Input(
                    "face_crop_y_offset",
                    default=config.get("face_crop_y_offset", 0.0),
                    min=-1.0,
                    max=1.0,
                    step=0.05,
                    tooltip="Face crop Y offset ratio. Positive = move down (include more neck/shoulders). Default 0.0.",
                    display_name="face_crop_y_offset",
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
        text_encoder,
        vae_model,
        diffusion_model,
        clip_vision_model,
        steps,
        cfg,
        shift,
        scheduler,
        sampler,
        sam_mask_grow_ratio,
        lego_mask_w_div,
        lego_mask_h_div,
        face_crop_scale: float,
        face_crop_y_offset: float,
        negative_prompt: str,
        chunk_frame_count: int,
        chunk_motion_frame_count: int,
        loras
    ) -> io.NodeOutput:

        config_data = {
            "text_encoder": text_encoder,
            "vae_model": vae_model,
            "diffusion_model": diffusion_model,
            "clip_vision_model": clip_vision_model,
            "steps": steps,
            "cfg": cfg,
            "shift": shift,
            "scheduler": scheduler,
            "sampler": sampler,
            "sam_mask_grow_ratio": sam_mask_grow_ratio,
            "lego_mask_w_div": lego_mask_w_div,
            "lego_mask_h_div": lego_mask_h_div,
            "face_crop_scale": face_crop_scale,
            "face_crop_y_offset": face_crop_y_offset,
            "negative_prompt": negative_prompt,
            "chunk_frame_count": chunk_frame_count,
            "chunk_motion_frame_count": chunk_motion_frame_count,
            "loras": loras
        }

        utils.save_json(config_data, _CONFIG_FILE_PATH)

        return io.NodeOutput()

class DAWanAnimate(io.ComfyNode):
    @classmethod
    def reset_cache(cls):
        _CACHE.positive.clear()
        _CACHE.negative.clear()
        _CACHE.clip_vision_output.clear()
        _CACHE.concat_latents.clear()

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DAWanAnimate",
            display_name="DA Wan2.2 Animate",
            category="DALab/Video/Wan2.2 Animate",
            description="Generate animated videos using the Wan2.2 Animate model with pose/face control.",
            is_input_list=True,
            is_output_node=True,
            inputs=[
                io.Int.Input(
                    "width",
                    default=640,
                    min=16,
                    max=2560,
                    step=8,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Width of the output video. Recommended: 832.",
                    display_name="width",
                ),
                io.Int.Input(
                    "height",
                    default=360,
                    min=16,
                    max=2560,
                    step=8,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Height of the output video. Recommended: 480.",
                    display_name="height",
                ),
                io.Int.Input(
                    "max_frame_count",
                    default=-1,
                    min=-1,
                    max=2048,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Max frame count of the output video. Default -1 (no limit).",
                    display_name="max_frame_count",
                ),
                io.Float.Input(
                    "fps",
                    default=16.0,
                    min=1,
                    max=60,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Frames per second. Default 16.0.",
                    display_name="fps",
                ),
                io.Combo.Input(
                    "detect_types",
                    default="default",
                    options=["default","advanced"],
                    tooltip="Select the detect type.",
                    display_name="detect_types",
                ),
                io.Int.Input(
                    "character_ids",
                    default=-1,
                    min=-1,
                    max=200,
                    step=1,
                    display_mode=io.NumberDisplay.number,
                    tooltip="Character ID. Default 0.",
                    display_name="character_ids",
                ),
                io.String.Input(
                    "prompts",
                    multiline=True,
                    default="A beautiful girl dancing",
                    tooltip="Text prompts describing the animation.",
                    display_name="prompts",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Random seed for generation.",
                    display_name="seed",
                ),
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Reference image of the character.",
                    display_name="images",
                ),
                io.Video.Input(
                    "videos",
                    optional=True,
                    tooltip="Background video.",
                    display_name="videos",
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
        fps: list[float],
        max_frame_count: list[int],
        detect_types: list[str],
        character_ids: list[int],
        prompts: list[str],
        seed: list[int],
        images=None,
        videos=None,
    ) -> io.NodeOutput:
        scale_factor = 8
        target_width = width[0]
        target_height = height[0]
        latent_height = ((target_height + (scale_factor * 2) - 1) // (scale_factor * 2)) * 2
        latent_width =  ((target_width  + (scale_factor * 2) - 1) // (scale_factor * 2)) * 2
        aligen_width = latent_width * scale_factor
        aligen_height = latent_height * scale_factor

        fps = int(fps[0])
        seed = seed[0]
        max_frame_count = max_frame_count[0]

        batch_inputs = utils.inputs_to_batch(
            defaults={"prompt": "", "character_id": -1, "detect_type": "default"},
            prompt=prompts,
            reference_image=images,
            video=videos,
            character_id=character_ids,
            detect_type=detect_types,
        )

        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)

        clip_path = folder_paths.get_full_path_or_raise(
            "text_encoders",
            config.get("text_encoder")
        )
        vae_path = folder_paths.get_full_path_or_raise(
            "vae",
            config.get("vae_model")
        )
        diffusion_model_path = folder_paths.get_full_path_or_raise(
            "diffusion_models",
            config.get("diffusion_model")
        )
        clip_vision_path = folder_paths.get_full_path_or_raise(
            "clip_vision",
            config.get("clip_vision_model")
        )

        cfg = config.get("cfg")
        shift = config.get("shift")
        steps = config.get("steps")
        scheduler = config.get("scheduler")
        sampler = config.get("sampler")
        negative_prompt = config.get("negative_prompt")
        chunk_frame_count = config.get("chunk_frame_count") 
        chunk_motion_frame_count = config.get("chunk_motion_frame_count")

        batch_latent_t = (chunk_frame_count - 1) // 4 + 1
        batch_motion_latent_t = (chunk_motion_frame_count + 3) // 4

        manager = ModelManager()
        cls.reset_cache()

        output_videos = []
        try:
            text_encoder = manager.get_text_encoder(
                paths=[clip_path],
                clip_type=comfy.sd.CLIPType.WAN
            )

            all_conditionings = []
            for idx, input in enumerate(batch_inputs):
                logger.info(f"Wan2.2 Animate (text encoding) {idx + 1}/{len(batch_inputs)}")

                prompt = input["prompt"]
                reference_image = input["reference_image"]
                character_id = input["character_id"]["value"]
                detect_type = input["detect_type"]["value"]
                video = input["video"]["value"]

                if video is None:
                    logger.info(f"Wan2.2 Animate processing video is None {idx + 1}/{len(batch_inputs)}")
                    continue

                if character_id == -1:
                    logger.info(f"Wan2.2 Animate processing character has -1 need manual detection {idx + 1}/{len(batch_inputs)}")
                    output = cls.execute_florence(video, detect_type, character_id, seed)
                    return io.NodeOutput([], ui=output.ui)

                if reference_image["value"] is None:
                    logger.info(f"Reference image is None, skipping : {idx+1}")
                    continue

                logger.info(f"Wan2.2 Animate processing with character id: {character_id},{idx + 1}/{len(batch_inputs)}")
                pose_video, face_video, background_video, character_mask, orgin_fps = cls.get_videos(
                    video, detect_type, character_id, seed, config
                )
                if pose_video is None:
                    logger.warning(f"Failed to get videos for character_id: {character_id}, skipping")
                    continue

                positive, negative = cls.encode_text(text_encoder, prompt, negative_prompt, cfg)

                all_conditionings.append({
                    "positive": positive,
                    "negative": negative,
                    "reference_image": reference_image,
                    "video": input["video"],
                    "pose_video": pose_video,
                    "face_video": face_video,
                    "background_video": background_video,
                    "character_mask": character_mask,
                    "orgin_fps": orgin_fps,
                })

            clip_vision = manager.get_clip_vision(clip_vision_path)

            for idx, item in enumerate(all_conditionings):
                logger.info(f"Wan2.2 Animate (CLIP Vision encoding) {idx + 1}/{len(all_conditionings)}")
                reference_image = item["reference_image"]
                item["clip_vision_output"] = cls.clip_vision_encode(clip_vision, reference_image)

            vae = manager.get_vae(vae_path)
            diffusion_model = manager.get_diffusion_model(diffusion_model_path)
            diffusion_model = utils.patch_model_with_loras(diffusion_model, config)
            diffusion_model = utils.patch_model_sampling(diffusion_model, shift=shift, multiplier=1000.0)

            for idx, item in enumerate(all_conditionings):
                logger.info(f"Wan2.2 Animate (sampling) {idx + 1}/{len(all_conditionings)}")

                positive = item["positive"]
                negative = item["negative"]
                reference_image = item["reference_image"]
                video = item["video"]["value"]
                pose_video = item["pose_video"]
                face_video = item["face_video"]
                background_video = item["background_video"]
                character_mask = item["character_mask"]
                orgin_fps = item["orgin_fps"]
                clip_vision_output = item["clip_vision_output"]

                ref_image_latent, ref_image_latent_mask = cls.encode_ref_image(
                    vae, reference_image, latent_width, latent_height
                )
                trim_latent_t = ref_image_latent.shape[2]

                if clip_vision_output is not None:
                    positive = node_helpers.conditioning_set_values(
                        positive, {"clip_vision_output": clip_vision_output}
                    )

                pose_video_pixels = utils.scale_by_width_height(pose_video, aligen_width, aligen_height, "area", "center")
                face_video_pixels = utils.scale_by_width_height(face_video, FACE_SCALE_WIDTH, FACE_SCALE_HEIGHT, "area", "center")
                character_mask_pixels = cls.character_mask_scale(character_mask, latent_width, latent_height)
                bg_video_pixels = utils.scale_by_width_height(background_video, aligen_width, aligen_height, "area", "center")

                if pose_video_pixels is None:
                    logger.info(f"Pose video is None, skipping : {idx+1}")
                    continue

                input_frame_count = pose_video_pixels.shape[0]
                if max_frame_count > 0 and input_frame_count > max_frame_count:
                    input_frame_count = max_frame_count
                    logger.info(f"Limiting input frames to {max_frame_count}")

                total_frames = utils.get_valid_len(
                    input_frame_count,
                    chunk_frame_count,
                    chunk_motion_frame_count
                )

                videos_to_pad = {
                    'pose': pose_video_pixels,
                    'face': face_video_pixels,
                    'mask': character_mask_pixels,
                    'bg': bg_video_pixels,
                }
                for key, video_pixels in videos_to_pad.items():
                    if video_pixels is not None:
                        videos_to_pad[key] = utils.pingpong_video_padding(video_pixels, total_frames)

                pose_video_pixels = videos_to_pad['pose']
                face_video_pixels = videos_to_pad['face']
                character_mask_pixels = videos_to_pad['mask']
                bg_video_pixels = videos_to_pad['bg']

                logger.info(f"WanAnimate summary {idx+1} total frames: {total_frames},org_fps:{orgin_fps},fps:{fps}")

                curr_frame_idx = 0
                pre_motion_pixels = None
                generated_frames_list = []
                while curr_frame_idx < total_frames:
                    logger.info(f"Wan2.2 Animate processing: {idx+1}/{len(all_conditionings)}  part frame:{curr_frame_idx}/{total_frames}")

                    concat_image = torch.ones(
                        (chunk_frame_count, latent_height * 8, latent_width * 8, 3),
                    ) * 0.5
                    concat_latent_mask = torch.ones((1, 1, batch_latent_t * 4, latent_height, latent_width))

                    concat_image_start = 0
                    if curr_frame_idx > 0:
                        concat_image[:chunk_motion_frame_count] = pre_motion_pixels[-chunk_motion_frame_count:]
                        concat_latent_mask[:, :, :batch_motion_latent_t * 4] = 0.0

                        concat_image_start += chunk_motion_frame_count

                    pose_face_start = curr_frame_idx - concat_image_start
                    pose_face_end = pose_face_start + chunk_frame_count

                    bg_mask_start = curr_frame_idx
                    bg_mask_end = bg_mask_start + chunk_frame_count - concat_image_start

                    if bg_video_pixels is not None:
                        bg_video = torch.stack(bg_video_pixels[bg_mask_start:bg_mask_end])
                        concat_image[concat_image_start:] = bg_video

                    if character_mask_pixels is not None:
                        char_mask = torch.stack(character_mask_pixels[bg_mask_start:bg_mask_end])
                        char_mask = torch.cat((char_mask[0:1].repeat(4, 1, 1, 1, 1), char_mask[1:]))
                        char_mask = char_mask.movedim(0, 2)
                        concat_latent_mask[:, :, concat_image_start:] = char_mask

                    concat_image_latent = vae.encode(concat_image)
                    concat_image_latent = torch.cat((ref_image_latent, concat_image_latent),dim=2)

                    concat_latent_mask = concat_latent_mask.view(1, concat_latent_mask.shape[2]//4, 4, latent_height, latent_width).transpose(1, 2)
                    concat_latent_mask = torch.cat((ref_image_latent_mask, concat_latent_mask), dim=2)

                    positive = node_helpers.conditioning_set_values(
                        positive, {"concat_latent_image": concat_image_latent, "concat_mask": concat_latent_mask}
                    )

                    if pose_video_pixels is not None:
                        pose_video = torch.stack(pose_video_pixels[pose_face_start:pose_face_end])
                        pose_video_latent = vae.encode(pose_video)
                        positive = node_helpers.conditioning_set_values(
                            positive, {"pose_video_latent": pose_video_latent}
                        )

                    if face_video_pixels is not None:
                        face_video = torch.stack(face_video_pixels[pose_face_start:pose_face_end])
                        face_video = face_video.permute(3, 0, 1, 2).unsqueeze(0)
                        face_video = face_video * 2.0 - 1.0
                        positive = node_helpers.conditioning_set_values(
                            positive, {"face_video_pixels": face_video}
                        )

                    latent_image = torch.zeros(
                        [1, 16, batch_latent_t + trim_latent_t, latent_height, latent_width],
                        device=comfy.model_management.intermediate_device()
                    )
                    noise = comfy.sample.prepare_noise(latent_image, seed)
                    callback = latent_preview.prepare_callback(diffusion_model, steps)

                    samples = comfy.sample.sample(
                        diffusion_model,
                        noise,
                        steps=steps,
                        cfg=cfg,
                        sampler_name=sampler,
                        scheduler=scheduler,
                        positive=positive,
                        negative=negative,
                        latent_image=latent_image,
                        callback=callback,
                        seed=seed,
                    )

                    decoded_images = vae.decode(samples[:,:,trim_latent_t:])
                    decoded_images = decoded_images.flatten(0, 1)

                    output_decoded_images = utils.scale_by_width_height(
                        decoded_images, target_width, target_height, "bilinear", "center"
                    )

                    if curr_frame_idx == 0:
                        generated_frames_list.append(output_decoded_images)
                    else:
                        generated_frames_list.append(output_decoded_images[chunk_motion_frame_count:])

                    pre_motion_pixels = decoded_images
                    curr_frame_idx += chunk_frame_count

                    generated_frames_len = sum(t.shape[0] for t in generated_frames_list)
                    if generated_frames_len >= total_frames:
                        break

                final_video_frames = torch.cat(generated_frames_list, dim=0)
                final_video_frames = final_video_frames[:int(total_frames)]

                video_output = InputImpl.VideoFromComponents(
                    Types.VideoComponents(
                        images=final_video_frames,
                        audio=video.get_components().audio,
                        frame_rate=Fraction(fps)
                    )
                )
                output_videos.append(video_output)

        finally:
            if manager.release_after_run:
                manager.release_all()
            elif manager.offload_after_run:
                manager.offload_all()

        return io.NodeOutput(output_videos)

    @classmethod
    def character_mask_scale(cls, character_mask, latent_width, latent_height):
        if character_mask is None:
            return None
        if character_mask.ndim != 3:
            raise ValueError("[DALab] Character mask must be a 3D tensor")

        character_mask = character_mask.unsqueeze(0).unsqueeze(0)
        character_mask = comfy.utils.common_upscale(
            character_mask, latent_width, latent_height, "nearest-exact", "center"
        )
        character_mask = character_mask.movedim(2, 0)

        return character_mask

    @classmethod
    def clip_vision_encode(cls, clip_vision, reference_image):
        if reference_image["value"] is None:
            return None

        if reference_image["cache_key"] in _CACHE.clip_vision_output:
            clip_vision_output = _CACHE.clip_vision_output[reference_image["cache_key"]]
        else:
            clip_vision_output = clip_vision.encode_image(reference_image["value"])
            _CACHE.clip_vision_output[reference_image["cache_key"]] = clip_vision_output

        return clip_vision_output

    @classmethod
    def encode_ref_image(cls, vae, reference_image, latent_width, latent_height):
        if reference_image["value"] is None:
            ref_image = torch.zeros((1, latent_width * 8, latent_height * 8, 3))
        else:
            ref_image = reference_image["value"]
            ref_image = utils.scale_by_width_height(ref_image, latent_width * 8, latent_height * 8, "area", "center")
        
        ref_image_latent = vae.encode(ref_image[:, :, :, :3])
        ref_image_latent_mask = torch.zeros(
            (1, 4, 1, latent_height, latent_width)
        )

        return ref_image_latent, ref_image_latent_mask

    @classmethod
    def encode_text(cls, text_encoder, prompt, negative_prompt, cfg):
        if prompt["cache_key"] in _CACHE.positive:
            logger.info(f"WanAnimate positive prompt cache hit: {prompt['cache_key']}")
            positive = copy.copy(_CACHE.positive[prompt["cache_key"]])
        else:
            tokens = text_encoder.tokenize(prompt["value"])
            positive = text_encoder.encode_from_tokens_scheduled(tokens)
            _CACHE.positive[prompt["cache_key"]] = positive

        if cfg > 1.0:
            if prompt["cache_key"] in _CACHE.negative:
                negative = copy.copy(_CACHE.negative[prompt["cache_key"]])
            else:
                negative_tokens = text_encoder.tokenize(negative_prompt)
                negative = text_encoder.encode_from_tokens_scheduled(negative_tokens)
                _CACHE.negative[prompt["cache_key"]] = negative
        else:
            negative = []

        return positive, negative

    @classmethod
    def execute_florence(cls, video, detect_type, character_id, seed = 0):
        if detect_type == "default":
            output = DAFlorence2.execute(
                [video], 
                {'task': ['use_text'], 'prompts': ['body'], 'bbox_index': [character_id], 'task_options': ['bbox_by_text']}, 
                [seed]
            )
        else:
            output = DAFlorence2.execute(
                [video], 
                {'task': ['object'], 'bbox_index': [character_id], 'task_options': ['bbox_by_index']},
                [seed]
            )

        return output

    @classmethod
    def execute_sam(cls, video, bbox_data, seed):
        sam_output = DASAM2.execute(
            [video],
            [bbox_data],
            [seed]
        )
        return sam_output
    
    @classmethod
    def execute_dwpose(cls, video, enable_body, enable_hand, enable_face):
        dwpose_output = DADWPose.execute(
            [video],
            [enable_body],
            [enable_hand],
            [enable_face],
        )

        return dwpose_output

    @classmethod
    def get_videos(cls, video, detect_type, character_id, seed, config):
        video_frames = video.get_components().images
        video_fps = video.get_components().frame_rate

        florence_output = cls.execute_florence(video, detect_type, character_id, seed)
        bbox_info = json.loads(florence_output[1][0])
        if bbox_info.get("detection") is None:
            logger.warning(f"No detection found for character_id: {character_id}")
            return None, None, None, None, video_fps
        
        sam_output = cls.execute_sam(video, florence_output[1][0], seed)
        video_masks = sam_output[0][0]

        dwpose_pose_output = cls.execute_dwpose(video, True, True, False)
        pose_video = dwpose_pose_output[1][0].get_components().images
        face_keypoints = dwpose_pose_output[2][0]

        background_video, final_person_mask, face_pixel_video, isolated_pose_video = cls.process_data_for_wan(
            video_frames, video_masks, pose_video, face_keypoints, config
        )

        return isolated_pose_video,face_pixel_video,background_video,final_person_mask,video_fps

    @staticmethod
    def morph_to_lego_mask(mask_np, w_div=10, h_div=20):
        y_coords, x_coords = np.nonzero(mask_np)
        if len(y_coords) == 0: return mask_np
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        if bbox_w == 0 or bbox_h == 0: return mask_np
        
        w_step = max(1, int(bbox_w / w_div))
        h_step = max(1, int(bbox_h / h_div))
        
        aug_mask = mask_np.copy()
        
        for x in range(x_min, x_max, w_step):
            x_end = min(x + w_step, x_max + 1)
            for y in range(y_min, y_max, h_step):
                y_end = min(y + h_step, y_max + 1)
                
                if aug_mask[y:y_end, x:x_end].sum() > 0:
                    aug_mask[y:y_end, x:x_end] = 1
                    
        return aug_mask

    @classmethod
    def process_data_for_wan(cls, video_frames, video_masks, pose_video, face_keypoints_list, config):
        try:
            import cv2
        except ImportError:
            raise ImportError(f"[DALab] Node DAWanAnimate: Failed to import cv2. please pip install cv2 and restart ComfyUI.")

        if video_frames is None or len(video_frames) == 0:
            raise ValueError("[DALab] video_frames is empty")
        if video_masks is None:
            raise ValueError("[DALab] video_masks is None")
        if pose_video is None or len(pose_video) == 0:
            raise ValueError("[DALab] pose_video is empty")
        if face_keypoints_list is None or len(face_keypoints_list) == 0:
            raise ValueError("[DALab] face_keypoints_list is empty")

        frames_np = (video_frames.cpu().numpy() * 255).astype(np.uint8)
        T, H, W, C = frames_np.shape 

        sam_mask_grow_ratio = config.get("sam_mask_grow_ratio", 5)
        lego_mask_w_div = config.get("lego_mask_w_div", 10)
        lego_mask_h_div = config.get("lego_mask_h_div", 20)
        face_crop_scale = config.get("face_crop_scale", 2.0)
        face_crop_y_offset = config.get("face_crop_y_offset", 0.0)
        
        mask_tensor = video_masks.float().unsqueeze(0).unsqueeze(0)
        mask_resized = torch.nn.functional.interpolate(mask_tensor, size=(T, H, W), mode='nearest-exact')
        mask_resized = mask_resized.squeeze(0).squeeze(0)
        mask_np = mask_resized.cpu().numpy().astype(np.uint8) 

        background_frames = []
        lego_masks_list = [] 
        
        for i in tqdm(range(T), desc="Processing background frames"):
            frame = frames_np[i]
            sam_mask = mask_np[i]
            
            base_kernel = np.ones((sam_mask_grow_ratio, sam_mask_grow_ratio), np.uint8)
            mask_dilated = cv2.dilate(sam_mask, base_kernel, iterations=1)
            
            lego_mask = cls.morph_to_lego_mask(mask_dilated, w_div=lego_mask_w_div, h_div=lego_mask_h_div)
            lego_masks_list.append(lego_mask)
            
            inverse_mask = (lego_mask == 0).astype(np.uint8)
            inverse_mask_3c = np.stack([inverse_mask] * 3, axis=-1)
            
            blacked_out_frame = frame * inverse_mask_3c
            background_frames.append(blacked_out_frame)
            
        background_video = torch.from_numpy(np.array(background_frames)).float() / 255.0
        lego_masks_np = np.array(lego_masks_list) # [T, H, W]
        
        if pose_video.shape[1] != H or pose_video.shape[2] != W:
             pose_permuted = pose_video.permute(0, 3, 1, 2)
             pose_resized = torch.nn.functional.interpolate(pose_permuted, size=(H, W), mode='bilinear')
             pose_resized = pose_resized.permute(0, 2, 3, 1)
        else:
             pose_resized = pose_video
        pose_video_np = (pose_resized.cpu().numpy() * 255).astype(np.uint8)

        face_crops = []
        isolated_pose_frames = [] 
        
        kp_canvas_h = face_keypoints_list[0].get('canvas_height', H)
        kp_canvas_w = face_keypoints_list[0].get('canvas_width', W)
        scale_x = W / kp_canvas_w
        scale_y = H / kp_canvas_h

        for i in tqdm(range(T), desc="Processing face crops"):
            frame_data = face_keypoints_list[i] if i < len(face_keypoints_list) else {}
            people = frame_data.get('people') or []
            current_sam_mask = mask_np[i]
            
            best_person = None
            max_hit_score = 0.0 

            for person in people:
                pose_kps = person.get('pose_keypoints_2d') or []
                if len(pose_kps) > 0:
                    xs = pose_kps[0::3]
                    ys = pose_kps[1::3]
                    hit_count = 0
                    valid_count = 0
                    for x, y in zip(xs[:5], ys[:5]): 
                        if x > 0 and y > 0:
                            mx = min(max(0, int(x * scale_x)), W - 1)
                            my = min(max(0, int(y * scale_y)), H - 1)
                            valid_count += 1
                            if current_sam_mask[my, mx] > 0: 
                                hit_count += 1
                    
                    if valid_count > 0:
                        score = hit_count / valid_count
                        if score > max_hit_score and score > 0.4:
                            max_hit_score = score
                            best_person = person

            crop_box = None
            if best_person:
                face_kps = best_person.get('face_keypoints_2d') or []
                if len(face_kps) > 0:
                    xs = face_kps[0::3]
                    ys = face_kps[1::3]
                    valid_xs = [x for x in xs if x > 0]
                    valid_ys = [y for y in ys if y > 0]

                    if valid_xs and valid_ys:
                        min_x, max_x = min(valid_xs), max(valid_xs)
                        min_y, max_y = min(valid_ys), max(valid_ys)
                        cx = (min_x + max_x) / 2 * scale_x
                        cy = (min_y + max_y) / 2 * scale_y
                        box_size = max((max_x - min_x) * scale_x, (max_y - min_y) * scale_y) * face_crop_scale
                        half = box_size // 2
                        cy_offset = cy + box_size * face_crop_y_offset
                        crop_box = [int(cx - half), int(cy_offset - half), int(cx + half), int(cy_offset + half)]

            if crop_box:
                x1, y1, x2, y2 = crop_box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                crop = frames_np[i, y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frames_np[i]
                face_crops.append(cv2.resize(crop, (FACE_SCALE_WIDTH, FACE_SCALE_HEIGHT), interpolation=cv2.INTER_AREA))
            else:
                if len(face_crops) > 0:
                    face_crops.append(face_crops[-1])
                else:
                    face_crops.append(cv2.resize(frames_np[i], (FACE_SCALE_WIDTH, FACE_SCALE_HEIGHT), interpolation=cv2.INTER_AREA))

            current_lego_mask = lego_masks_np[i]
            current_pose = pose_video_np[i].astype(np.float32)

            if current_lego_mask.sum() == 0:
                isolated_frame = current_pose
            else:
                pose_mask_float = current_lego_mask.astype(np.float32)
                pose_mask_3c = np.stack([pose_mask_float] * 3, axis=-1)
                isolated_frame = current_pose * pose_mask_3c 
            
            isolated_pose_frames.append(isolated_frame)
            
        face_pixel_video = torch.from_numpy(np.array(face_crops)).float() / 255.0
        isolated_pose_video = torch.from_numpy(np.array(isolated_pose_frames)).float() / 255.0 
        final_person_mask = torch.from_numpy(lego_masks_np).float()
        
        return background_video, final_person_mask, face_pixel_video, isolated_pose_video
    
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            config_mtime = os.path.getmtime(_CONFIG_FILE_PATH)
            global_config_mtime = os.path.getmtime(get_config_file_path("global"))
        except:
            config_mtime = 0
            global_config_mtime = 0

        return hash((str(kwargs), str(config_mtime), str(global_config_mtime)))
