import logging
import torch
import json
import os
import numpy as np
from PIL import Image
import math
import torch.nn.functional as F

import folder_paths
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.utils
import comfy.sd
from comfy_api.latest import io
from comfy_extras.nodes_easycache import EasyCacheNode

from .logger import logger
from .model_manager import ModelManager

def unload_model(model):
    current_loaded_models = comfy.model_management.current_loaded_models
    
    for i in range(len(current_loaded_models)):
        need_unload = False
        if isinstance(model, comfy.model_patcher.ModelPatcher) and current_loaded_models[i].model is model:
            need_unload = True
        elif hasattr(model, "patcher") and current_loaded_models[i].model is model.patcher:
            need_unload = True
        
        if need_unload:
            logging.info(f"[DALab] Manually unloading model: {model.__class__.__name__}")
            
            current_loaded_models[i].model_unload()
            current_loaded_models.pop(i)
            comfy.model_management.soft_empty_cache()
            
            return True
            
    logging.warning(f"[DALab] Model not found in currently loaded list, cannot unload: {model.__class__.__name__}")

    return False

def dynamic_combo_easycache(config):
    selected_easycache = config.get("easycache",{}).get("easycache", "false")

    easycache_options = [
        io.DynamicCombo.Option(
            key="false",
            inputs= []
        ),
        io.DynamicCombo.Option(
            key="true",
            inputs=[
                io.Float.Input(
                    "reuse_threshold",
                    default=config.get("easycache",{}).get("reuse_threshold", 0.2),
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="The threshold for reusing cached steps.",
                ),
                io.Float.Input(
                    "start_percent",
                    default=config.get("easycache",{}).get("start_percent", 0.1),
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="The relative sampling step to begin use of EasyCache.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=config.get("easycache",{}).get("end_percent", 0.95),
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="The relative sampling step to end use of EasyCache.",
                ),
            ]
        ),
    ]
    
    easycache_options.sort(key=lambda x: x.key == selected_easycache, reverse=True)
    
    return easycache_options

def patch_model_easycache(model, config):
    if "reuse_threshold" in config.get("easycache"):
        logger.info("EasyCache is used")
        model = EasyCacheNode.execute(
            model=model,
            reuse_threshold=config.get("easycache").get("reuse_threshold", 0.2),
            start_percent=config.get("easycache").get("start_percent", 0.1),
            end_percent=config.get("easycache").get("end_percent", 0.95),
            verbose=False
        ).result[0]
    else:
        logger.info("EasyCache is not used")

    return model

def dynamic_combo_feishu_options(config):
    text_options = config.get("text_options",{}).get("text_options", "[0]")
    text_combos = []
    for i in range(6):
        inputs = []
        for j in range(i):
            inputs.append(io.String.Input(
                f"text{j+1}",
                default=config.get("text_options",{}).get(f"text{j+1}", f"text{j+1}"),
                display_name=f"Text {j+1}",
                tooltip=f"The text {j+1} for prompts.",
            ))
        text_combos.append(io.DynamicCombo.Option(key=f"[{i}]", inputs=inputs))
    text_combos.sort(key=lambda x: x.key == text_options, reverse=True)

    image_options = config.get("image_options",{}).get("image_options", "[0]")
    image_combos = []
    for i in range(6):
        inputs = []
        for j in range(i):
            inputs.append(io.String.Input(
                f"image{j+1}",
                default=config.get("image_options",{}).get(f"image{j+1}", f"image{j+1}"),
                display_name=f"Image {j+1}",
                tooltip=f"The image {j+1} for reference.",
            ))
        image_combos.append(io.DynamicCombo.Option(key=f"[{i}]", inputs=inputs))
    image_combos.sort(key=lambda x: x.key == image_options, reverse=True)

    audio_options = config.get("audio_options",{}).get("audio_options", "[0]")
    audio_combos = []
    for i in range(6):
        inputs = []
        for j in range(i):
            inputs.append(io.String.Input(
                f"audio{j+1}",
                default=config.get("audio_options",{}).get(f"audio{j+1}", f"audio{j+1}"),
                display_name=f"Audio {j+1}",
                tooltip=f"The audio {j+1} for reference.",
            ))
        audio_combos.append(io.DynamicCombo.Option(key=f"[{i}]", inputs=inputs))
    audio_combos.sort(key=lambda x: x.key == audio_options, reverse=True)

    video_options = config.get("video_options",{}).get("video_options", "[0]")
    video_combos = []
    for i in range(6):
        inputs = []
        for j in range(i):
            inputs.append(io.String.Input(
                f"video{j+1}",
                default=config.get("video_options",{}).get(f"video{j+1}", f"video{j+1}"),
                display_name=f"Video {j+1}",
                tooltip=f"The video {j+1} for reference.",
            ))
        video_combos.append(io.DynamicCombo.Option(key=f"[{i}]", inputs=inputs))
    video_combos.sort(key=lambda x: x.key == video_options, reverse=True)

    return text_combos, image_combos, audio_combos, video_combos

def get_feishu_outputs_from_config(config):
    text_options = config.get("text_options",{}).get("text_options", "[0]")
    image_options = config.get("image_options",{}).get("image_options", "[0]")
    audio_options = config.get("audio_options",{}).get("audio_options", "[0]")
    video_options = config.get("video_options",{}).get("video_options", "[0]")

    texts = []
    for i in range(int(text_options.replace("[", "").replace("]", ""))):
        texts.append(
            io.String.Output(
                f"text{i+1}",
                display_name=config.get("text_options",{}).get(f"text{i+1}",f"text{i+1}"),
                tooltip=f"The text {i+1} for prompts.",
                is_output_list=True,
            )
        )

    images = []
    for i in range(int(image_options.replace("[", "").replace("]", ""))):
        images.append(
            io.Image.Output(
                f"image{i+1}",
                display_name=config.get("image_options",{}).get(f"image{i+1}", f"image{i+1}"),
                tooltip=f"The image {i+1} for reference.",
                is_output_list=True,
            )
        )

    audios = []
    for i in range(int(audio_options.replace("[", "").replace("]", ""))):
        audios.append(
            io.Audio.Output(
                f"audio{i+1}",
                display_name=config.get("audio_options",{}).get(f"audio{i+1}", f"audio{i+1}"),
                tooltip=f"The audio {i+1} for reference.",
                is_output_list=True,
            )
        )

    videos = []
    for i in range(int(video_options.replace("[", "").replace("]", ""))):
        videos.append(
            io.Video.Output(
                f"video{i+1}",
                display_name=config.get("video_options",{}).get(f"video{i+1}", f"video{i+1}"),
                tooltip=f"The video {i+1} for reference.",
                is_output_list=True,
            )
        )

    return texts, images, audios, videos

def dynamic_combo_loras(config,lora_options,key_name="loras"):
    selected_loras = config.get(key_name,{}).get(key_name, "[0]")

    loras = []
    for i in range(6):
        inputs = []
        for j in range(i):
            inputs.append(io.Combo.Input(
                f"lora_name{j+1}",
                default=config.get(key_name,{}).get(f"lora_name{j+1}", ""),
                options=lora_options,
                display_name=f"LoRA {j+1}",
                tooltip=f"The LoRA {j+1} for generation. Default: models/loras",
            ))
            inputs.append(io.Float.Input(
                f"lora_strength{j+1}",
                default=config.get(key_name,{}).get(f"lora_strength{j+1}", 1.0),
                min=0.0,
                max=2.0,
                step=0.01,
                tooltip=f"The strength of the LoRA {j+1} for generation. Default: 1.0",
            ))
        loras.append(io.DynamicCombo.Option(key=f"[{i}]", inputs=inputs))
        
    loras.sort(key=lambda x: x.key == selected_loras, reverse=True)

    return loras

def patch_model_with_loras(model, config, key_name="loras"):
    loras_config = config.get(key_name,{})
    lora_number = loras_config.get(key_name, "[0]")
    manager = ModelManager()
    
    try:
        log_logger = logging.getLogger()
        log_logger.setLevel(logging.ERROR)

        for i in range(int(lora_number.replace("[", "").replace("]", ""))):
            path = folder_paths.get_full_path_or_raise(
                "loras", loras_config.get(f"lora_name{i+1}")
            )
            strength = loras_config.get(f"lora_strength{i+1}", 1.0)

            lora_dict = manager.get_lora_dict(path)
            model, _ = comfy.sd.load_lora_for_models(
                model, None, lora_dict, strength, 0
            )
    finally:
        log_logger.setLevel(logging.INFO)

    return model

def patch_model_sampling(model, shift, multiplier=1000):
    model_clone = model.clone()
    sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
    sampling_type = comfy.model_sampling.CONST

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    model_sampling = ModelSamplingAdvanced(model_clone.model.model_config)
    model_sampling.set_parameters(shift=shift, multiplier=multiplier)
    model_clone.add_object_patch("model_sampling", model_sampling)
    return model_clone

def patch_cfg_norm(model, strength=1.0):
    model_clone = model.clone()
    def cfg_norm(args):
        cond_p = args['cond_denoised']
        pred_text_ = args["denoised"]

        norm_full_cond = torch.norm(cond_p, dim=1, keepdim=True)
        norm_pred_text = torch.norm(pred_text_, dim=1, keepdim=True)
        scale = (norm_full_cond / (norm_pred_text + 1e-8)).clamp(min=0.0, max=1.0)
        return pred_text_ * scale * strength

    model_clone.set_model_sampler_post_cfg_function(cfg_norm)
    return model_clone

def save_json(config_data, config_file_path):
    try:
        os.makedirs(os.path.dirname(config_file_path), exist_ok=True)

        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
        msg = f"[DALab] Config saved successfully to {config_file_path}"
        logging.info(msg)
    except Exception as e:
        msg = f"[DALab] Failed to save config: {str(e)}"
        raise Exception(msg)

def save_image_to_file(images, filename_prefix="uncleda_tools"):
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, 
        folder_paths.get_output_directory(), 
        images[0].width, 
        images[0].height
    )

    results = list()
    for (batch_number, image) in enumerate(images):
        metadata = None
        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
        file = f"{filename_with_batch_num}_{counter:05}_.png"
        image.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": "output"
        })
        counter += 1

    return results

def save_image_tensor_to_file(images, filename_prefix="uncleda_tools", output_dir=None, ext=".png"):
    if output_dir is None:
        output_dir = folder_paths.get_output_directory()

    if len(images) == 0:
        return []
    
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, 
        output_dir, 
        images[0].shape[2], 
        images[0].shape[1]
    )

    results = list()
    for (batch_number, image) in enumerate(images):
        image = image.squeeze(0)
        
        img_array = 255. * image.cpu().numpy()
        img_array = img_array.clip(0, 255).astype(np.uint8)

        img = Image.fromarray(img_array)

        if ext == ".jpg" and img.mode != "RGB":
            img = img.convert("RGB")

        metadata = None
        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))

        if ext == ".jpg":
            file = f"{filename_with_batch_num}_{counter:05}_.jpg"
            img.save(os.path.join(full_output_folder, file), quality=95, optimize=True)
        elif ext == ".webp":
            file = f"{filename_with_batch_num}_{counter:05}_.webp"
            img.save(os.path.join(full_output_folder, file), quality=95, lossless=False)
        else:
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
        
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": "output"
        })
        
        counter += 1

    return results

def scale_by_total_pixels(samples, total_pixels, upscale_method="area", crop="disabled"):
    samples = samples.movedim(-1, 1)
    scale_by = math.sqrt(total_pixels / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    scaled_samples = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
    scaled_samples = scaled_samples.movedim(1, -1)
    return scaled_samples[:, :, :, :3]

def scale_by_width_height(samples, width, height, upscale_method="area", crop="disabled"):
    if samples.shape[2] != height or samples.shape[3] != width:
        samples = samples.movedim(-1, 1)
        scaled_samples = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
        scaled_samples = scaled_samples.movedim(1, -1)
        return scaled_samples[:, :, :, :3]
    else:
        return samples

def inputs_to_batch(nested_inputs: dict = None, defaults: dict = None, **flat_inputs):
    defaults = defaults or {}
    nested_inputs = nested_inputs or {}
    
    cleaned_flat = {k: v or [] for k, v in flat_inputs.items()}
    cleaned_nested = {}
    for key, val in nested_inputs.items():
        if not val or isinstance(val, list):
            cleaned_nested[key] = {}
        else:
            cleaned_nested[key] = {k: v or [] for k, v in val.items()}

    all_lengths = []
    all_lengths.extend(len(v) for v in cleaned_flat.values())
    for group in cleaned_nested.values():
        all_lengths.extend(len(v) for v in group.values())
        
    count_loop = max(all_lengths, default=0)

    def get_item(items, key_prefix, idx):
        length = len(items)
        default_val = defaults.get(key_prefix, None)

        if idx < length:
            return {"value": items[idx], "cache_key": f"{key_prefix}_{idx}"}
        if length == 1:
            return {"value": items[0], "cache_key": f"{key_prefix}_0"}
        return {"value": default_val, "cache_key": f"{key_prefix}_{idx}"}

    batch_results = []
    for i in range(count_loop):
        batch_item = {}
        for k, v in cleaned_flat.items():
            batch_item[k] = get_item(v, k, i)
        
        for main_key, group_data in cleaned_nested.items():
            batch_item[main_key] = {
                sub_key: get_item(sub_list, f"{main_key}_{sub_key}", i)
                for sub_key, sub_list in group_data.items()
            }
        
        batch_results.append(batch_item)

    return batch_results

def resample_video_tensor(video_tensor, src_fps, target_fps):
    if src_fps == target_fps:
        return video_tensor

    f, h, w, c = video_tensor.shape
    
    target_frames = int(f * (target_fps / src_fps))
    
    tensor_permuted = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)
    
    output = F.interpolate(
        tensor_permuted, 
        size=(target_frames, h, w), 
        mode='nearest-exact' # or 'trilinear'
    )
    
    output = output.squeeze(0).permute(1, 2, 3, 0)
    
    return output

def match_and_blend_colors(source_frames: torch.Tensor, reference_frame: torch.Tensor, strength: float) -> torch.Tensor:
    try:
        from skimage import color
    except ImportError:
        logging.warning("[DALab] skimage is not installed. Color matching will not be available.")
        return source_frames

    if strength <= 0.0:
        return source_frames

    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")

    device = source_frames.device
    dtype = source_frames.dtype
    
    src_np = source_frames.cpu().numpy()
    ref_np = reference_frame.cpu().numpy()

    if ref_np.ndim == 4:
        ref_np = ref_np[0]

    try:
        src_lab = color.rgb2lab(src_np)
        ref_lab = color.rgb2lab(ref_np)
    except ValueError as e:
        logging.warning(f"[DALab] Color conversion failed: {e}. Skipping correction.")
        return source_frames

    ref_mean = np.mean(ref_lab, axis=(0, 1))
    ref_std = np.std(ref_lab, axis=(0, 1))

    src_mean = np.mean(src_lab, axis=(1, 2), keepdims=True)
    src_std = np.std(src_lab, axis=(1, 2), keepdims=True)

    src_std = np.maximum(src_std, 1e-6)
    
    corrected_lab = (src_lab - src_mean) * (ref_std / src_std) + ref_mean

    try:
        corrected_rgb = color.lab2rgb(corrected_lab)
    except ValueError as e:
        logging.warning(f"[DALab] LAB to RGB conversion failed: {e}. Skipping correction.")
        return source_frames

    corrected_rgb = np.clip(corrected_rgb, 0.0, 1.0)
    final_np = (1.0 - strength) * src_np + strength * corrected_rgb
    
    output_tensor = torch.from_numpy(final_np).to(device=device, dtype=dtype)
    
    return output_tensor

def change_audio_volume(audio, volume):
    if volume == 0:
        return audio
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]

    if volume < 0:
        target_db = volume * 6
    else:
        target_db = volume * 1.2

    gain = 10 ** (target_db / 20)
    waveform = waveform * gain
    waveform = torch.clamp(waveform, min=-1.0, max=1.0)

    return {"waveform": waveform, "sample_rate": sample_rate}

def parse_glossary_text(text: str):
    if not text:
        return {}
    
    final_dict = {}

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        parts = [p.strip() for p in line.replace('｜', '|').split('|')]
        parts = [p for p in parts if p]
        
        if len(parts) == 2:
            term, replacement = parts
            final_dict[term] = replacement
            
        elif len(parts) == 3:
            term, lang_code, replacement = parts
            
            target_lang = None
            if lang_code.lower() in ['cn', 'zh', '中文']:
                target_lang = 'zh'
            elif lang_code.lower() in ['en', 'us', '英文']:
                target_lang = 'en'
            
            if target_lang:
                if term not in final_dict or not isinstance(final_dict[term], dict):
                    final_dict[term] = {}
                
                final_dict[term][target_lang] = replacement

    return final_dict

def pingpong_video_padding(array, target_len):
    if len(array) == 1:
        return [array[0]] * target_len

    idx = 0
    flip = False
    target_array = []
    while len(target_array) < target_len:
        target_array.append(array[idx])
        if flip:
            idx -= 1
        else:
            idx += 1
        if idx == 0 or idx == len(array) - 1:
            flip = not flip
    return target_array[:target_len]

def get_valid_len(real_len, clip_len=77, overlap=1):
    import math
    if real_len <= clip_len:
        return clip_len

    n_rounds = math.ceil((real_len + overlap) / clip_len)
    target_len = n_rounds * clip_len - overlap
    return target_len