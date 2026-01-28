import os
import torch

import folder_paths
from comfy_api.latest import io,ui

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.wrappers.voxcpm15 import get_voxcpm15
from ..utils.wrappers.base import handle_wrapper_after_run
from ..utils.logger import logger
from ..utils.paths import get_config_file_path

_CONFIG_FILE_PATH = get_config_file_path("voxcpm15")

class DAVoxCPM15Config(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        model_path = os.path.join(folder_paths.models_dir, "voxcpm15")
        config = ConfigLoader(_CONFIG_FILE_PATH,strict=False)

        return io.Schema(
            node_id="DAVoxCPM15Config",
            display_name="DA VoxCPM1.5 Config",
            category="DALab/Audio/VoxCPM1.5",
            description="Configure the VoxCPM1.5 model,Run first to save the config",
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "model_path",
                    default=config.get("model_path", model_path),
                    display_name="model_path",
                    tooltip="The VoxCPM1.5 model. Default path: models/dalab/voxcpm15",
                ),
                io.Combo.Input(
                    "device",
                    default=config.get("device", "cuda"),
                    options=["cuda", "cpu"],
                    tooltip="The device to use for the model. Default: cuda",
                    display_name="device",
                ),
                io.Boolean.Input(
                    "enable_denoiser",
                    default=config.get("enable_denoiser", True),
                    tooltip="Whether to enable the denoiser. Default: True",
                    display_name="enable_denoiser",
                ),
                io.Boolean.Input(
                    "optimize",
                    default=False,
                    tooltip="Whether to torch compile the model. Default: False",
                    display_name="optimize",
                ),
                io.Boolean.Input(
                    "normalize",
                    default=False,
                    tooltip="Whether to normalize the prompt. Default: False",
                    display_name="normalize",
                ),
                io.Boolean.Input(
                    "denoise",
                    default=False,
                    tooltip="Whether to denoise the prompt. Default: False",
                    display_name="denoise",
                ),
                io.Float.Input(
                    "cfg_value",
                    default=config.get("cfg_value", 2.0),
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    tooltip="The cfg value to use for the model. Default: 2.0",
                    display_name="cfg_value",
                ),
                io.Int.Input(
                    "inference_timesteps",
                    default=config.get("inference_timesteps", 10),
                    min=1,
                    max=100,
                    step=1,
                    tooltip="The inference timesteps to use for the model. Default: 10",
                ),
                io.Int.Input(
                    "min_len",
                    default=config.get("min_len", 2),
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="The min length to use for the model. Default: 2",
                ),
                io.Int.Input(
                    "max_len",
                    default=config.get("max_len", 4096),
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="The max length to use for the model. Default: 4096",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls, 
        model_path: str,    
        device: str,
        enable_denoiser: bool,
        optimize: bool,
        normalize: bool,
        denoise: bool,
        cfg_value: float,
        inference_timesteps: int,
        min_len: int,
        max_len: int,
    ) -> io.NodeOutput:
        
        config_data = {
            "model_path": model_path,
            "device": device,
            "enable_denoiser": enable_denoiser,
            "optimize": optimize,
            "normalize": normalize,
            "denoise": denoise,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "min_len": min_len,
            "max_len": max_len,
        }

        utils.save_json(config_data, _CONFIG_FILE_PATH)

        return io.NodeOutput()

class DAVoxCPM15(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DAVoxCPM15",
            display_name="DA VoxCPM1.5",
            category="DALab/Audio/VoxCPM1.5",
            description="Generate audios using the VoxCPM1.5 model",
            is_input_list=True,
            inputs=[
                io.Audio.Input(
                    "ref_audios",
                    tooltip="The reference audio",
                    display_name="ref_audios",
                ),
                io.String.Input(
                    "prompts",
                    multiline=True,
                    default="床前明月光，疑是地上霜",
                    display_name="prompts",
                ),
                io.Int.Input(   
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="The random seed used for generation.",
                    display_name="seed",
                ),
            ],
            outputs=[
                io.Audio.Output(
                    "audios", 
                    is_output_list=True, 
                    tooltip="The generated audios", 
                    display_name="audios",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, 
        ref_audios: list[io.Audio],
        prompts: list[str],
        seed: list[int],
    ) -> io.NodeOutput:
        batch_inputs = utils.inputs_to_batch(
            defaults={
                "prompt": "",
            },
            prompt=prompts,
            ref_audio=ref_audios,
        )

        config = ConfigLoader(_CONFIG_FILE_PATH,strict=True)

        model_path = config.get("model_path")
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        device = config.get("device")
        enable_denoiser = config.get("enable_denoiser")
        optimize = config.get("optimize")
        normalize = config.get("normalize")
        denoise = config.get("denoise")
        cfg_value = config.get("cfg_value")
        inference_timesteps = config.get("inference_timesteps")
        min_len = config.get("min_len")
        max_len = config.get("max_len")

        model_wrapper = get_voxcpm15(model_path, device, enable_denoiser, optimize)
        model_wrapper.load_wrapper()
        model = model_wrapper.model

        output_audios = []
        for idx, input in enumerate(batch_inputs):
            logger.info(f"VoxCPM1.5 processing input {idx+1}/{len(batch_inputs)}")

            ref_audio = input["ref_audio"]["value"]
            prompt = input["prompt"]["value"]
            
            if prompt == "":
                logger.info(f"VoxCPM1.5 Prompt is empty, skipping : {idx+1}")
                continue

            if ref_audio is None:
                logger.info(f"VoxCPM1.5 Reference audio is empty, skipping : {idx+1}")
                continue

            temp_audio = ui.AudioSaveHelper.save_audio(
                ref_audio,
                filename_prefix="temp_audio",
                folder_type=io.FolderType.temp,
                cls=cls,
            )[0]
            temp_audio_path = os.path.join(
                folder_paths.get_temp_directory(),
                temp_audio["subfolder"], 
                temp_audio["filename"]
            )

            audio_text = model_wrapper.prompt_wav_recognition(temp_audio_path)

            result = model.generate(
                text=prompt,
                prompt_wav_path=temp_audio_path,
                prompt_text=audio_text,
                normalize=normalize,
                denoise=denoise,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                min_len=min_len,
                max_len=max_len,
            )
            audio = _convert_to_comfy_audio(result,model.tts_model.sample_rate)
            output_audios.append(audio)

        handle_wrapper_after_run("voxcpm15")

        return io.NodeOutput(output_audios)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        try:
            config_mtime = os.path.getmtime(_CONFIG_FILE_PATH)
            global_config_mtime = os.path.getmtime(get_config_file_path("global"))
        except:
            config_mtime = 0
            global_config_mtime = 0
            
        return hash((str(kwargs),str(config_mtime),str(global_config_mtime)))

def _convert_to_comfy_audio(wav, sample_rate):
    waveform = torch.from_numpy(wav).cpu()
    waveform = waveform.unsqueeze(0).repeat(2, 1).unsqueeze(0)
    
    return {
        "waveform": waveform,
        "sample_rate": int(sample_rate)
    }