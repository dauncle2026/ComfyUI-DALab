import os
import torch

import folder_paths
from comfy_api.latest import io,ui

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.wrappers.cosy_voice3 import get_cosy_voice3
from ..utils.wrappers.base import handle_wrapper_after_run
from ..utils.logger import logger
from ..utils.paths import get_config_file_path

_CONFIG_FILE_PATH = get_config_file_path("cosy_voice3")

class DACosyVoice3Config(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        model_path = os.path.join(folder_paths.models_dir, "cosy_voice3")
        config = ConfigLoader(_CONFIG_FILE_PATH,strict=False)

        return io.Schema(
            node_id="DACosyVoice3Config",
            display_name="DA Cosy Voice3 Config",
            category="DALab/Audio/Cosy Voice3",
            description="Configure the Cosy Voice3 model,Run first to save the config",
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "model_path",
                    default=config.get("model_path", model_path),
                    display_name="model_path",
                    tooltip="The Cosy Voice3 model. Default path: models/cosy_voice3",
                ),
                io.Combo.Input(
                    "device",
                    default=config.get("device", "cuda"),
                    options=["cuda", "cpu"],
                    tooltip="The device to use for the model. Default: cuda",
                    display_name="device",
                ),
                io.Boolean.Input(
                    "use_fp16",
                    default=config.get("use_fp16", True),
                    tooltip="True: Use FP16. False: Use FP32. Default: True",
                    display_name="use_fp16",
                ),
                io.Float.Input(
                    "speed",
                    default=config.get("speed", 1.0),
                    tooltip="The speed of the audio.",
                    display_name="speed",
                    min=0.5,
                    max=2.0,
                    step=0.1,
                ),
                io.String.Input(
                    "rich_tags",
                    default=config.get("rich_tags", "[breath], [laughter], [cough], [clucking], [accent], [quick_breath], [sigh], [lipsmack], [mn], <strong></strong>, <laughter></laughter>"),
                    tooltip="The rich tags to use for the audio.Default: [breath], [laughter], [cough], [clucking], [accent], [quick_breath], [sigh], [lipsmack], [mn], <strong></strong>, <laughter></laughter>",
                    display_name="rich_tags",
                    multiline=True,
                )
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls, 
        model_path: str, 
        device: str,
        use_fp16: bool,
        speed: float,
        rich_tags: str,
    ) -> io.NodeOutput:
        config_data = {
            "model_path": model_path,
            "device": device,
            "use_fp16": use_fp16,
            "speed": speed,
            "rich_tags": rich_tags,
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)

        return io.NodeOutput()

class DACosyVoice3(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        mode_options = [
            io.DynamicCombo.Option("auto", []),
            io.DynamicCombo.Option("rich_tags", []),
            io.DynamicCombo.Option("instruct", [
                io.String.Input(
                    "instruct_prompt",
                    tooltip="The instruct prompt.",
                    multiline=True,
                    default="",
                ),
            ]),
            io.DynamicCombo.Option("manual", [
                io.String.Input(
                    "custom_text",
                    tooltip="The custom text.",
                    multiline=True,
                    default="",
                ),
            ]),
        ]

        return io.Schema(
            node_id="DACosyVoice3",
            display_name="DA Cosy Voice3",
            category="DALab/Audio/Cosy Voice3",
            description="Generate audios using the Cosy Voice3 model",
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
                io.DynamicCombo.Input(
                    "modes",
                    options=mode_options,
                    tooltip="The mode used for the audio.",
                    display_name="modes",
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
        modes: list[str],
        seed: list[int],
    ) -> io.NodeOutput:
        batch_inputs = utils.inputs_to_batch(
            defaults={
                "prompt": "",
            },
            prompt=prompts,
            ref_audio=ref_audios,
            nested_inputs={
                "mode": modes,
            }
        )

        config = ConfigLoader(_CONFIG_FILE_PATH,strict=True)

        model_path = config.get("model_path")
        config_path = os.path.join(model_path, "cosyvoice3.yaml")
        
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        device = config.get("device")
        use_fp16 = config.get("use_fp16")
        speed = config.get("speed")

        model_wrapper = get_cosy_voice3(model_path, device, use_fp16)
        model_wrapper.load_wrapper()
        model = model_wrapper.model

        output_audios = []
        for idx, input in enumerate(batch_inputs):
            logger.info(f"Cosy Voice3 processing input {idx+1}/{len(batch_inputs)}")

            ref_audio = input["ref_audio"]["value"]
            prompt = input["prompt"]["value"]
            mode_type = input["mode"]["modes"]["value"]
            
            if prompt == "":
                logger.info(f"Cosy Voice3 Prompt is empty, skipping : {idx+1}")
                continue

            if ref_audio is None:
                logger.info(f"Cosy Voice3 Reference audio is empty, skipping : {idx+1}")
                continue

            temp_audio = ui.AudioSaveHelper.save_audio(
                ref_audio,
                filename_prefix="temp_audio",
                folder_type=io.FolderType.temp,
                cls=cls,
            )[0]
            ref_audio_path = os.path.join(
                folder_paths.get_temp_directory(),
                temp_audio["subfolder"], 
                temp_audio["filename"]
            )

            if mode_type == "auto":
                ref_text = model_wrapper.prompt_wav_recognition(ref_audio_path)
                output = model.inference_zero_shot(
                    tts_text=prompt,
                    prompt_text="You are a helpful assistant.<|endofprompt|>" + ref_text,
                    prompt_wav=ref_audio_path,
                    speed=speed,
                )
            elif mode_type == "manual":
                output = model.inference_zero_shot(
                    tts_text=prompt,
                    prompt_text="You are a helpful assistant.<|endofprompt|>" + input["mode"]["custom_text"]["value"],
                    prompt_wav=ref_audio_path,
                    speed=speed,
                )
            elif mode_type == "rich_tags":
                output = model.inference_cross_lingual(
                    tts_text="You are a helpful assistant.<|endofprompt|>" + prompt,
                    prompt_wav=ref_audio_path,
                    speed=speed,
                )
            elif mode_type == "instruct":
                output = model.inference_instruct2(
                    tts_text=prompt,
                    instruct_text="You are a helpful assistant." + input["mode"]["instruct_prompt"]["value"] + "<|endofprompt|>",
                    prompt_wav=ref_audio_path,
                    speed=speed,
                )
            else:
                raise ValueError(f"[DALab] Cosy Voice3 Invalid mode type: {mode_type}")
            
            audio = _convert_to_comfy_audio(output,model.sample_rate)
            output_audios.append(audio)

        handle_wrapper_after_run("cosy_voice3")

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

def _convert_to_comfy_audio(result, sample_rate):
    all_speech = [chunk['tts_speech'] for chunk in result]
    waveform = torch.cat(all_speech, dim=-1).repeat(2, 1).unsqueeze(0).cpu()

    return {
        "waveform": waveform,
        "sample_rate": sample_rate
    }