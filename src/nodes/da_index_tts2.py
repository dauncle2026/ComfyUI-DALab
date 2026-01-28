import os
import torch
import numpy as np

import folder_paths
from comfy_api.latest import io,ui

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.wrappers.index_tts2 import get_index_tts2
from ..utils.wrappers.base import handle_wrapper_after_run, log_memory
from ..utils.logger import logger
from ..utils.paths import get_config_file_path

_CONFIG_FILE_PATH = get_config_file_path("index_tts2")

class DAIndexTTS2Config(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        model_path = os.path.join(folder_paths.models_dir, "index-tts")
        config = ConfigLoader(_CONFIG_FILE_PATH,strict=False)

        return io.Schema(
            node_id="DAIndexTTS2Config",
            display_name="DA IndexTTS2 Config",
            category="DALab/Audio/IndexTTS2",
            description="Configure the IndexTTS2 model,Run first to save the config",
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "model_path",
                    default=config.get("model_path", model_path),
                    display_name="model_path",
                    tooltip="The IndexTTS2 model. Default path: models/dalab/index_tts2",
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
                io.Boolean.Input(
                    "verbose",
                    default=False,
                    tooltip="Whether to print the verbose output. Default: False",
                    display_name="verbose",
                ),
                io.Int.Input(
                    "interval_silence",
                    default=200,
                    min=0,
                    max=1000,
                    tooltip="The duration of silence between audio segments in milliseconds (ms). Default: 200ms (0.2s).",
                    display_name="interval_silence",
                ),
                io.Float.Input(
                    "speed",
                    default=1.0,
                    min=0.8,
                    max=1.5,
                    tooltip="The speed of the audio. Default: 1.0",
                    display_name="speed",
                ),
                io.String.Input(
                    "glossary_text",
                    default="M.2|cn|M二\nM.2|en|M dot two\nChatGPT|en|G-P-T\n",
                    tooltip="only use '|' to split. one line one item. Example: ChatGPT|en|G-P-T",
                    display_name="glossary_text",
                    multiline=True,
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls, 
        model_path: str, 
        device: str,
        use_fp16: bool,
        verbose: bool,
        interval_silence: int,
        speed: float,
        glossary_text: str,
    ) -> io.NodeOutput:
        config_data = {
            "model_path": model_path,
            "device": device,
            "use_fp16": use_fp16,
            "verbose": verbose,
            "interval_silence": interval_silence,
            "speed": speed,
            "glossary_text": glossary_text,
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()

class DAIndexTTS2(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        emotion_options = [
            io.DynamicCombo.Option("no_emotion", []),
            io.DynamicCombo.Option("emotion_text", [
                io.String.Input(
                    "emotion_text",
                    tooltip="The emotion text.",
                    multiline=True,
                    default="极度高兴",
                    display_name="emotion_text",
                ),
                io.Float.Input(
                    "emo_alphas",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    tooltip="The alpha used for the emotion. Default: 0.7",
                    display_mode=io.NumberDisplay.slider,
                    display_name="emo_alphas",
                ),
            ]),
            io.DynamicCombo.Option("emotion_audio", [
                io.Audio.Input(
                    "emotion_audio",
                    tooltip="The emotion audio.",
                    display_name="emotion_audio",
                ),
                io.Float.Input(
                    "emo_alphas",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    tooltip="The alpha used for the emotion. Default: 0.7",
                    display_mode=io.NumberDisplay.slider,
                    display_name="emo_alphas",
                ),
            ]),
            io.DynamicCombo.Option("emotion_vector", [
                io.Float.Input(
                    "happy",
                    tooltip="The emotion vector.",
                    default=0.0,
                    min=0.0,
                    max=1.2,
                    display_mode=io.NumberDisplay.slider,
                    display_name="happy|高兴",
                ),
                io.Float.Input(
                    "angry",
                    tooltip="The emotion vector.",
                    default=0.0,
                    min=0.0,
                    max=1.2,
                    display_mode=io.NumberDisplay.slider,
                    display_name="angry|愤怒",
                ),
                io.Float.Input(
                    "sad",
                    tooltip="The emotion vector.",
                    default=0.0,
                    min=0.0,
                    max=1.2,
                    display_mode=io.NumberDisplay.slider,
                    display_name="sad|悲伤",
                ),
                io.Float.Input(
                    "afraid",
                    tooltip="The emotion vector.",
                    default=0.0,
                    min=0.0,
                    max=1.2,
                    display_mode=io.NumberDisplay.slider,
                    display_name="afraid|恐惧",
                ),
                io.Float.Input(
                    "disgusted",
                    tooltip="The emotion vector.",
                    default=0.0,
                    min=0.0,
                    max=1.2,
                    display_mode=io.NumberDisplay.slider,
                    display_name="disgusted|反感",
                ),
                io.Float.Input(
                    "melancholic",
                    tooltip="The emotion vector.",
                    default=0.0,
                    min=0.0,
                    max=1.2,
                    display_mode=io.NumberDisplay.slider,
                    display_name="melancholic|低落",
                ),
                io.Float.Input(
                    "surprised",
                    tooltip="The emotion vector.",
                    default=0.0,
                    min=0.0,
                    max=1.2,
                    display_mode=io.NumberDisplay.slider,
                    display_name="surprised|惊讶",
                ),
                io.Float.Input(
                    "calm",
                    tooltip="The emotion vector.",
                    default=0.0,
                    min=0.0,
                    max=1.2,
                    display_mode=io.NumberDisplay.slider,
                    display_name="calm|自然",
                ),
                io.Float.Input(
                    "emo_alphas",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    tooltip="The alpha used for the emotion. Default: 0.7",
                    display_mode=io.NumberDisplay.slider,
                    display_name="emo_alphas",
                ),
            ]),
        ]

        return io.Schema(
            node_id="DAIndexTTS2",
            display_name="DA IndexTTS2",
            category="DALab/Audio/IndexTTS2",
            description="Generate audios using the IndexTTS2 model",
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
                    "emotions",
                    options=emotion_options,
                    tooltip="The emotion used for the audio.",
                    display_name="emotions",
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
        emotions: list[str],
        seed: list[int],
    ) -> io.NodeOutput:
        batch_inputs = utils.inputs_to_batch(
            defaults={
                "prompt": "",
            },
            prompt=prompts,
            ref_audio=ref_audios,
            nested_inputs={
                "emotion": emotions,
            },
        )

        config = ConfigLoader(_CONFIG_FILE_PATH,strict=True)

        model_path = config.get("model_path")
        config_path = os.path.join(model_path, "config.yaml")

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        device = config.get("device")
        use_fp16 = config.get("use_fp16")
        verbose = config.get("verbose")
        interval_silence = config.get("interval_silence")
        speed = config.get("speed")
        glossary_text = config.get("glossary_text")

        model_wrapper = get_index_tts2(config_path, model_path, device, use_fp16)
        model_wrapper.load_wrapper()
        model = model_wrapper.model
        model.normalizer.load_glossary(utils.parse_glossary_text(glossary_text))

        output_audios = []
        for idx, input in enumerate(batch_inputs):
            logger.info(f"IndexTTS2 processing input {idx+1}/{len(batch_inputs)}")

            ref_audio = input["ref_audio"]["value"]
            prompt = input["prompt"]["value"]
            emotion_type = input["emotion"]["emotions"]["value"]
            
            if prompt == "":
                logger.info(f"IndexTTS2 Prompt is empty, skipping : {idx+1}")
                continue

            if ref_audio is None:
                logger.info(f"IndexTTS2 Reference audio is empty, skipping : {idx+1}")
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

            emotion_prompt = prompt
            emotion_alpha = 1.0
            emotion_audio_path = None
            emotion_vector = None
            use_emo_text=True

            if emotion_type == "emotion_text":
                emotion_prompt = input["emotion"]["emotion_text"]["value"]
                emotion_alpha = input["emotion"]["emo_alphas"]["value"]
            elif emotion_type == "emotion_audio":
                ref_emotion_audio = input["emotion"]["emotion_audio"]["value"]
                temp_emotion_audio = ui.AudioSaveHelper.save_audio(
                    ref_emotion_audio,
                    filename_prefix="temp_audio",
                    folder_type=io.FolderType.temp,
                    cls=cls,
                )[0]
                emotion_audio_path = os.path.join(
                    folder_paths.get_temp_directory(),
                    temp_emotion_audio["subfolder"], 
                    temp_emotion_audio["filename"]
                )
                emotion_alpha = input["emotion"]["emo_alphas"]["value"]
                use_emo_text = False
            elif emotion_type == "emotion_vector":
                emotion_vector = [
                    input["emotion"]["happy"]["value"],
                    input["emotion"]["angry"]["value"],
                    input["emotion"]["sad"]["value"],
                    input["emotion"]["afraid"]["value"],
                    input["emotion"]["disgusted"]["value"],
                    input["emotion"]["melancholic"]["value"],
                    input["emotion"]["surprised"]["value"],
                    input["emotion"]["calm"]["value"],
                ]
                emotion_alpha = input["emotion"]["emo_alphas"]["value"]
                use_emo_text = False

            result = model.infer(
                spk_audio_prompt=ref_audio_path,
                text=prompt,
                output_path=None,
                emo_audio_prompt=emotion_audio_path,
                emo_alpha=emotion_alpha,
                emo_vector=emotion_vector,
                use_emo_text=use_emo_text,
                emo_text=emotion_prompt,
                verbose=verbose,
                interval_silence=interval_silence,
                speed=speed
            )

            audio = _convert_to_comfy_audio(result)
            output_audios.append(audio)

        handle_wrapper_after_run("index_tts2")

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

def _convert_to_comfy_audio(result):
    sample_rate, wav = result
    wav = wav.astype(np.float32) / 32767.0
    
    waveform = torch.from_numpy(wav)
    waveform = waveform.transpose(0, 1).repeat(2, 1).unsqueeze(0)
    
    return {
        "waveform": waveform,
        "sample_rate": int(sample_rate)
    }