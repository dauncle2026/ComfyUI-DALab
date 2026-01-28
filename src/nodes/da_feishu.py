'''
@author: 
@date: 2026-01-25
@description: This node is used to configure the feishu table params.

required:
- lark-oapi
'''
import random

from comfy_api.latest import io

from ..utils import utils
from ..utils.feishu_manager import FeishuManager
from ..utils.config_loader import ConfigLoader
from ..utils.paths import get_config_file_path
from ..utils.logger import logger

_CONFIG_FILE_PATH = get_config_file_path("feishu")

class DAFeishuConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)

        text_combos, image_combos, audio_combos, video_combos = utils.dynamic_combo_feishu_options(config)

        return io.Schema(
            node_id="DAFeishuConfig",
            display_name="DA Feishu Config",
            category="DALab/Tools/Feishu",
            description="Configure the feishu table params. Run first to save config.",
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "app_id",
                    default=config.get("app_id", ""),
                    display_name="app_id",
                    tooltip="The app id for Feishu.",
                ),
                io.String.Input(
                    "app_secret",
                    default=config.get("app_secret", ""),
                    display_name="app_secret",
                    tooltip="The app secret for Feishu.",
                ),
                io.String.Input(
                    "view_server_address",
                    default=config.get("view_server_address", "http://127.0.0.1:8188"),
                    display_name="view_server_address",
                    tooltip="The server address for Feishu.",
                ),
                io.String.Input(
                    "open_field_name",
                    default=config.get("open_field_name", "open"),
                    display_name="open_field_name",
                    tooltip="The open field alias name for Feishu.",
                ),
                io.String.Input(
                    "open_field_value",
                    default=config.get("open_field_value", "on"),
                    display_name="open_field_value",
                    tooltip="The open field when enable is true for Feishu.",
                ),
                io.String.Input(
                    "frame_count_field_name",
                    default=config.get("frame_count_field_name", "frame_count"),
                    display_name="frame_count_field_name",
                    tooltip="The frame count field alias name for Feishu.",
                ),
                io.Combo.Input(
                    "image_local_save_ext",
                    default=config.get("image_local_save_ext", ".png"),
                    options=[".png", ".jpg", ".webp"],
                    display_name="image_local_save_ext",
                    tooltip="The image local save extension for Feishu.",
                ),
                io.DynamicCombo.Input(
                    "text_options",
                    options=text_combos,
                    display_name="text_options",
                    tooltip="The text options for prompts.",
                ),
                io.DynamicCombo.Input(
                    "image_options",    
                    options=image_combos,
                    display_name="image_options",
                    tooltip="The image options for reference.",
                ),
                io.DynamicCombo.Input(
                    "audio_options",
                    options=audio_combos,
                    display_name="audio_options",
                    tooltip="The audio options for reference.",
                ),
                io.DynamicCombo.Input(
                    "video_options",
                    options=video_combos,
                    display_name="video_options",
                    tooltip="The video options for reference.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls, 
        app_id, 
        app_secret, 
        view_server_address,
        open_field_name,
        open_field_value,
        frame_count_field_name,
        image_local_save_ext,
        text_options, 
        image_options, 
        audio_options, 
        video_options,
    ) -> io.NodeOutput:
        config_data = {
            "app_id": app_id,
            "app_secret": app_secret,
            "view_server_address": view_server_address,
            "open_field_name": open_field_name,
            "open_field_value": open_field_value,
            "frame_count_field_name": frame_count_field_name,
            "image_local_save_ext": image_local_save_ext,
            "text_options": text_options,
            "image_options": image_options,
            "audio_options": audio_options,
            "video_options": video_options
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)
        return io.NodeOutput()
    
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return random.randint(0, 0xFFFFFFFFFFFFFFFF)

class DAFeishuLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)

        texts, images, audios, videos = utils.get_feishu_outputs_from_config(config)

        return io.Schema(
            node_id="DAFeishuLoad",   
            display_name="DA Feishu Load",
            category="DALab/Tools/Feishu",
            description="Load the feishu table data.",
            inputs=[
                io.String.Input(
                    "app_token", 
                    optional=False
                ),
                io.String.Input(
                    "table_id", 
                    optional=False
                ),
                io.String.Input(
                    "view_id", 
                    optional=False
                ),
            ],
            outputs=[
                *texts,
                *images,
                *audios,
                *videos,
                io.Int.Output(
                    "frame_count",
                    tooltip="The frame count number of each video record in the feishu.",
                    is_output_list=True,
                ),
                io.Custom("FEISHU_RECORD_IDS").Output(
                    "feishu_record_ids",
                    tooltip="The feishu record ids.",
                    is_output_list=True,
                ),
                io.Custom("FEISHU_CONFIG").Output(
                    "feishu_config",
                    tooltip="The feishu config.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, 
        app_token: str, 
        table_id: str, 
        view_id: str,
    ) -> io.NodeOutput:
        if not app_token or not table_id or not view_id:
            raise ValueError("app_token, table_id and view_id are required")
        
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=True)

        texts, images, audios, videos, frames, record_ids = FeishuManager(config).fetch_and_process_data(
            app_token, table_id, view_id
        )

        feishu_config = {
            "app_token": app_token,
            "table_id": table_id,
            "view_id": view_id,
        }
        
        results = [
            *texts,
            *images,
            *audios,
            *videos,
            frames,
            record_ids,
            feishu_config
        ]

        logger.info(f"[UncleDa Tools] DAFeishuLoad record number: {len(record_ids)}")

        return io.NodeOutput(*results)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return random.randint(0, 0xFFFFFFFFFFFFFFFF)
