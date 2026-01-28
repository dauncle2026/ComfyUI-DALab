import logging

from comfy_api.latest import io

from ..utils import utils
from ..utils.config_loader import ConfigLoader
from ..utils.model_manager import ModelManager
from ..utils.logger import logger
from ..utils.paths import get_config_file_path

_CONFIG_FILE_PATH = get_config_file_path("global")

class DAGlobalConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        config = ConfigLoader(_CONFIG_FILE_PATH, strict=False)

        return io.Schema(
            node_id="DAGlobalConfig",
            display_name="DA Global Config",
            category="DALab/Config",
            description="Global settings for all DALab nodes. Run once before other nodes.",
            is_output_node=True,
            inputs=[
                io.Boolean.Input(
                    "debug",
                    default=config.get("debug", False),
                    tooltip="Enable debug logging for DALab nodes.",
                    display_name="debug",
                ),
                io.Boolean.Input(
                    "model_switch_offload",
                    default=config.get("model_switch_offload", False),
                    tooltip="Offload current model when switching to another (e.g., text_encoder -> diffusion -> vae). Saves VRAM but slower.",
                    display_name="model_switch_offload",
                ),
                io.Boolean.Input(
                    "offload_after_run",
                    default=config.get("offload_after_run", False),
                    tooltip="Offload all models from VRAM to RAM after each node execution. Models stay cached for fast reload.",
                    display_name="offload_after_run",
                ),
                io.Boolean.Input(
                    "release_after_run",
                    default=config.get("release_after_run", False),
                    tooltip="Release all models and free VRAM/RAM after each node execution. Models need to be reloaded from disk.",
                    display_name="release_after_run",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls,
        debug: bool,
        model_switch_offload: bool,
        offload_after_run: bool,
        release_after_run: bool,
    ) -> io.NodeOutput:
        config_data = {
            "debug": debug,
            "model_switch_offload": model_switch_offload,
            "offload_after_run": offload_after_run,
            "release_after_run": release_after_run,
        }
        utils.save_json(config_data, _CONFIG_FILE_PATH)

        logger.set_debug(debug)

        manager = ModelManager()
        manager.configure(
            model_switch_offload=model_switch_offload,
            offload_after_run=offload_after_run,
            release_after_run=release_after_run,
        )

        logging.info(
            f"[DALab] Global config applied: "
            f"debug={debug}, "
            f"model_switch_offload={model_switch_offload}, "
            f"offload_after_run={offload_after_run}, "
            f"release_after_run={release_after_run}"
        )

        return io.NodeOutput()
