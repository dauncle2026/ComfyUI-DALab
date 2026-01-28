import os, sys
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

base_dir = os.path.dirname(os.path.abspath(__file__))
libs_dir = os.path.join(base_dir, "src","libs")
if libs_dir not in sys.path:
    sys.path.insert(0, libs_dir)

from .src.nodes.da_flux1 import DAFlux1, DAFlux1Config
from .src.nodes.da_flux2 import DAFlux2, DAFlux2Config
from .src.nodes.da_flux2_klein import DAFlux2Klein, DAFlux2KleinConfig
from .src.nodes.da_z_image import DAZImage, DAZImageConfig
from .src.nodes.da_qwen_image import DAQwenImage, DAQwenImageConfig
from .src.nodes.da_qwen_image_edit import DAQwenImageEdit, DAQwenImageEditConfig
from .src.nodes.da_global_config import DAGlobalConfig

from .src.nodes.da_wan_t2v import DAWanT2V, DAWanT2VConfig
from .src.nodes.da_wan_i2v import DAWanI2V, DAWanI2VConfig
from .src.nodes.da_wan_s2v import DAWanS2V, DAWanS2VConfig
from .src.nodes.da_ltx2 import DALTX2, DALTX2Config

from .src.nodes.da_index_tts2 import DAIndexTTS2, DAIndexTTS2Config
from .src.nodes.da_cosy_voice3 import DACosyVoice3, DACosyVoice3Config
from .src.nodes.da_voxcpm15 import DAVoxCPM15, DAVoxCPM15Config

from .src.nodes.da_feishu import DAFeishuConfig, DAFeishuLoad
from .src.nodes.da_file import DASaveImage, DASaveVideo, DAConcatVideo, DASaveAudio
from .src.nodes.da_qwen_llm import DAQwenLLM, DAQwenLLMConfig, DAQwenVL

from .src.nodes.da_infinite_talk import DAInfiniteTalk, DAInfiniteTalkConfig

from .src.nodes.da_dwpose import DADWPose, DADWPoseConfig
from .src.nodes.da_florence2 import DAFlorence2, DAFlorence2Config
from .src.nodes.da_sam2 import DASAM2, DASAM2Config
from .src.nodes.da_wan_animate import DAWanAnimate, DAWanAnimateConfig

class DauncleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            DAGlobalConfig,
            DAFlux1,
            DAFlux1Config,
            DAFlux2,
            DAFlux2Config,
            DAFlux2Klein,
            DAFlux2KleinConfig,
            DAQwenImage,
            DAQwenImageConfig,
            DAQwenImageEdit,
            DAQwenImageEditConfig,
            DAZImage,
            DAZImageConfig,
            DAWanT2V,
            DAWanT2VConfig,
            DAWanI2V,
            DAWanI2VConfig,
            DAWanS2V,
            DAWanS2VConfig,
            DAWanAnimate,
            DAWanAnimateConfig,
            DALTX2,
            DALTX2Config,
            DAFeishuLoad,
            DAFeishuConfig,
            DASaveImage,
            DASaveVideo,
            DAConcatVideo,
            DASaveAudio,
            DAQwenLLM,
            DAQwenVL,
            DAQwenLLMConfig,
            DAIndexTTS2,
            DAIndexTTS2Config,
            DACosyVoice3,
            DACosyVoice3Config,
            DAVoxCPM15,
            DAVoxCPM15Config,
            DAInfiniteTalk,
            DAInfiniteTalkConfig,
            DADWPose,
            DADWPoseConfig,
            DAFlorence2,
            DAFlorence2Config,
            DASAM2,
            DASAM2Config,
        ]

async def comfy_entrypoint() -> DauncleExtension: 
    return DauncleExtension()

WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY"]