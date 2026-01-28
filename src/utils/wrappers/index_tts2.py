import torch
import comfy.model_management
import logging

from .base import BaseModelWrapper, _WRAPPERS_CACHE
from ..logger import logger


class IndexTTSModelWrapper(BaseModelWrapper):
    SUBMODULES = ["gpt", "semantic_model", "s2mel", "campplus_model", "bigvgan", "semantic_codec"]

    TENSOR_NAMES = [
        "cache_spk_cond",
        "cache_s2mel_style",
        "cache_s2mel_prompt",
        "cache_emo_cond",
        "cache_mel",
        "semantic_mean",
        "semantic_std",
    ]

    def __init__(self, config_path, model_dir, device, use_fp16):
        super().__init__()
        self.config_path = config_path
        self.model_dir = model_dir
        self.use_fp16 = use_fp16
        self.current_device = device

        try:
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
        except Exception:
            pass

        try:
            from ...libs.indextts.index_tts2 import IndexTTS2
        except ImportError as e:
            logger.error(f"Failed to import IndexTTS2. Please check if the environment matches requirements. Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred while importing IndexTTS2: {e}")
            raise e

        self.model = IndexTTS2(
            cfg_path=config_path,
            model_dir=model_dir,
            use_fp16=use_fp16,
            device=str(device),
            use_cuda_kernel=False,
            use_deepspeed=False,
        )

        self.size = self._calculate_size()

    def _calculate_size(self):
        total = 0
        for name in self.SUBMODULES:
            mod = getattr(self.model, name, None)
            if mod and hasattr(mod, "state_dict"):
                total += comfy.model_management.module_size(mod)

        if hasattr(self.model, 'qwen_emo') and hasattr(self.model.qwen_emo, 'model'):
            total += comfy.model_management.module_size(self.model.qwen_emo.model)

        return total

    def model_dtype(self):
        return torch.float16 if self.use_fp16 else torch.float32

    def model_patches_to(self, device):
        if isinstance(device, torch.dtype):
            return

        target_device = device

        for submodule in self.SUBMODULES:
            mod = getattr(self.model, submodule, None)
            if mod and hasattr(mod, "to"):
                mod.to(target_device)

        for name in self.TENSOR_NAMES:
            current_tensor = getattr(self.model, name, None)
            if current_tensor is not None and isinstance(current_tensor, torch.Tensor):
                moved_tensor = current_tensor.to(target_device)
                setattr(self.model, name, moved_tensor)

        if hasattr(self.model, 'qwen_emo') and self.model.qwen_emo is not None:
            if hasattr(self.model.qwen_emo, 'model') and self.model.qwen_emo.model is not None:
                self.model.qwen_emo.model.to(target_device)

        if hasattr(self.model, 'emo_matrix') and self.model.emo_matrix is not None:
            if isinstance(self.model.emo_matrix, (list, tuple)):
                self.model.emo_matrix = tuple(t.to(target_device) for t in self.model.emo_matrix)
            elif isinstance(self.model.emo_matrix, torch.Tensor):
                self.model.emo_matrix = self.model.emo_matrix.to(target_device)

        if hasattr(self.model, 'spk_matrix') and self.model.spk_matrix is not None:
            if isinstance(self.model.spk_matrix, (list, tuple)):
                self.model.spk_matrix = tuple(t.to(target_device) for t in self.model.spk_matrix)
            elif isinstance(self.model.spk_matrix, torch.Tensor):
                self.model.spk_matrix = self.model.spk_matrix.to(target_device)

        self.current_device = target_device
        self.model.device = target_device

    def _clear_extras(self):
        if hasattr(self.model, 'qwen_emo') and self.model.qwen_emo is not None:
            if hasattr(self.model.qwen_emo, 'model') and self.model.qwen_emo.model is not None:
                del self.model.qwen_emo.model
                self.model.qwen_emo.model = None
            if hasattr(self.model.qwen_emo, 'tokenizer'):
                del self.model.qwen_emo.tokenizer
                self.model.qwen_emo.tokenizer = None
            del self.model.qwen_emo
            self.model.qwen_emo = None

        if hasattr(self.model, 'extract_features'):
            del self.model.extract_features
            self.model.extract_features = None

        if hasattr(self.model, 'emo_matrix'):
            del self.model.emo_matrix
            self.model.emo_matrix = None
        if hasattr(self.model, 'spk_matrix'):
            del self.model.spk_matrix
            self.model.spk_matrix = None

        if hasattr(self.model, 'normalizer'):
            del self.model.normalizer
            self.model.normalizer = None
        if hasattr(self.model, 'tokenizer'):
            del self.model.tokenizer
            self.model.tokenizer = None

        if hasattr(self.model, 'cfg'):
            del self.model.cfg
            self.model.cfg = None

        if hasattr(self.model, 'mel_fn'):
            del self.model.mel_fn
            self.model.mel_fn = None

def get_index_tts2(config_path, model_dir, device, use_fp16):
    class WhutupWeText(logging.Filter):
        def filter(self, record):
            return False

    zh_logger = logging.getLogger("wetext-zh_normalizer")
    en_logger = logging.getLogger("wetext-en_normalizer")

    zh_logger.addFilter(WhutupWeText())
    en_logger.addFilter(WhutupWeText())

    key = ("index_tts2", config_path, model_dir, device, use_fp16)
    if key not in _WRAPPERS_CACHE:
        wrapper = IndexTTSModelWrapper(
            config_path,
            model_dir,
            device=device,
            use_fp16=use_fp16
        )
        _WRAPPERS_CACHE[key] = wrapper
    return _WRAPPERS_CACHE[key]
