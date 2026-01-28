import torch
import warnings
import comfy.model_management

from .base import BaseModelWrapper, _WRAPPERS_CACHE

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError:
    raise ImportError(
        "transformers not found or too old. "
        "Please update transformers: pip install -U transformers>=4.51.0"
    )


class Florence2ModelWrapper(BaseModelWrapper):
    DTYPE_MAP = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    def __init__(self, model_path: str, precision: str):
        super().__init__()
        self.model_path = model_path
        self.precision = precision
        self.dtype = self.DTYPE_MAP.get(precision, torch.float16)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=self.dtype,
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(self.offload_device)

            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.size = self._calculate_size()

    def _calculate_size(self):
        return comfy.model_management.module_size(self.model)

    def model_dtype(self):
        return self.dtype

    def model_patches_to(self, device):
        if isinstance(device, torch.dtype):
            return

        self.model.to(device)
        self.current_device = device

    def _clear_extras(self):
        """Clear Florence2 processor."""
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
            self.processor = None


def get_florence2(model_path: str, precision: str):
    key = ("florence2", model_path, precision)
    if key not in _WRAPPERS_CACHE:
        wrapper = Florence2ModelWrapper(model_path, precision)
        _WRAPPERS_CACHE[key] = wrapper
    return _WRAPPERS_CACHE[key]
