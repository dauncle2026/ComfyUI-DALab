import torch
import comfy.model_management

from .base import BaseModelWrapper, _WRAPPERS_CACHE

try:
    from transformers import Sam2Model, Sam2Processor
    from transformers import Sam2VideoModel, Sam2VideoProcessor
except ImportError:
    raise ImportError(
        "SAM2 not found in transformers. "
        "Please update transformers: pip install -U transformers>=4.45.0"
    )


class SAM2ImageModelWrapper(BaseModelWrapper):
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

        import transformers.utils.logging as tf_logging
        original_verbosity = tf_logging.get_verbosity()
        tf_logging.set_verbosity_error()
        try:
            self.model = Sam2Model.from_pretrained(
                model_path,
                dtype=self.dtype,
            ).to(self.offload_device)

            self.processor = Sam2Processor.from_pretrained(model_path)
        finally:
            tf_logging.set_verbosity(original_verbosity)

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
        """Clear SAM2 processor."""
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
            self.processor = None


class SAM2VideoModelWrapper(BaseModelWrapper):
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

        self.model = Sam2VideoModel.from_pretrained(
            model_path,
            dtype=self.dtype,
        ).to(self.offload_device)

        self.processor = Sam2VideoProcessor.from_pretrained(model_path)

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
        """Clear SAM2 Video processor."""
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
            self.processor = None


def get_sam2_image(model_path: str, precision: str):
    key = ("sam2_image", model_path, precision)
    if key not in _WRAPPERS_CACHE:
        wrapper = SAM2ImageModelWrapper(model_path, precision)
        _WRAPPERS_CACHE[key] = wrapper
    return _WRAPPERS_CACHE[key]


def get_sam2_video(model_path: str, precision: str):
    key = ("sam2_video", model_path, precision)
    if key not in _WRAPPERS_CACHE:
        wrapper = SAM2VideoModelWrapper(model_path, precision)
        _WRAPPERS_CACHE[key] = wrapper
    return _WRAPPERS_CACHE[key]
