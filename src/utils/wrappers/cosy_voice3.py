import torch
import comfy.model_management
import logging

from .base import BaseModelWrapper, _WRAPPERS_CACHE,log_memory

logger = logging.getLogger(__name__)


class CosyVoice3ModelWrapper(BaseModelWrapper):
    def __init__(self, model_dir, device, use_fp16):
        super().__init__()
        self.model_dir = model_dir
        self.use_fp16 = use_fp16
        self.current_device = self.load_device

        try:
            from ...libs.cosyvoice.cli.cosyvoice import CosyVoice3
            import whisper
        except ImportError as e:
            logger.error(f"Failed to import CosyVoice3. Please check if the environment matches requirements. Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred while importing CosyVoice3: {e}")
            raise e

        self.model = CosyVoice3(
            model_dir=model_dir,
            fp16=use_fp16,
            device=self.current_device
        )
        self.model_controller = self.model.model

        self.asr_model = whisper.load_model(
            "base",
            download_root=model_dir,
            device=self.current_device
        )

        self.size = self._calculate_size()

    def _calculate_size(self):
        total_size = 0

        for submodule in [self.model_controller.llm, self.model_controller.flow, self.model_controller.hift]:
            if submodule is not None:
                total_size += comfy.model_management.module_size(submodule)

        total_size += comfy.model_management.module_size(self.asr_model)

        return total_size

    def model_dtype(self):
        return torch.float16 if self.use_fp16 else torch.float32

    def model_patches_to(self, device):
        if isinstance(device, torch.dtype):
            return

        target_device = device

        for submodel in [self.model_controller.llm, self.model_controller.flow, self.model_controller.hift]:
            if submodel and hasattr(submodel, "to"):
                submodel.to(target_device)
            else:
                logger.warning(f"Submodel {submodel} not found or does not have a 'to' method.")

        self.asr_model.to(target_device)

        self.model_controller.device = target_device
        self.current_device = target_device
 
    def prompt_wav_recognition(self, prompt_wav) -> str:
        if prompt_wav is None:
            return ""

        res = self.asr_model.transcribe(prompt_wav)
        text = res["text"].strip()

        return text

    def _clear_extras(self):
        if hasattr(self, 'asr_model') and self.asr_model is not None:
            del self.asr_model
            self.asr_model = None

        if hasattr(self, 'model_controller') and self.model_controller is not None:
            if hasattr(self.model_controller, 'llm'):
                del self.model_controller.llm
            if hasattr(self.model_controller, 'flow'):
                del self.model_controller.flow
            if hasattr(self.model_controller, 'hift'):
                del self.model_controller.hift
            del self.model_controller
            self.model_controller = None


def get_cosy_voice3(model_dir, device, use_fp16):
    key = ("cosy_voice3", model_dir, device, use_fp16)
    if key not in _WRAPPERS_CACHE:
        wrapper = CosyVoice3ModelWrapper(
            model_dir,
            device=device,
            use_fp16=use_fp16
        )
        _WRAPPERS_CACHE[key] = wrapper
    return _WRAPPERS_CACHE[key]
