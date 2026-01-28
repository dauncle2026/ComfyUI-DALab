import os
import torch
import comfy.model_management
import logging

from .base import BaseModelWrapper, _WRAPPERS_CACHE

logger = logging.getLogger(__name__)


class VoxCPM15ModelWrapper(BaseModelWrapper):
    def __init__(self, model_dir, device, enable_denoiser=True, optimize=True):
        super().__init__()
        self.model_dir = model_dir
        self.current_device = self.load_device

        voxcpm_model_path = os.path.join(model_dir, "VoxCPM1.5")
        zipenhancer_model_path = os.path.join(model_dir, "speech_zipenhancer_ans_multiloss_16k_base")
        asr_model_id = os.path.join(model_dir, "SenseVoiceSmall")

        try:
            from funasr import AutoModel
            from ...libs.voxcpm import VoxCPM
        except ImportError as e:
            logger.error(f"Failed to import AutoModel. Please check if the environment matches requirements. Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred while importing AutoModel: {e}")
            raise e

        root_logger = logging.getLogger()
        original_level = root_logger.level

        try:
            root_logger.setLevel(logging.ERROR)

            self.asr_model = AutoModel(
                model=asr_model_id,
                disable_update=True,
                log_level='ERROR',
                device=str(self.current_device),
            )

            self.model = VoxCPM(
                voxcpm_model_path=voxcpm_model_path,
                zipenhancer_model_path=zipenhancer_model_path,
                enable_denoiser=enable_denoiser,
                optimize=optimize
            )
        finally:
            root_logger.setLevel(original_level)

        self.size = self._calculate_size()

    def prompt_wav_recognition(self, prompt_wav) -> str:
        if prompt_wav is None:
            return ""

        res = self.asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        text = res[0]["text"].split('|>')[-1]
        return text

    def _calculate_size(self):
        total_size = 0

        for submodule in [self.model.tts_model]:
            if submodule is not None:
                total_size += comfy.model_management.module_size(submodule)

        # include denoiser size if exists
        if hasattr(self.model, 'denoiser') and self.model.denoiser is not None:
            denoiser = self.model.denoiser
            if hasattr(denoiser, '_pipeline') and denoiser._pipeline is not None:
                pipeline_obj = denoiser._pipeline
                if hasattr(pipeline_obj, 'model') and pipeline_obj.model is not None:
                    total_size += comfy.model_management.module_size(pipeline_obj.model)

        return total_size

    def model_dtype(self):
        return torch.float16

    def model_patches_to(self, device):
        if isinstance(device, torch.dtype):
            return

        target_device = device

        self.model.tts_model.to(target_device)
        self.model.tts_model.device = target_device

        self.asr_model.model.to(target_device)
        self.asr_model.model.device = target_device

        if hasattr(self.model, 'denoiser') and self.model.denoiser is not None:
            denoiser = self.model.denoiser
            if hasattr(denoiser, '_pipeline') and denoiser._pipeline is not None:
                pipeline_obj = denoiser._pipeline
                if hasattr(pipeline_obj, 'model') and pipeline_obj.model is not None:
                    pipeline_obj.model.to(target_device)

        self.current_device = target_device

    def _clear_extras(self):
        if hasattr(self, 'asr_model') and self.asr_model is not None:
            if hasattr(self.asr_model, 'model'):
                del self.asr_model.model
            del self.asr_model
            self.asr_model = None

        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'denoiser') and self.model.denoiser is not None:
                denoiser = self.model.denoiser
                if hasattr(denoiser, '_pipeline') and denoiser._pipeline is not None:
                    if hasattr(denoiser._pipeline, 'model'):
                        del denoiser._pipeline.model
                    del denoiser._pipeline
                    denoiser._pipeline = None
                del self.model.denoiser
                self.model.denoiser = None


def get_voxcpm15(model_dir, device, enable_denoiser=True, optimize=True):
    key = ("voxcpm15", model_dir, device, enable_denoiser, optimize)
    if key not in _WRAPPERS_CACHE:
        wrapper = VoxCPM15ModelWrapper(
            model_dir,
            device=device,
            enable_denoiser=enable_denoiser,
            optimize=optimize
        )
        _WRAPPERS_CACHE[key] = wrapper
    return _WRAPPERS_CACHE[key]
