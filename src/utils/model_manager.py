import weakref
import logging

import comfy.sd
import comfy.utils
import comfy.model_management as mm
import comfy.model_patcher
import comfy.audio_encoders.audio_encoders

from .config_loader import ConfigLoader
from .logger import logger
from .paths import get_config_file_path

_GLOBAL_CONFIG_PATH = get_config_file_path("global")


class ModelManager:
    _instance = None
    _cache = {}
    _lora_cache = {} 

    _model_switch_offload = False
    _offload_after_run = False
    _release_after_run = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        try:
            config = ConfigLoader(_GLOBAL_CONFIG_PATH, strict=False)
            self._model_switch_offload = config.get("model_switch_offload", False)
            self._offload_after_run = config.get("offload_after_run", False)
            self._release_after_run = config.get("release_after_run", False)
        except Exception:
            pass

    def configure(
        self,
        model_switch_offload: bool = False,
        offload_after_run: bool = False,
        release_after_run: bool = False
    ):
        self._model_switch_offload = model_switch_offload
        self._offload_after_run = offload_after_run
        self._release_after_run = release_after_run
        logger.info(
            f"ModelManager configured: "
            f"model_switch_offload={self._model_switch_offload}, "
            f"offload_after_run={self._offload_after_run}, "
            f"release_after_run={self._release_after_run}"
        )

    @property
    def model_switch_offload(self) -> bool:
        return self._model_switch_offload

    @property
    def offload_after_run(self) -> bool:
        return self._offload_after_run

    @property
    def release_after_run(self) -> bool:
        return self._release_after_run

    @staticmethod
    def _make_key(*args):
        return str(args)

    def _get_or_load(self, cache_key, loader_fn):
        logger.info(f"Loading model: {cache_key[:80]}...")

        if self._model_switch_offload:
            self._offload_other_models(cache_key)

        if cache_key in self._cache:
            model_ref = self._cache[cache_key]
            model = model_ref() if isinstance(model_ref, weakref.ref) else model_ref
            if model is not None:
                logger.info(f"Using cached model: {cache_key[:80]}...")
                return model
            else:
                del self._cache[cache_key]
                logger.info(f"Cached model was garbage collected: {cache_key[:80]}...")

        try:
            log_logger = logging.getLogger()
            log_logger.setLevel(logging.ERROR)

            model = loader_fn()
        finally:
            log_logger.setLevel(logging.INFO)

        if self._release_after_run:
            try:
                self._cache[cache_key] = weakref.ref(model)
            except TypeError:
                self._cache[cache_key] = model
        else:
            self._cache[cache_key] = model

        return model

    def _offload_other_models(self, exclude_key: str = None):
        for key, cached_ref in self._cache.items():
            if key == exclude_key:
                continue
            cached_model = cached_ref() if isinstance(cached_ref, weakref.ref) else cached_ref
            if cached_model is not None:
                self._offload_model(key, cached_model)

    def get_text_encoder(self, paths: list, clip_type):
        cache_key = self._make_key("clip", tuple(paths), clip_type)
        return self._get_or_load(cache_key, lambda: comfy.sd.load_clip(
            ckpt_paths=paths,
            clip_type=clip_type
        ))

    def get_vae(self, path: str):
        cache_key = self._make_key("vae", path)
        return self._get_or_load(cache_key, lambda: comfy.sd.VAE(
            sd=comfy.utils.load_torch_file(path)
        ))

    def get_diffusion_model(self, path: str, model_options=None):
        model_options = model_options or {}
        cache_key = self._make_key("diffusion", path)
        return self._get_or_load(cache_key, lambda: comfy.sd.load_diffusion_model(
            path, model_options=model_options
        ))

    def get_audio_encoder(self, path: str):
        cache_key = self._make_key("audio_encoder", path)
        return self._get_or_load(
            cache_key,
            lambda: comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd(
                comfy.utils.load_torch_file(path, safe_load=True)
            )
        )

    def get_clip_vision(self, path: str):
        import comfy.clip_vision
        cache_key = self._make_key("clip_vision", path)
        return self._get_or_load(cache_key, lambda: comfy.clip_vision.load(path))

    def get_audio_vae(self, path: str):
        from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
        cache_key = self._make_key("audio_vae", path)

        def loader():
            sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
            return AudioVAE(sd, metadata)

        return self._get_or_load(cache_key, loader)

    def get_lora_dict(self, path: str):
        if path in self._lora_cache:
            logger.info(f"Using cached LoRA: {path}")
            return self._lora_cache[path]

        logger.info(f"Loading LoRA from disk: {path}")
        try:
            lora_dict = comfy.utils.load_torch_file(path, safe_load=True)
        except Exception as e:
            logger.error(f"Failed to load LoRA: {path}, error: {e}")
            raise e

        self._lora_cache[path] = lora_dict
        
        return lora_dict

    def _unload_from_comfyui(self, model):
        if isinstance(model, comfy.model_patcher.ModelPatcher):
            patcher = model
        elif hasattr(model, "patcher"):
            patcher = model.patcher
        elif hasattr(model, "device_manager") and hasattr(model.device_manager, "patcher"):
            patcher = model.device_manager.patcher
        else:
            return False

        current_loaded_models = mm.current_loaded_models
        indices_to_remove = []

        for i, loaded in enumerate(current_loaded_models):
            loaded_model = loaded.model
            if loaded_model is None:
                continue

            is_match = loaded_model is patcher
            if not is_match and hasattr(patcher, 'is_clone'):
                is_match = patcher.is_clone(loaded_model)
            if not is_match and hasattr(loaded_model, 'is_clone'):
                is_match = loaded_model.is_clone(patcher)

            if is_match:
                indices_to_remove.append(i)

        for i in reversed(indices_to_remove):
            loaded_model = current_loaded_models.pop(i)
            model_name = loaded_model.model.model.__class__.__name__ if loaded_model.model.model else loaded_model.model.__class__.__name__
            logger.info(f"Unloading by ComfyUI: {model_name}")
            loaded_model.model_unload()

        if indices_to_remove:
            mm.soft_empty_cache()

        return True

    def _offload_model(self, cache_key: str, model):
        import torch

        if self._unload_from_comfyui(model):
            return True

        if isinstance(model, torch.nn.Module):
            try:
                model.to(mm.unet_offload_device())
                logger.info(f"Unloading by Manual: {model.__class__.__name__}")
                mm.soft_empty_cache()
                return True
            except Exception as e:
                logger.warning(f"Failed to offload model to {mm.unet_offload_device()}: {e}")

        return False

    def release_all(self):
        logger.info("Releasing all models...")

        for key, cached_ref in list(self._cache.items()):
            cached_model = cached_ref() if isinstance(cached_ref, weakref.ref) else cached_ref
            if cached_model is not None:
                self._offload_model(key, cached_model)

        self._cache.clear()
        self._lora_cache.clear()
        mm.soft_empty_cache()
        logger.info("All models released and VRAM cleared")

    def offload_all(self):
        logger.info("Offloading all models to CPU...")

        for key, cached_ref in list(self._cache.items()):
            cached_model = cached_ref() if isinstance(cached_ref, weakref.ref) else cached_ref
            if cached_model is not None:
                self._offload_model(key, cached_model)

        mm.soft_empty_cache()
        logger.info("All models offloaded to CPU")
