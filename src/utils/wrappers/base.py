import gc
import torch
import psutil
import comfy.model_management
import warnings

from ..paths import get_config_file_path
from ..config_loader import ConfigLoader
from ..logger import logger

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_WRAPPERS_CACHE: dict = {}

def get_memory_info() -> tuple[float, float]:
    """Get current RAM and VRAM usage in GB."""
    process = psutil.Process()
    ram_gb = process.memory_info().rss / 1024 / 1024 / 1024
    vram_gb = torch.cuda.memory_allocated() / 1024 / 1024 / 1024 if torch.cuda.is_available() else 0
    return ram_gb, vram_gb


def format_memory(ram_gb: float, vram_gb: float) -> str:
    """Format memory info as string."""
    return f"RAM: {ram_gb:.2f} GB, VRAM: {vram_gb:.2f} GB"


def log_memory(stage: str):
    """Log current memory usage with stage name."""
    ram_gb, vram_gb = get_memory_info()
    logger.info(f"[{stage}] {format_memory(ram_gb, vram_gb)}")


class MemoryTracker:
    """Context manager for tracking memory changes during an operation."""

    def __init__(self, operation_name: str, wrapper_name: str = ""):
        self.operation_name = operation_name
        self.wrapper_name = wrapper_name
        self.ram_before = 0.0
        self.vram_before = 0.0

    def __enter__(self):
        self.ram_before, self.vram_before = get_memory_info()
        prefix = f"{self.wrapper_name} " if self.wrapper_name else ""
        logger.info(f"{prefix}{self.operation_name} | {format_memory(self.ram_before, self.vram_before)}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ram_after, vram_after = get_memory_info()
        ram_diff = self.ram_before - ram_after
        vram_diff = self.vram_before - vram_after

        prefix = f"{self.wrapper_name} " if self.wrapper_name else ""
        logger.info(
            f"{prefix}{self.operation_name} done | {format_memory(ram_after, vram_after)} | "
            f"delta: RAM {-ram_diff:+.2f} GB, VRAM {-vram_diff:+.2f} GB"
        )
        return False

    def log_step(self, step_name: str):
        """Log memory at an intermediate step."""
        ram, vram = get_memory_info()
        ram_diff = self.ram_before - ram
        vram_diff = self.vram_before - vram
        logger.info(f"  - {step_name}: {format_memory(ram, vram)} (delta: RAM {-ram_diff:+.2f}, VRAM {-vram_diff:+.2f})")

class BaseModelWrapper:
    """Base class for model wrappers with memory management support."""

    parent = None
    _global_config_path = get_config_file_path("global")

    SUBMODULES: list[str] = []
    TENSOR_NAMES: list[str] = []

    def __init__(self):
        self.load_device = comfy.model_management.get_torch_device()
        self.offload_device = comfy.model_management.unet_offload_device()
        self.current_device = self.offload_device
        self.size = 0
        self.model = None

    @property
    def wrapper_name(self) -> str:
        return self.__class__.__name__

    def load_wrapper(self):
        comfy.model_management.load_model_gpu(self)
        logger.info(f"{self.wrapper_name} loaded to {self.current_device}")

    def unload_wrapper(self):
        ram_before, vram_before = get_memory_info()
        self.model_patches_to(self.offload_device)
        comfy.model_management.soft_empty_cache()
        ram_after, vram_after = get_memory_info()
        logger.info(
            f"{self.wrapper_name} offloaded | {format_memory(ram_after, vram_after)} | "
            f"delta: RAM {ram_after - ram_before:+.2f} GB, VRAM {vram_after - vram_before:+.2f} GB"
        )

    @classmethod
    def should_release_after_run(cls) -> bool:
        try:
            config = ConfigLoader(cls._global_config_path, strict=False)
            return config.get("release_after_run", False)
        except FileNotFoundError:
            return False

    @classmethod
    def should_offload_after_run(cls) -> bool:
        try:
            config = ConfigLoader(cls._global_config_path, strict=False)
            return config.get("offload_after_run", False)
        except FileNotFoundError:
            return False

    def _clear_submodules(self):
        """Clear all submodules defined in SUBMODULES. Override for custom cleanup."""
        for name in self.SUBMODULES:
            if hasattr(self.model, name):
                mod = getattr(self.model, name)
                if mod is not None:
                    del mod
                setattr(self.model, name, None)

    def _clear_tensors(self):
        """Clear all tensors defined in TENSOR_NAMES. Override for custom cleanup."""
        for name in self.TENSOR_NAMES:
            if hasattr(self.model, name):
                tensor = getattr(self.model, name)
                if tensor is not None:
                    del tensor
                setattr(self.model, name, None)

    def _clear_extras(self):
        """Override this method to clear additional model-specific components."""
        pass

    def release(self):
        """Release the model and free memory."""
        if self.model is None:
            return

        ram_before, vram_before = get_memory_info()

        self._clear_tensors()
        gc.collect()
        torch.cuda.empty_cache()

        # Clear submodules
        self._clear_submodules()
        gc.collect()
        torch.cuda.empty_cache()

        self._clear_extras()
        gc.collect()
        torch.cuda.empty_cache()

        if self.model is not None:
            try:
                for param in self.model.parameters():
                    param.data = torch.empty(0)
                for buffer in self.model.buffers():
                    buffer.data = torch.empty(0)
            except Exception:
                pass 

        del self.model
        self.model = None

        gc.collect()
        torch.cuda.empty_cache()
        comfy.model_management.soft_empty_cache()

        ram_after, vram_after = get_memory_info()
        logger.info(
            f"{self.wrapper_name} release | {format_memory(ram_after, vram_after)} | "
            f"delta: RAM {ram_after - ram_before:+.2f} GB, VRAM {vram_after - vram_before:+.2f} GB"
        )

    def model_dtype(self):
        return torch.float16

    def model_size(self):
        return self.size

    def model_patches_models(self):
        return []

    def current_loaded_device(self):
        return self.current_device

    def loaded_size(self):
        return self.size

    def partially_load(self, device, extra_memory=0, force_patch_weights=False):
        pass

    def is_clone(self, other):
        if hasattr(other, 'model') and self.model is other.model:
            return True
        return False

    def detach(self, unpatch_all=True):
        pass

    def model_patches_to(self, device):
        pass

def release_wrapper_cache(wrapper_type: str = None):
    """Release wrappers and free memory completely."""
    global _WRAPPERS_CACHE

    if wrapper_type is None:
        for key, wrapper in list(_WRAPPERS_CACHE.items()):
            if wrapper is not None:
                wrapper.release()
        _WRAPPERS_CACHE.clear()
    else:
        keys_to_remove = [k for k in _WRAPPERS_CACHE if k[0] == wrapper_type]
        for key in keys_to_remove:
            wrapper = _WRAPPERS_CACHE.pop(key, None)
            if wrapper is not None:
                wrapper.release()

    comfy.model_management.soft_empty_cache()


def offload_wrapper_cache(wrapper_type: str = None):
    """Offload wrappers to CPU, keeping them cached for fast reload."""
    global _WRAPPERS_CACHE

    if wrapper_type is None:
        for key, wrapper in list(_WRAPPERS_CACHE.items()):
            if wrapper is not None:
                wrapper.unload_wrapper()
    else:
        keys_to_offload = [k for k in _WRAPPERS_CACHE if k[0] == wrapper_type]
        for key in keys_to_offload:
            wrapper = _WRAPPERS_CACHE.get(key)
            if wrapper is not None:
                wrapper.unload_wrapper()

    comfy.model_management.soft_empty_cache()


def handle_wrapper_after_run(wrapper_type: str = None):
    """Handle wrapper cleanup based on global config (release or offload)."""
    if BaseModelWrapper.should_release_after_run():
        release_wrapper_cache(wrapper_type)
    elif BaseModelWrapper.should_offload_after_run():
        offload_wrapper_cache(wrapper_type)
