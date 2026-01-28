import logging

from .config_loader import ConfigLoader
from .paths import get_config_file_path

_GLOBAL_CONFIG_PATH = get_config_file_path("global")


class DALabLogger:
    """Centralized logger for all DALab nodes."""

    _instance = None
    _logger = None
    _debug = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger()
        return cls._instance

    def _init_logger(self):
        self._logger = logging.getLogger("DALab")

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[DALab] %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        self._logger.propagate = False
        self._load_config()

    def _load_config(self):
        try:
            config = ConfigLoader(_GLOBAL_CONFIG_PATH, strict=False)
            self._debug = config.get("debug", False)
        except Exception:
            self._debug = False
        self._update_level()

    def _update_level(self):
        self._logger.setLevel(logging.INFO if self._debug else logging.WARNING)

    def set_debug(self, debug: bool):
        self._debug = debug
        self._update_level()

    @property
    def debug(self) -> bool:
        return self._debug

    def info(self, message: str):
        self._logger.info(message)

    def warning(self, message: str):
        self._logger.warning(message)

    def error(self, message: str):
        self._logger.error(message)


# Global instance - use directly: logger.info("message")
logger = DALabLogger()
