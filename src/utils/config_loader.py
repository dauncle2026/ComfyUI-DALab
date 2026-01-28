import os
import json

class ConfigLoader:
    _cache = {}

    def __init__(self, config_path, strict=True):
        self.config_path = config_path
        self.strict = strict
        self._data = self._load_data()

    def _load_data(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"[DALab] ConfigLoader: Config file missing: {self.config_path}")

        current_mtime = os.path.getmtime(self.config_path)

        cached = ConfigLoader._cache.get(self.config_path)
        if cached and cached["mtime"] == current_mtime:
            return cached["data"]

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"[DALab] ConfigLoader: Invalid JSON format: {e}")

        ConfigLoader._cache[self.config_path] = {
            "mtime": current_mtime,
            "data": data
        }

        return data

    def get(self, key, default=None, override_strict=None):
        is_strict = override_strict if override_strict is not None else self.strict

        if key not in self._data:
            if is_strict:
                raise KeyError(f"[DALab] ConfigLoader: Missing required key '{key}' in {self.config_path}")
            return default

        return self._data[key]

    def __getitem__(self, key):
        return self.get(key, override_strict=True)

    def __repr__(self):
        return f"<Config strict={self.strict} data={self._data}>"
