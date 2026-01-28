import os


def get_config_file_path(config_name: str) -> str:
    """Get the path to a config file by name."""
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        f"{config_name}_config.json"
    )
