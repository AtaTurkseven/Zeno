import yaml
from pathlib import Path

DEFAULT_CONFIG = {
    "llm": {
        "provider": "ollama",
        "model": "mistral",
        "base_url": "http://localhost:11434",
        "timeout": 90,
    },
    "project": {
        "max_file_size_kb": 150,
        "include_extensions": [
            ".py", ".ino", ".cpp", ".c", ".h",
            ".md", ".txt", ".log", ".json",
            ".yaml", ".yml", ".sh", ".rs",
            ".toml", ".cfg", ".ini",
        ],
        "exclude_dirs": [
            ".git", "__pycache__", "node_modules",
            ".venv", "venv", "env", "build", "dist",
            ".pio", ".platformio",
        ],
    },
    "memory": {
        "path": "./memory",
        "session_notes": "SESSION_NOTES.md",
    },
    "display": {
        "card_width": 80,
        "use_color": True,
    },
}


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    # Shallow merge per section — user values override defaults
    cfg = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    for section, values in user_cfg.items():
        if isinstance(values, dict) and section in cfg:
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg
