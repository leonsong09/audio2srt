import logging
import os
from pathlib import Path

VERSION = "v1.4.0"
YEAR = 2025
APP_NAME = "VideoCaptioner"
AUTHOR = "Weifeng"

HELP_URL = "https://github.com/WEIFENG2333/VideoCaptioner"
GITHUB_REPO_URL = "https://github.com/WEIFENG2333/VideoCaptioner"
RELEASE_URL = "https://github.com/WEIFENG2333/VideoCaptioner/releases/latest"
FEEDBACK_URL = "https://github.com/WEIFENG2333/VideoCaptioner/issues"

# 路径
def _env_path(*keys: str, default: Path) -> Path:
    for key in keys:
        value = os.getenv(key)
        if value:
            return Path(value)
    return default


ROOT_PATH = _env_path(
    "VIDEO_CAPTIONER_ROOT",
    "OPENI_CODE_PATH",
    "C2NET_CODE_PATH",
    default=Path(__file__).parent.parent,
)

RESOURCE_PATH = ROOT_PATH / "resource"
APPDATA_PATH = _env_path(
    "VIDEO_CAPTIONER_APPDATA",
    default=ROOT_PATH / "AppData",
)
WORK_PATH = _env_path(
    "VIDEO_CAPTIONER_WORK",
    default=ROOT_PATH / "work-dir",
)
OUTPUT_PATH = _env_path(
    "VIDEO_CAPTIONER_OUTPUT",
    "OPENI_OUTPUT_PATH",
    "C2NET_OUTPUT_PATH",
    default=ROOT_PATH / "output",
)
DATASET_PATH = _env_path(
    "VIDEO_CAPTIONER_DATASET",
    "OPENI_DATASET_PATH",
    "C2NET_DATASET_PATH",
    default=Path("/tmp/dataset"),
)


SUBTITLE_STYLE_PATH = RESOURCE_PATH / "subtitle_style"

LOG_PATH = APPDATA_PATH / "logs"
SETTINGS_PATH = APPDATA_PATH / "settings.json"
CACHE_PATH = APPDATA_PATH / "cache"
MODEL_PATH = APPDATA_PATH / "models"

# 日志配置
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 创建路径
for p in [CACHE_PATH, LOG_PATH, WORK_PATH, MODEL_PATH, OUTPUT_PATH]:
    p.mkdir(parents=True, exist_ok=True)
