"""项目配置：负责读取环境变量与模型配置。"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE_PATH = PROJECT_ROOT_DIR / ".env"

# 显式加载项目根目录 .env，避免依赖运行时 cwd。
load_dotenv(dotenv_path=ENV_FILE_PATH)


def _get_required_env(key: str) -> str:
    """读取必填环境变量。

    Args:
        key: 环境变量名称。

    Returns:
        str: 对应环境变量的值。
    """
    value = os.getenv(key)
    if not value:
        raise ValueError(f"未检测到 {key}，请先在 .env 文件中配置。")
    return value


def _get_env_with_default(key: str, default: str) -> str:
    """读取可选环境变量，未配置时使用默认值。

    Args:
        key: 环境变量名称。
        default: 默认值。

    Returns:
        str: 环境变量值或默认值。
    """
    return os.getenv(key, default)


DASHSCOPE_API_KEY = _get_required_env("DASHSCOPE_API_KEY")
CHAT_MODEL_NAME = _get_env_with_default("CHAT_MODEL_NAME", "qwen3-max")
EMBEDDING_MODEL_NAME = _get_env_with_default("EMBEDDING_MODEL_NAME", "text-embedding-v4")
