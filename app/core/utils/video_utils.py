import os
import subprocess
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger("video_utils")


def video2audio(input_file: str, output: str = "", audio_track_index: int = 0) -> bool:
    """使用 ffmpeg 将视频转换为音频."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = str(output_path)

    logger.info(f"提取音轨索引 {audio_track_index}")
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-map",
        f"0:a:{audio_track_index}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-y",
        output,
    ]

    logger.info(f"转换为音频执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            ),
        )
        if result.returncode == 0 and Path(output).is_file():
            logger.info("音频转换成功")
            return True
        logger.error("音频转换失败")
        return False
    except subprocess.CalledProcessError as e:
        logger.error("== ffmpeg 执行失败 ==")
        logger.error(f"返回码: {e.returncode}")
        logger.error(f"命令: {' '.join(e.cmd)}")
        if e.stdout:
            logger.error(f"标准输出: {e.stdout}")
        if e.stderr:
            logger.error(f"标准错误: {e.stderr}")
        return False
    except Exception as e:
        logger.exception(f"音频转换出错: {str(e)}")
        return False
