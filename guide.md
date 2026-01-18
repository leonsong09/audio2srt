# OpenI GPU 版 audio2srt 使用指南

本项目已移除 GUI，仅保留在 GPU 服务器上运行的命令行功能。核心功能包括：
1) Whisper 语音转录（FasterWhisper 或 Whisper API）
2) Spleeter 人声/伴奏分离（最佳分离=5 stems）
3) LLM 字幕断句、优化、翻译
4) 字幕样式（ASS）选择

输出只生成字幕文件，默认输出到 `output` 目录（OpenI 使用 `c2net_context.output_path`）。

## 1. 打开 Jupyter 终端
在 OpenI 项目里打开 Jupyter，进入终端操作。

## 2. 代码放到 /tmp/code
如果服务器没有 git：
```bash
git --version || (apt-get update && apt-get install -y git)
```

推荐方式（用 Git）：
```bash
cd /tmp/code
git clone https://github.com/leonsong09/audio2srt.git
cd audio2srt
```

如果 GitHub 无法访问，可用 Gitee：
```bash
cd /tmp/code
git clone https://gitee.com/leons9/audio2srt.git
cd audio2srt
```

或者一键脚本（Gitee，无交互）：
```bash
bash fetch_gitee.sh
```

如果你把代码上传到数据集，也可以先复制到 `/tmp/code`：
```bash
cp -r /tmp/dataset/你的代码目录 /tmp/code/audio2srt
cd /tmp/code/audio2srt
```

## 3. 安装依赖
推荐一键安装（环境 + 模型）：
```bash
MODE=faster-whisper MODEL_NAME=belle-whisper-large-v3-zh-punct bash setup.sh
```

可选模型（`MODEL_NAME`）：
- `large-v2`
- `large-v3`
- `large-v3-turbo`
- `belle-whisper-large-v2-zh`
- `belle-whisper-large-v3-zh`
- `belle-whisper-large-v3-zh-punct`（默认）
- `belle-whisper-large-v3-turbo-zh`

## 3.1 模型选择说明
推荐中文场景优先使用 `belle-whisper-large-v3-zh-punct`（带标点）。  
如果想速度更快，选择 `large-v3-turbo` 或 `belle-whisper-large-v3-turbo-zh`。  
需要对比效果时，再尝试 `large-v2` / `belle-whisper-large-v2-zh`。

或手动安装：
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

检查 ffmpeg：
```bash
ffmpeg -version
```

如果没有 `faster-whisper-xxl`，可以改用 Whisper API（见交互模式）。

## 4. 交互模式（新手推荐）
```bash
python -m app.cli.main interactive --use-c2net --dataset-subdir whisper-audio
```

交互提示会依次问你：
- 输入文件（相对 `/tmp/dataset/whisper-audio` 或绝对路径）
- 输出格式（SRT/ASS/TXT/JSON）
- 是否断句、优化、翻译
- 翻译服务（OpenAI/Bing/Google/DeepLX）
- LLM 配置（API Key、Base URL、Model）
- 是否使用 Spleeter（人声分离，默认 5 stems）

## 5. 常用命令（非交互）
完整流程（转录 + 断句 + 优化）：
```bash
python -m app.cli.main full \
  --use-c2net --dataset-subdir whisper-audio \
  --input demo.mp4 \
  --spleeter --spleeter-stems 5 \
  --need-split --need-optimize \
  --output-suffix .srt \
  --llm-service OPENAI --llm-api-key $OPENAI_API_KEY --llm-model gpt-4o-mini
```

只做语音转录（不走 LLM）：
```bash
python -m app.cli.main transcribe \
  --use-c2net --dataset-subdir whisper-audio \
  --input demo.mp4 \
  --output-format SRT
```

字幕翻译（LLM）：
```bash
python -m app.cli.main full \
  --use-c2net --dataset-subdir whisper-audio \
  --input demo.mp4 \
  --need-translate --translator-service OPENAI \
  --target-language SIMPLIFIED_CHINESE \
  --llm-service OPENAI --llm-api-key $OPENAI_API_KEY --llm-model gpt-4o-mini
```

只处理已有字幕（断句/优化/翻译）：
```bash
python -m app.cli.main subtitle \
  --use-c2net --dataset-subdir whisper-audio \
  --input demo.srt \
  --need-split --need-optimize \
  --output-suffix .ass \
  --subtitle-style default \
  --llm-service OPENAI --llm-api-key $OPENAI_API_KEY --llm-model gpt-4o-mini
```

## 6. 字幕样式（ASS）
样式文件在 `resource/subtitle_style/`：
- `default.txt`
- `毕导科普风.txt`
- `番剧可爱风.txt`
- `竖屏.txt`

命令中使用：
```bash
--output-suffix .ass --subtitle-style 竖屏
```

也可以传自定义样式文件路径：
```bash
--subtitle-style /tmp/dataset/custom_style.txt
```

## 7. OpenI c2net 路径说明
项目内部会调用：
```python
from c2net.context import prepare, upload_output

c2net_context = prepare()
whisper_audio_path = c2net_context.dataset_path + "/whisper-audio"
you_should_save_here = c2net_context.output_path
```

对应命令行参数：
- `--use-c2net`：启用 c2net
- `--dataset-subdir whisper-audio`：数据集子目录
- `--upload-output`：自动回传结果（训练任务可用）

## 8. 配置文件（推荐）
项目里自带模板：
- `configs/openi.env`
- `configs/kaggle.env`

用法示例（OpenI 终端）：
```bash
set -a
source configs/openi.env
source configs/runtime.env
set +a
python -m app.cli.main interactive --use-c2net --dataset-subdir whisper-audio
```

## 9. Kaggle 使用说明
在 Kaggle Notebook 里执行：
```bash
!git clone https://github.com/leonsong09/audio2srt.git
%cd audio2srt
!pip install -r requirements.txt
```

加载环境变量：
```bash
!bash -lc "set -a; source configs/kaggle.env; set +a; python -m app.cli.main transcribe --input /kaggle/input/你的数据集/xxx.mp4"
```

如果你想启用 LLM：
```bash
!bash -lc "set -a; source configs/kaggle.env; set +a; python -m app.cli.main full --input /kaggle/input/你的数据集/xxx.mp4 --need-optimize --llm-service OPENAI --llm-api-key $OPENAI_API_KEY --llm-model gpt-4o-mini"
```

DeepSeek 示例：
```bash
!bash -lc "set -a; source configs/kaggle.env; set +a; python -m app.cli.main full --input /kaggle/input/你的数据集/xxx.mp4 --need-optimize --llm-service DEEPSEEK --llm-api-key $OPENAI_API_KEY --llm-model deepseek-chat"
```

## 10. 结果位置
输出字幕文件在：
```
/tmp/output
```
或 `c2net_context.output_path` 指向的目录。

## 11. 常见问题
1) 找不到 `faster-whisper-xxl`  
   - 用 Whisper API：`--transcribe-model WHISPER_API`
2) 没有 ffmpeg  
   - 需要在镜像中安装 ffmpeg 或换到包含 ffmpeg 的镜像
3) LLM 报错  
   - 检查 `--llm-api-key` 和 `--llm-base-url`
   - 可以加 `--skip-llm-check` 跳过测试

## 12. Gitee 推送说明
本地推送到 Gitee（建议使用个人访问令牌）：
```bash
git remote add gitee https://gitee.com/leons9/audio2srt.git
git push -u gitee main
```
