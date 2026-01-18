# OpenI Guide (GPU Server + Jupyter)

This guide assumes you are using OpenI Jupyter with:
- code path: `/tmp/code`
- dataset path: `/tmp/dataset`
- output path: `/tmp/output`

## 1. Install git (if missing)
```bash
git --version || (apt-get update && apt-get install -y git)
```

## 2. Clone the repo
```bash
cd /tmp/code
git clone https://github.com/leonsong09/audio2srt.git
cd audio2srt
```

If GitHub is blocked, use Gitee:
```bash
cd /tmp/code
git clone https://gitee.com/leons9/audio2srt.git
cd audio2srt
```

Non-interactive script (Gitee):
```bash
bash fetch_gitee.sh
```

## 3. One-step setup (env + model)
```bash
MODE=faster-whisper MODEL_NAME=belle-whisper-large-v3-zh-punct bash setup.sh
```

Optional models:
- `large-v2`
- `large-v3`
- `large-v3-turbo`
- `belle-whisper-large-v2-zh`
- `belle-whisper-large-v3-zh`
- `belle-whisper-large-v3-zh-punct` (default)
- `belle-whisper-large-v3-turbo-zh`

This will:
- create `.venv`
- install Python deps
- download the FasterWhisper model
- generate `configs/runtime.env` with paths

## 4. Load OpenI env config (recommended)
```bash
set -a
source configs/openi.env
source configs/runtime.env
set +a
```

## 5. Interactive mode (recommended)
```bash
python -m app.cli.main interactive --use-c2net --dataset-subdir whisper-audio
```

## 6. One-command pipeline
Transcribe + split + optimize:
```bash
python -m app.cli.main full \
  --use-c2net --dataset-subdir whisper-audio \
  --input demo.mp4 \
  --spleeter --spleeter-stems 5 \
  --need-split --need-optimize \
  --output-suffix .srt \
  --llm-service OPENAI --llm-api-key $OPENAI_API_KEY --llm-model gpt-4o-mini
```

Use DeepSeek:
```bash
python -m app.cli.main full \
  --use-c2net --dataset-subdir whisper-audio \
  --input demo.mp4 \
  --need-optimize \
  --llm-service DEEPSEEK --llm-api-key $OPENAI_API_KEY --llm-model deepseek-chat
```

## 7. Output location
Results are saved to:
```
/tmp/output
```
or `c2net_context.output_path` if `--use-c2net` is on.

## 8. Notes
- If `faster-whisper-xxl` is not available, use `--transcribe-model WHISPER_API`.
- Add `--upload-output` when your OpenI task supports output upload.
