# Kaggle Guide (Notebook)

This guide assumes:
- input datasets are under `/kaggle/input`
- outputs should go to `/kaggle/working/output`

## 1. Clone the repo
```bash
!git clone https://github.com/leonsong09/audio2srt.git
%cd audio2srt
```

If git is missing in the notebook:
```bash
!apt-get update -y && apt-get install -y git
```

If GitHub is blocked, use Gitee:
```bash
!git clone https://gitee.com/leons9/audio2srt.git
%cd audio2srt
```

## 2. One-step setup (env + model)
```bash
!bash -lc "MODE=whisper-api bash setup.sh"
```

## 3. Load Kaggle env config (recommended)
```bash
!bash -lc "set -a; source configs/kaggle.env; source configs/runtime.env; set +a"
```

## 4. Transcribe only (no LLM)
```bash
!bash -lc "set -a; source configs/kaggle.env; set +a; python -m app.cli.main transcribe --input /kaggle/input/your-dataset/demo.mp4 --output-format SRT"
```

## 5. Full pipeline with LLM
```bash
!bash -lc "set -a; source configs/kaggle.env; set +a; python -m app.cli.main full --input /kaggle/input/your-dataset/demo.mp4 --need-split --need-optimize --llm-service OPENAI --llm-api-key $OPENAI_API_KEY --llm-model gpt-4o-mini"
```

Use DeepSeek:
```bash
!bash -lc "set -a; source configs/kaggle.env; set +a; python -m app.cli.main full --input /kaggle/input/your-dataset/demo.mp4 --need-optimize --llm-service DEEPSEEK --llm-api-key $OPENAI_API_KEY --llm-model deepseek-chat"
```

## 6. Output location
```
/kaggle/working/output
```

## 7. Notes
- Kaggle may not have `faster-whisper-xxl`; use `--transcribe-model WHISPER_API` if needed.
- If ffmpeg is missing, install it in the notebook or use a kernel that includes it.
- If you want local FasterWhisper models, set `MODE=faster-whisper` and choose `MODEL_NAME`:
  `large-v2`, `large-v3`, `large-v3-turbo`,
  `belle-whisper-large-v2-zh`, `belle-whisper-large-v3-zh`,
  `belle-whisper-large-v3-zh-punct` (default), `belle-whisper-large-v3-turbo-zh`.

Model selection tips:
- Chinese + punctuation: `belle-whisper-large-v3-zh-punct`
- Faster speed: `large-v3-turbo` or `belle-whisper-large-v3-turbo-zh`
- Baseline comparison: `large-v2` or `belle-whisper-large-v2-zh`
