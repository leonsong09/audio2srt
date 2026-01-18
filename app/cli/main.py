import argparse
import getpass
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

from app.config import DATASET_PATH, MODEL_PATH, OUTPUT_PATH, SUBTITLE_STYLE_PATH
from app.core.asr import transcribe
from app.core.asr.asr_data import ASRData
from app.core.entities import (
    LANGUAGES,
    FasterWhisperModelEnum,
    LLMServiceEnum,
    SubtitleConfig,
    SubtitleLayoutEnum,
    TranscribeConfig,
    TranscribeModelEnum,
    TranscribeOutputFormatEnum,
    TranslatorServiceEnum,
    VadMethodEnum,
)
from app.core.llm.check_llm import check_llm_connection
from app.core.optimize.optimize import SubtitleOptimizer
from app.core.split.split import SubtitleSplitter
from app.core.translate import BingTranslator, DeepLXTranslator, GoogleTranslator, LLMTranslator
from app.core.translate.types import TargetLanguage
from app.core.utils.logger import setup_logger
from app.core.utils.video_utils import video2audio

logger = setup_logger("videocaptioner.cli")

DEFAULT_LLM_CONFIGS = {
    LLMServiceEnum.OPENAI: {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    LLMServiceEnum.SILICON_CLOUD: {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "gpt-4o-mini",
    },
    LLMServiceEnum.DEEPSEEK: {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    LLMServiceEnum.OLLAMA: {
        "base_url": "http://localhost:11434/v1",
        "model": "llama2",
    },
    LLMServiceEnum.LM_STUDIO: {
        "base_url": "http://localhost:1234/v1",
        "model": "qwen2.5:7b",
    },
    LLMServiceEnum.GEMINI: {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-pro",
    },
    LLMServiceEnum.CHATGLM: {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4",
    },
}


def _default_code_path() -> Path:
    for env in ("OPENI_CODE_PATH", "C2NET_CODE_PATH", "CODE_PATH"):
        value = os.getenv(env)
        if value:
            return Path(value)
    if Path("/tmp/code").exists():
        return Path("/tmp/code")
    return Path.cwd()


def _default_dataset_path() -> Path:
    for env in ("OPENI_DATASET_PATH", "C2NET_DATASET_PATH", "DATASET_PATH"):
        value = os.getenv(env)
        if value:
            return Path(value)
    if DATASET_PATH.exists():
        return DATASET_PATH
    if Path("/tmp/dataset").exists():
        return Path("/tmp/dataset")
    return _default_code_path()


def _default_output_dir() -> Path:
    for env in ("OPENI_OUTPUT_PATH", "C2NET_OUTPUT_PATH", "OUTPUT_PATH"):
        value = os.getenv(env)
        if value:
            return Path(value)
    return OUTPUT_PATH


def _prepare_c2net_context(dataset_subdir: str) -> tuple[Path, Path, Callable[[], None]]:
    try:
        from c2net.context import prepare, upload_output
    except ImportError as exc:
        raise RuntimeError("c2net SDK not available. Disable --use-c2net.") from exc

    context = prepare()
    dataset_path = Path(context.dataset_path)
    if dataset_subdir:
        dataset_path = dataset_path / dataset_subdir
    output_path = Path(context.output_path)
    return dataset_path, output_path, upload_output


def _resolve_runtime_paths(args) -> tuple[Path, Path, Optional[Callable[[], None]]]:
    dataset_path = _default_dataset_path()
    output_dir_value = getattr(args, "output_dir", "")
    output_dir = Path(output_dir_value) if output_dir_value else _default_output_dir()
    upload_func = None

    if getattr(args, "use_c2net", False):
        dataset_path, c2net_output, upload_output = _prepare_c2net_context(
            getattr(args, "dataset_subdir", "whisper-audio")
        )
        if not output_dir_value:
            output_dir = c2net_output
        if getattr(args, "upload_output", False):
            upload_func = upload_output

    return dataset_path, output_dir, upload_func


def _parse_enum(value: Optional[str], enum_cls):
    if value is None:
        return None
    raw = value.strip()
    raw_norm = raw.lower().replace("-", "_").replace(" ", "_")
    for item in enum_cls:
        if raw_norm == item.name.lower():
            return item
        if str(item.value).lower() == raw.lower():
            return item
        if str(item.value).lower().replace("-", "_").replace(" ", "_") == raw_norm:
            return item
    return None


def _parse_language(value: str) -> str:
    raw = value.strip()
    if raw in LANGUAGES:
        return LANGUAGES[raw]
    for key, lang_code in LANGUAGES.items():
        if raw.lower() == lang_code.lower():
            return lang_code
        if raw.lower() == str(key).lower():
            return lang_code
    return raw


def _resolve_input_path(input_path: str, dataset_path: Path) -> Path:
    path = Path(input_path)
    if path.exists():
        return path
    candidate = dataset_path / input_path
    if candidate.exists():
        return candidate
    return path


def _resolve_subtitle_style(style_value: str) -> str:
    if not style_value:
        return ""
    style_path = Path(style_value)
    if style_path.is_file():
        return style_path.read_text(encoding="utf-8")

    candidate = Path(style_value)
    if candidate.suffix.lower() != ".txt":
        candidate = SUBTITLE_STYLE_PATH / f"{style_value}.txt"
    else:
        candidate = SUBTITLE_STYLE_PATH / style_value
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")

    for style_file in SUBTITLE_STYLE_PATH.glob("*.txt"):
        if style_file.stem.lower() == style_value.lower():
            return style_file.read_text(encoding="utf-8")

    raise FileNotFoundError(f"Subtitle style not found: {style_value}")


def _build_output_path(
    output_dir: Path, input_path: Path, suffix: str, name_override: Optional[str] = None
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = name_override or input_path.stem
    if not suffix.startswith("."):
        suffix = "." + suffix
    return output_dir / f"{file_stem}{suffix}"


def _apply_llm_env(base_url: str, api_key: str) -> None:
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key


def _resolve_llm_config(args) -> tuple[str, str, str, LLMServiceEnum]:
    service = _parse_enum(args.llm_service, LLMServiceEnum) or LLMServiceEnum.OPENAI
    defaults = DEFAULT_LLM_CONFIGS.get(service, DEFAULT_LLM_CONFIGS[LLMServiceEnum.OPENAI])
    base_url = args.llm_base_url or os.getenv("OPENAI_BASE_URL") or defaults["base_url"]
    api_key = args.llm_api_key or os.getenv("OPENAI_API_KEY") or ""
    model = args.llm_model or defaults["model"]
    if not api_key:
        raise ValueError("LLM API key is required (use --llm-api-key or OPENAI_API_KEY).")
    return base_url, api_key, model, service


def _run_spleeter(input_audio: Path, stems: int, output_dir: Path) -> Path:
    import subprocess

    if stems not in (2, 4, 5):
        raise ValueError("Spleeter stems must be 2, 4, or 5.")

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "spleeter",
        "separate",
        "-p",
        f"spleeter:{stems}stems",
        "-o",
        str(output_dir),
        str(input_audio),
    ]
    logger.info("Running Spleeter: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Spleeter CLI not found. Please install spleeter.") from exc
    stem_dir = output_dir / input_audio.stem
    vocals_path = stem_dir / "vocals.wav"
    if not vocals_path.exists():
        raise FileNotFoundError(f"Spleeter vocals not found: {vocals_path}")
    return vocals_path


def _transcribe_to_asr(
    input_path: Path,
    config: TranscribeConfig,
    use_spleeter: bool,
    spleeter_stems: int,
    audio_track_index: int,
) -> ASRData:
    with tempfile.TemporaryDirectory(prefix="videocaptioner_audio_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        wav_path = temp_dir_path / "audio.wav"
        logger.info("Extracting audio...")
        if not video2audio(
            str(input_path), output=str(wav_path), audio_track_index=audio_track_index
        ):
            raise RuntimeError("Audio extraction failed")

        audio_input = wav_path
        if use_spleeter:
            logger.info("Separating vocals with Spleeter...")
            spleeter_dir = temp_dir_path / "spleeter"
            audio_input = _run_spleeter(wav_path, spleeter_stems, spleeter_dir)

        logger.info("Transcribing with %s", config.transcribe_model)
        return transcribe(
            str(audio_input),
            config,
            callback=lambda p, m: logger.info("%s%% %s", p, m),
        )


def _process_subtitles(
    subtitle_path: Path,
    output_path: Path,
    config: SubtitleConfig,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    skip_llm_check: bool = False,
) -> None:
    asr_data = ASRData.from_subtitle_file(str(subtitle_path))

    needs_llm = config.need_optimize or config.need_split or (
        config.need_translate and config.translator_service == TranslatorServiceEnum.OPENAI
    )
    if needs_llm:
        if not (llm_base_url and llm_api_key and llm_model):
            raise ValueError("LLM config is required for split/optimize/LLM translate.")
        _apply_llm_env(llm_base_url, llm_api_key)
        config.base_url = llm_base_url
        config.api_key = llm_api_key
        config.llm_model = llm_model
        if not skip_llm_check:
            ok, message = check_llm_connection(llm_base_url, llm_api_key, llm_model)
            if not ok:
                raise RuntimeError(f"LLM check failed: {message or ''}")

    if config.need_split:
        splitter = SubtitleSplitter(
            thread_num=config.thread_num,
            model=config.llm_model or "",
            max_word_count_cjk=config.max_word_count_cjk,
            max_word_count_english=config.max_word_count_english,
        )
        logger.info("Splitting subtitles...")
        asr_data = splitter.split_subtitle(asr_data)

    if config.need_optimize:
        if not config.llm_model:
            raise ValueError("LLM model is required for optimize.")
        optimizer = SubtitleOptimizer(
            thread_num=config.thread_num,
            batch_num=config.batch_size,
            model=config.llm_model,
            custom_prompt=config.custom_prompt_text or "",
        )
        logger.info("Optimizing subtitles...")
        asr_data = optimizer.optimize_subtitle(asr_data)
        asr_data.remove_punctuation()

    if config.need_translate:
        logger.info("Translating subtitles...")
        translator_service = config.translator_service
        if not config.target_language:
            raise ValueError("Target language is required for translate.")
        if translator_service == TranslatorServiceEnum.OPENAI:
            if not config.llm_model:
                raise ValueError("LLM model is required for LLM translate.")
            translator = LLMTranslator(
                thread_num=config.thread_num,
                batch_num=config.batch_size,
                target_language=config.target_language,
                model=config.llm_model,
                custom_prompt=config.custom_prompt_text or "",
                is_reflect=config.need_reflect,
            )
        elif translator_service == TranslatorServiceEnum.GOOGLE:
            translator = GoogleTranslator(
                thread_num=config.thread_num,
                batch_num=5,
                target_language=config.target_language,
                timeout=20,
            )
        elif translator_service == TranslatorServiceEnum.BING:
            translator = BingTranslator(
                thread_num=config.thread_num,
                batch_num=10,
                target_language=config.target_language,
            )
        elif translator_service == TranslatorServiceEnum.DEEPLX:
            os.environ["DEEPLX_ENDPOINT"] = config.deeplx_endpoint or ""
            translator = DeepLXTranslator(
                thread_num=config.thread_num,
                batch_num=5,
                target_language=config.target_language,
                timeout=20,
            )
        else:
            raise ValueError(f"Unsupported translator service: {translator_service}")
        asr_data = translator.translate_subtitle(asr_data)
        asr_data.remove_punctuation()

    asr_data.save(
        save_path=str(output_path),
        ass_style=config.subtitle_style or "",
        layout=config.subtitle_layout or SubtitleLayoutEnum.ORIGINAL_ON_TOP,
    )


def _add_common_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-service", default="OPENAI")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--skip-llm-check", action="store_true")


def _add_c2net_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use-c2net", action="store_true")
    parser.add_argument("--dataset-subdir", default="whisper-audio")
    parser.add_argument("--upload-output", action="store_true")


def _add_common_subtitle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subtitle-layout", default="ORIGINAL_ON_TOP")
    parser.add_argument("--subtitle-style", default="")
    parser.add_argument("--target-language", default="")
    parser.add_argument("--translator-service", default="OPENAI")
    parser.add_argument("--deeplx-endpoint", default="")
    parser.add_argument("--need-split", action="store_true")
    parser.add_argument("--need-optimize", action="store_true")
    parser.add_argument("--need-translate", action="store_true")
    parser.add_argument("--need-reflect", action="store_true")
    parser.add_argument("--thread-num", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-word-count-cjk", type=int, default=12)
    parser.add_argument("--max-word-count-english", type=int, default=18)
    parser.add_argument("--custom-prompt", default="")


def _build_subtitle_config(args) -> SubtitleConfig:
    layout = _parse_enum(args.subtitle_layout, SubtitleLayoutEnum) or SubtitleLayoutEnum.ORIGINAL_ON_TOP
    translator = _parse_enum(args.translator_service, TranslatorServiceEnum) or TranslatorServiceEnum.OPENAI
    target_language = _parse_enum(args.target_language, TargetLanguage)
    subtitle_style = _resolve_subtitle_style(args.subtitle_style) if args.subtitle_style else ""
    return SubtitleConfig(
        base_url="",
        api_key="",
        llm_model="",
        deeplx_endpoint=args.deeplx_endpoint,
        translator_service=translator,
        need_translate=args.need_translate,
        need_optimize=args.need_optimize,
        need_reflect=args.need_reflect,
        thread_num=args.thread_num,
        batch_size=args.batch_size,
        subtitle_layout=layout,
        max_word_count_cjk=args.max_word_count_cjk,
        max_word_count_english=args.max_word_count_english,
        need_split=args.need_split,
        target_language=target_language,
        subtitle_style=subtitle_style,
        custom_prompt_text=args.custom_prompt,
    )


def cmd_transcribe(args) -> None:
    dataset_path, output_dir, upload_func = _resolve_runtime_paths(args)
    input_path = _resolve_input_path(args.input, dataset_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    model = (
        _parse_enum(args.transcribe_model, TranscribeModelEnum)
        or TranscribeModelEnum.FASTER_WHISPER
    )
    if model not in (TranscribeModelEnum.FASTER_WHISPER, TranscribeModelEnum.WHISPER_API):
        raise ValueError("Only FASTER_WHISPER and WHISPER_API are supported.")
    output_format = (
        _parse_enum(args.output_format, TranscribeOutputFormatEnum)
        or TranscribeOutputFormatEnum.SRT
    )
    faster_model = _parse_enum(args.faster_whisper_model, FasterWhisperModelEnum)
    vad_method = _parse_enum(args.vad_method, VadMethodEnum)

    language = _parse_language(args.language) if args.language else ""
    output_suffix = output_format.value.lower()
    output_path = _build_output_path(
        output_dir, input_path, output_suffix, args.output_name
    )
    subtitle_style = _resolve_subtitle_style(args.subtitle_style) if args.subtitle_style else ""

    config = TranscribeConfig(
        transcribe_model=model,
        transcribe_language=language,
        need_word_time_stamp=args.word_timestamp,
        output_format=output_format,
        whisper_api_key=args.whisper_api_key,
        whisper_api_base=args.whisper_api_base,
        whisper_api_model=args.whisper_api_model,
        whisper_api_prompt=args.whisper_api_prompt,
        faster_whisper_program=args.faster_whisper_program,
        faster_whisper_model=faster_model,
        faster_whisper_model_dir=args.faster_whisper_model_dir or str(MODEL_PATH),
        faster_whisper_device=args.faster_whisper_device,
        faster_whisper_vad_filter=not args.disable_vad,
        faster_whisper_vad_threshold=args.vad_threshold,
        faster_whisper_vad_method=vad_method,
        faster_whisper_ff_mdx_kim2=args.ff_mdx_kim2,
        faster_whisper_one_word=args.one_word,
        faster_whisper_prompt=args.faster_whisper_prompt,
    )

    asr_data = _transcribe_to_asr(
        input_path,
        config,
        use_spleeter=args.spleeter,
        spleeter_stems=args.spleeter_stems,
        audio_track_index=args.audio_track_index,
    )
    if output_format == TranscribeOutputFormatEnum.ALL:
        formats_to_export = [
            fmt.value.lower()
            for fmt in TranscribeOutputFormatEnum
            if fmt != TranscribeOutputFormatEnum.ALL
        ]
        for fmt in formats_to_export:
            if fmt == "vtt":
                logger.warning("VTT output is not supported yet, skipping.")
                continue
            save_path = _build_output_path(output_dir, input_path, fmt, args.output_name)
            asr_data.save(
                str(save_path),
                ass_style=subtitle_style if fmt == "ass" else None,
                layout=SubtitleLayoutEnum.ORIGINAL_ON_TOP,
            )
            logger.info("Subtitle saved: %s", save_path)
    else:
        if output_suffix == "vtt":
            raise ValueError("VTT output is not supported yet.")
        asr_data.save(
            str(output_path),
            ass_style=subtitle_style if output_suffix == "ass" else None,
            layout=SubtitleLayoutEnum.ORIGINAL_ON_TOP,
        )
        logger.info("Subtitle saved: %s", output_path)
    if upload_func:
        logger.info("Uploading output to OpenI...")
        upload_func()


def cmd_subtitle(args) -> None:
    dataset_path, output_dir, upload_func = _resolve_runtime_paths(args)
    subtitle_path = _resolve_input_path(args.input, dataset_path)
    if not subtitle_path.exists():
        raise FileNotFoundError(f"Subtitle not found: {subtitle_path}")

    output_suffix = args.output_suffix or ".srt"
    output_path = _build_output_path(
        output_dir, subtitle_path, output_suffix, args.output_name
    )

    config = _build_subtitle_config(args)
    llm_base_url = llm_api_key = llm_model = None
    if config.need_split or config.need_optimize or (
        config.need_translate and config.translator_service == TranslatorServiceEnum.OPENAI
    ):
        llm_base_url, llm_api_key, llm_model, _ = _resolve_llm_config(args)

    _process_subtitles(
        subtitle_path,
        output_path,
        config,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        skip_llm_check=args.skip_llm_check,
    )
    logger.info("Subtitle saved: %s", output_path)
    if upload_func:
        logger.info("Uploading output to OpenI...")
        upload_func()


def cmd_split(args) -> None:
    args.need_split = True
    args.need_optimize = False
    args.need_translate = False
    cmd_subtitle(args)


def cmd_optimize(args) -> None:
    args.need_split = False
    args.need_optimize = True
    args.need_translate = False
    cmd_subtitle(args)


def cmd_translate(args) -> None:
    if not args.target_language:
        raise ValueError("--target-language is required for translate.")
    args.need_split = False
    args.need_optimize = False
    args.need_translate = True
    cmd_subtitle(args)


def _prompt(text: str, default: Optional[str] = None, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"{text}{suffix}: ").strip()
        if not value and default is not None:
            value = default
        if required and not value:
            print("Value required.")
            continue
        return value


def _prompt_yes_no(text: str, default: bool = False) -> bool:
    while True:
        suffix = "Y/n" if default else "y/N"
        value = input(f"{text} ({suffix}): ").strip().lower()
        if not value:
            return default
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False
        print("Please enter y or n.")


def _prompt_choice(text: str, options: list[tuple[str, str]], default_index: int = 0) -> str:
    print(text)
    for idx, (label, _) in enumerate(options, 1):
        print(f"  {idx}. {label}")
    while True:
        choice = input(f"Select [default {default_index + 1}]: ").strip()
        if not choice:
            return options[default_index][1]
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx][1]
        for label, value in options:
            if choice.lower() == label.lower():
                return value
            if choice.lower() == value.lower():
                return value
        print("Invalid selection.")


def _list_subtitle_style_options() -> list[tuple[str, str]]:
    options = [("Built-in default", "")]
    if SUBTITLE_STYLE_PATH.exists():
        for style_file in sorted(SUBTITLE_STYLE_PATH.glob("*.txt")):
            options.append((style_file.stem, style_file.stem))
    options.append(("Custom path", "__custom__"))
    return options


def _parse_target_language(value: str) -> Optional[TargetLanguage]:
    raw = value.strip().lower()
    alias_map = {
        "zh": TargetLanguage.SIMPLIFIED_CHINESE,
        "zh-cn": TargetLanguage.SIMPLIFIED_CHINESE,
        "zh-hans": TargetLanguage.SIMPLIFIED_CHINESE,
        "zh-tw": TargetLanguage.TRADITIONAL_CHINESE,
        "zh-hant": TargetLanguage.TRADITIONAL_CHINESE,
        "en": TargetLanguage.ENGLISH,
        "en-us": TargetLanguage.ENGLISH_US,
        "en-uk": TargetLanguage.ENGLISH_UK,
        "ja": TargetLanguage.JAPANESE,
        "ko": TargetLanguage.KOREAN,
        "fr": TargetLanguage.FRENCH,
        "de": TargetLanguage.GERMAN,
        "es": TargetLanguage.SPANISH,
        "ru": TargetLanguage.RUSSIAN,
        "pt": TargetLanguage.PORTUGUESE,
        "pt-br": TargetLanguage.PORTUGUESE_BR,
        "it": TargetLanguage.ITALIAN,
    }
    if raw in alias_map:
        return alias_map[raw]
    return _parse_enum(value, TargetLanguage)


def cmd_interactive(args) -> None:
    use_c2net_default = Path("/tmp/dataset").exists() or Path("/tmp/code").exists()
    use_c2net = args.use_c2net or _prompt_yes_no("Use c2net paths", default=use_c2net_default)
    dataset_subdir = args.dataset_subdir or "whisper-audio"
    if use_c2net and not args.use_c2net:
        dataset_subdir = _prompt("Dataset subdir", default=dataset_subdir)

    upload_output = args.upload_output

    if use_c2net:
        dataset_path, output_dir_default, _ = _prepare_c2net_context(dataset_subdir)
        if not args.upload_output:
            upload_output = _prompt_yes_no("Upload output to OpenI", default=False)
    else:
        dataset_path = _default_dataset_path()
        output_dir_default = _default_output_dir()

    print("VideoCaptioner CLI interactive mode")
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_dir_default}")

    input_path_str = _prompt("Input file (relative to dataset or absolute)", required=True)
    input_path = _resolve_input_path(input_path_str, dataset_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = Path(_prompt("Output directory", default=str(output_dir_default)))
    output_name = _prompt("Output file name (without extension)", default=input_path.stem)

    output_suffix = _prompt_choice(
        "Output subtitle format",
        [("SRT", ".srt"), ("ASS", ".ass"), ("TXT", ".txt"), ("JSON", ".json")],
        default_index=0,
    )
    subtitle_style = ""
    if output_suffix == ".ass":
        style_choice = _prompt_choice(
            "Subtitle style (ASS only)",
            _list_subtitle_style_options(),
            default_index=0,
        )
        if style_choice == "__custom__":
            style_choice = _prompt("Style file path", required=True)
        subtitle_style = style_choice

    subtitle_suffixes = {".srt", ".ass", ".json", ".vtt", ".txt"}
    is_subtitle = input_path.suffix.lower() in subtitle_suffixes

    need_split = _prompt_yes_no("Enable subtitle split", default=True)
    need_optimize = _prompt_yes_no("Enable subtitle optimize", default=False)
    need_translate = _prompt_yes_no("Enable subtitle translate", default=False)

    translator_service = "OPENAI"
    target_language = ""
    deeplx_endpoint = ""
    need_reflect = False
    if need_translate:
        translator_service = _prompt_choice(
            "Select translator service",
            [
                ("OpenAI LLM", "OPENAI"),
                ("Bing", "BING"),
                ("Google", "GOOGLE"),
                ("DeepLX", "DEEPLX"),
            ],
            default_index=0,
        )
        target_input = _prompt("Target language (e.g. zh, en, SIMPLIFIED_CHINESE)", required=True)
        target_enum = _parse_target_language(target_input)
        if not target_enum:
            raise ValueError("Unsupported target language.")
        target_language = target_enum.name
        if translator_service == "DEEPLX":
            deeplx_endpoint = _prompt("DeepLX endpoint", default="")
        if translator_service == "OPENAI":
            need_reflect = _prompt_yes_no("Enable reflection translation", default=False)

    llm_service = "OPENAI"
    llm_base_url = ""
    llm_api_key = ""
    llm_model = ""
    skip_llm_check = False
    if need_split or need_optimize or translator_service == "OPENAI":
        llm_service = _prompt_choice(
            "Select LLM service",
            [
                ("OpenAI", "OPENAI"),
                ("SiliconCloud", "SILICON_CLOUD"),
                ("DeepSeek", "DEEPSEEK"),
                ("Ollama", "OLLAMA"),
                ("LM Studio", "LM_STUDIO"),
                ("Gemini", "GEMINI"),
                ("ChatGLM", "CHATGLM"),
            ],
            default_index=0,
        )
        defaults = DEFAULT_LLM_CONFIGS.get(
            _parse_enum(llm_service, LLMServiceEnum) or LLMServiceEnum.OPENAI,
            DEFAULT_LLM_CONFIGS[LLMServiceEnum.OPENAI],
        )
        llm_base_url = _prompt("LLM base URL", default=defaults["base_url"])
        llm_model = _prompt("LLM model", default=defaults["model"])
        llm_api_key = getpass.getpass("LLM API key: ").strip()
        if not llm_api_key:
            raise ValueError("LLM API key is required.")
        skip_llm_check = _prompt_yes_no("Skip LLM connectivity check", default=False)

    thread_num = 10
    batch_size = 10
    max_word_count_cjk = 12
    max_word_count_english = 18
    subtitle_layout = "ORIGINAL_ON_TOP"
    if _prompt_yes_no("Advanced subtitle settings", default=False):
        subtitle_layout = _prompt_choice(
            "Subtitle layout",
            [
                ("Original on top", "ORIGINAL_ON_TOP"),
                ("Translate on top", "TRANSLATE_ON_TOP"),
                ("Only original", "ONLY_ORIGINAL"),
                ("Only translate", "ONLY_TRANSLATE"),
            ],
            default_index=0,
        )
        thread_num = int(_prompt("Thread num", default=str(thread_num)))
        batch_size = int(_prompt("Batch size", default=str(batch_size)))
        max_word_count_cjk = int(_prompt("Max word count (CJK)", default=str(max_word_count_cjk)))
        max_word_count_english = int(
            _prompt("Max word count (English)", default=str(max_word_count_english))
        )

    base_subtitle_args = {
        "output_dir": str(output_dir),
        "output_name": output_name,
        "output_suffix": output_suffix,
        "subtitle_layout": subtitle_layout,
        "subtitle_style": subtitle_style,
        "target_language": target_language,
        "translator_service": translator_service,
        "deeplx_endpoint": deeplx_endpoint,
        "need_split": need_split,
        "need_optimize": need_optimize,
        "need_translate": need_translate,
        "need_reflect": need_reflect,
        "thread_num": thread_num,
        "batch_size": batch_size,
        "max_word_count_cjk": max_word_count_cjk,
        "max_word_count_english": max_word_count_english,
        "custom_prompt": "",
        "llm_service": llm_service,
        "llm_base_url": llm_base_url,
        "llm_api_key": llm_api_key,
        "llm_model": llm_model,
        "skip_llm_check": skip_llm_check,
        "use_c2net": use_c2net,
        "dataset_subdir": dataset_subdir,
        "upload_output": upload_output,
    }

    if is_subtitle:
        subtitle_args = argparse.Namespace(
            **base_subtitle_args,
            input=str(input_path),
        )
        cmd_subtitle(subtitle_args)
        return

    transcribe_model = _prompt_choice(
        "Select transcribe model",
        [
            ("FasterWhisper (local)", "FASTER_WHISPER"),
            ("Whisper API", "WHISPER_API"),
        ],
        default_index=0,
    )
    language = _prompt("Transcribe language (empty for auto)", default="")
    audio_track_index = int(_prompt("Audio track index", default="0"))
    use_spleeter = _prompt_yes_no("Use Spleeter vocal separation", default=False)

    whisper_api_key = ""
    whisper_api_base = ""
    whisper_api_model = ""
    whisper_api_prompt = ""
    faster_whisper_model = "LARGE_V3"
    faster_whisper_program = ""
    faster_whisper_model_dir = ""
    faster_whisper_device = "cuda"
    disable_vad = False
    vad_threshold = 0.5
    vad_method = "SILERO_V4"
    ff_mdx_kim2 = False
    one_word = False
    faster_whisper_prompt = ""

    if transcribe_model == "WHISPER_API":
        whisper_api_base = _prompt("Whisper API base URL", default="https://api.openai.com/v1")
        whisper_api_model = _prompt("Whisper API model", default="whisper-1")
        whisper_api_key = getpass.getpass("Whisper API key: ").strip()
        whisper_api_prompt = _prompt("Whisper API prompt (optional)", default="")
    else:
        faster_whisper_model = _prompt("FasterWhisper model", default=faster_whisper_model)
        faster_whisper_device = _prompt("FasterWhisper device (cuda/cpu)", default="cuda")
        disable_vad = _prompt_yes_no("Disable VAD", default=False)
        if not disable_vad:
            vad_threshold = float(_prompt("VAD threshold", default=str(vad_threshold)))
            vad_method = _prompt("VAD method", default=vad_method)
        ff_mdx_kim2 = _prompt_yes_no("Enable ff_mdx_kim2", default=False)
        one_word = _prompt_yes_no("One word per segment", default=False)
        faster_whisper_prompt = _prompt("FasterWhisper prompt (optional)", default="")

    full_args = argparse.Namespace(
        **base_subtitle_args,
        input=str(input_path),
        language=language,
        transcribe_model=transcribe_model,
        word_timestamp=False,
        audio_track_index=audio_track_index,
        whisper_api_key=whisper_api_key,
        whisper_api_base=whisper_api_base,
        whisper_api_model=whisper_api_model,
        whisper_api_prompt=whisper_api_prompt,
        faster_whisper_program=faster_whisper_program,
        faster_whisper_model=faster_whisper_model,
        faster_whisper_model_dir=faster_whisper_model_dir,
        faster_whisper_device=faster_whisper_device,
        disable_vad=disable_vad,
        vad_threshold=vad_threshold,
        vad_method=vad_method,
        ff_mdx_kim2=ff_mdx_kim2,
        one_word=one_word,
        faster_whisper_prompt=faster_whisper_prompt,
        spleeter=use_spleeter,
        spleeter_stems=5,
    )
    cmd_full(full_args)


def cmd_full(args) -> None:
    dataset_path, output_dir, upload_func = _resolve_runtime_paths(args)
    input_path = _resolve_input_path(args.input, dataset_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    model = (
        _parse_enum(args.transcribe_model, TranscribeModelEnum)
        or TranscribeModelEnum.FASTER_WHISPER
    )
    if model not in (TranscribeModelEnum.FASTER_WHISPER, TranscribeModelEnum.WHISPER_API):
        raise ValueError("Only FASTER_WHISPER and WHISPER_API are supported.")
    faster_model = _parse_enum(args.faster_whisper_model, FasterWhisperModelEnum)
    vad_method = _parse_enum(args.vad_method, VadMethodEnum)
    language = _parse_language(args.language) if args.language else ""

    config = TranscribeConfig(
        transcribe_model=model,
        transcribe_language=language,
        need_word_time_stamp=args.word_timestamp or args.need_split,
        output_format=TranscribeOutputFormatEnum.SRT,
        whisper_api_key=args.whisper_api_key,
        whisper_api_base=args.whisper_api_base,
        whisper_api_model=args.whisper_api_model,
        whisper_api_prompt=args.whisper_api_prompt,
        faster_whisper_program=args.faster_whisper_program,
        faster_whisper_model=faster_model,
        faster_whisper_model_dir=args.faster_whisper_model_dir or str(MODEL_PATH),
        faster_whisper_device=args.faster_whisper_device,
        faster_whisper_vad_filter=not args.disable_vad,
        faster_whisper_vad_threshold=args.vad_threshold,
        faster_whisper_vad_method=vad_method,
        faster_whisper_ff_mdx_kim2=args.ff_mdx_kim2,
        faster_whisper_one_word=args.one_word,
        faster_whisper_prompt=args.faster_whisper_prompt,
    )

    asr_data = _transcribe_to_asr(
        input_path,
        config,
        use_spleeter=args.spleeter,
        spleeter_stems=args.spleeter_stems,
        audio_track_index=args.audio_track_index,
    )

    output_suffix = args.output_suffix or ".srt"
    output_path = _build_output_path(output_dir, input_path, output_suffix, args.output_name)
    temp_path = output_dir / f".{input_path.stem}.raw.srt"
    asr_data.save(str(temp_path), layout=SubtitleLayoutEnum.ORIGINAL_ON_TOP)

    subtitle_config = _build_subtitle_config(args)
    llm_base_url = llm_api_key = llm_model = None
    if subtitle_config.need_split or subtitle_config.need_optimize or (
        subtitle_config.need_translate
        and subtitle_config.translator_service == TranslatorServiceEnum.OPENAI
    ):
        llm_base_url, llm_api_key, llm_model, _ = _resolve_llm_config(args)

    _process_subtitles(
        temp_path,
        output_path,
        subtitle_config,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        skip_llm_check=args.skip_llm_check,
    )
    temp_path.unlink(missing_ok=True)
    logger.info("Subtitle saved: %s", output_path)
    if upload_func:
        logger.info("Uploading output to OpenI...")
        upload_func()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="videocaptioner-cli")
    subparsers = parser.add_subparsers(dest="command")

    transcribe_parser = subparsers.add_parser("transcribe")
    transcribe_parser.add_argument("--input", required=True)
    transcribe_parser.add_argument("--output-dir", default="")
    transcribe_parser.add_argument("--output-name", default="")
    transcribe_parser.add_argument("--language", default="")
    transcribe_parser.add_argument("--output-format", default="SRT")
    transcribe_parser.add_argument("--subtitle-style", default="")
    transcribe_parser.add_argument("--transcribe-model", default="FASTER_WHISPER")
    transcribe_parser.add_argument("--word-timestamp", action="store_true")
    transcribe_parser.add_argument("--audio-track-index", type=int, default=0)

    transcribe_parser.add_argument("--whisper-api-key", default="")
    transcribe_parser.add_argument("--whisper-api-base", default="")
    transcribe_parser.add_argument("--whisper-api-model", default="")
    transcribe_parser.add_argument("--whisper-api-prompt", default="")

    transcribe_parser.add_argument("--faster-whisper-program", default="")
    transcribe_parser.add_argument("--faster-whisper-model", default="LARGE_V3")
    transcribe_parser.add_argument("--faster-whisper-model-dir", default="")
    transcribe_parser.add_argument("--faster-whisper-device", default="cuda")
    transcribe_parser.add_argument("--disable-vad", action="store_true")
    transcribe_parser.add_argument("--vad-threshold", type=float, default=0.5)
    transcribe_parser.add_argument("--vad-method", default="SILERO_V4")
    transcribe_parser.add_argument("--ff-mdx-kim2", action="store_true")
    transcribe_parser.add_argument("--one-word", action="store_true")
    transcribe_parser.add_argument("--faster-whisper-prompt", default="")

    transcribe_parser.add_argument("--spleeter", action="store_true")
    transcribe_parser.add_argument("--spleeter-stems", type=int, default=5)
    _add_c2net_args(transcribe_parser)
    transcribe_parser.set_defaults(func=cmd_transcribe)

    subtitle_parser = subparsers.add_parser("subtitle")
    subtitle_parser.add_argument("--input", required=True)
    subtitle_parser.add_argument("--output-dir", default="")
    subtitle_parser.add_argument("--output-name", default="")
    subtitle_parser.add_argument("--output-suffix", default=".srt")
    _add_common_subtitle_args(subtitle_parser)
    _add_common_llm_args(subtitle_parser)
    _add_c2net_args(subtitle_parser)
    subtitle_parser.set_defaults(func=cmd_subtitle)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--input", required=True)
    split_parser.add_argument("--output-dir", default="")
    split_parser.add_argument("--output-name", default="")
    split_parser.add_argument("--output-suffix", default=".srt")
    _add_common_subtitle_args(split_parser)
    _add_common_llm_args(split_parser)
    _add_c2net_args(split_parser)
    split_parser.set_defaults(func=cmd_split)

    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument("--input", required=True)
    optimize_parser.add_argument("--output-dir", default="")
    optimize_parser.add_argument("--output-name", default="")
    optimize_parser.add_argument("--output-suffix", default=".srt")
    _add_common_subtitle_args(optimize_parser)
    _add_common_llm_args(optimize_parser)
    _add_c2net_args(optimize_parser)
    optimize_parser.set_defaults(func=cmd_optimize)

    translate_parser = subparsers.add_parser("translate")
    translate_parser.add_argument("--input", required=True)
    translate_parser.add_argument("--output-dir", default="")
    translate_parser.add_argument("--output-name", default="")
    translate_parser.add_argument("--output-suffix", default=".srt")
    _add_common_subtitle_args(translate_parser)
    _add_common_llm_args(translate_parser)
    _add_c2net_args(translate_parser)
    translate_parser.set_defaults(func=cmd_translate)

    interactive_parser = subparsers.add_parser("interactive")
    _add_c2net_args(interactive_parser)
    interactive_parser.set_defaults(func=cmd_interactive)

    full_parser = subparsers.add_parser("full")
    full_parser.add_argument("--input", required=True)
    full_parser.add_argument("--output-dir", default="")
    full_parser.add_argument("--output-name", default="")
    full_parser.add_argument("--output-suffix", default=".srt")
    full_parser.add_argument("--language", default="")
    full_parser.add_argument("--transcribe-model", default="FASTER_WHISPER")
    full_parser.add_argument("--word-timestamp", action="store_true")
    full_parser.add_argument("--audio-track-index", type=int, default=0)

    full_parser.add_argument("--whisper-api-key", default="")
    full_parser.add_argument("--whisper-api-base", default="")
    full_parser.add_argument("--whisper-api-model", default="")
    full_parser.add_argument("--whisper-api-prompt", default="")

    full_parser.add_argument("--faster-whisper-program", default="")
    full_parser.add_argument("--faster-whisper-model", default="LARGE_V3")
    full_parser.add_argument("--faster-whisper-model-dir", default="")
    full_parser.add_argument("--faster-whisper-device", default="cuda")
    full_parser.add_argument("--disable-vad", action="store_true")
    full_parser.add_argument("--vad-threshold", type=float, default=0.5)
    full_parser.add_argument("--vad-method", default="SILERO_V4")
    full_parser.add_argument("--ff-mdx-kim2", action="store_true")
    full_parser.add_argument("--one-word", action="store_true")
    full_parser.add_argument("--faster-whisper-prompt", default="")
    full_parser.add_argument("--spleeter", action="store_true")
    full_parser.add_argument("--spleeter-stems", type=int, default=5)

    _add_common_subtitle_args(full_parser)
    _add_common_llm_args(full_parser)
    _add_c2net_args(full_parser)
    full_parser.set_defaults(func=cmd_full)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        args.func(args)
        return 0
    except Exception as exc:
        logger.exception("CLI error: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
