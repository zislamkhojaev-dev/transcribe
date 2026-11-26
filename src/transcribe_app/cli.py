import argparse
import sys
from pathlib import Path

from transcribe_app.asr import DEFAULT_REVISION
from transcribe_app.diarize import DiarizationUnavailable
from transcribe_app.pipeline import transcribe_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Офлайн-расшифровка интервью в TXT.")
    parser.add_argument("audio", nargs="?", type=Path, help="Аудио или видео интервью")
    parser.add_argument("-o", "--output", type=Path, help="Куда записать TXT")
    parser.add_argument("--diarize", action="store_true", help="Добавить метки Спикер N")
    parser.add_argument(
        "--timestamps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Писать таймкод перед каждой фразой (по умолчанию включено)",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Ревизия ai-sage/GigaAM-v3")
    parser.add_argument("--device", choices=("cpu", "mps"), default=None)
    parser.add_argument("--ui", action="store_true", help="Открыть локальный веб-интерфейс")
    args = parser.parse_args(argv)

    if args.ui:
        from transcribe_app.app import launch

        launch()
        return

    if args.audio is None:
        parser.error("Укажите файл интервью или запустите с --ui")

    try:
        text = transcribe_file(
            args.audio,
            with_speakers=args.diarize,
            with_timestamps=args.timestamps,
            revision=args.revision,
            device=args.device,
            on_progress=print,
        )
    except DiarizationUnavailable as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    output = args.output or args.audio.with_suffix(".txt")
    output.write_text(text, encoding="utf-8")
    print(output)


def cli() -> None:
    main()
