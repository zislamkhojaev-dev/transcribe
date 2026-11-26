import gc
from collections.abc import Callable
from pathlib import Path

from transcribe_app.asr import DEFAULT_REVISION, Transcriber
from transcribe_app.audio import convert_to_wav, temporary_directory, write_chunk
from transcribe_app.diarize import diarize
from transcribe_app.export import render_txt
from transcribe_app.phrases import assign_speakers, group_phrases
from transcribe_app.vad import SpeechSegmenter

Progress = Callable[[str], None]


def transcribe_file(
    source: Path,
    *,
    with_speakers: bool = False,
    with_timestamps: bool = True,
    revision: str = DEFAULT_REVISION,
    device: str | None = None,
    on_progress: Progress | None = None,
) -> str:
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Файл не найден: {source}")

    def report(message: str) -> None:
        if on_progress:
            on_progress(message)

    with temporary_directory() as raw_dir:
        work = Path(raw_dir)
        wav_path = work / "source.wav"
        report("Преобразование аудио")
        convert_to_wav(source, wav_path)

        turns: list[tuple[float, float, str]] = []
        if with_speakers:
            report("Определение говорящих")
            turns = diarize(wav_path)
            gc.collect()

        report("Поиск фрагментов речи")
        segmenter = SpeechSegmenter()
        chunks = segmenter.chunks(wav_path)
        report(f"Фрагментов: {len(chunks)}")

        transcriber = Transcriber(revision=revision, device=device)
        words = []
        try:
            for index, (start, end) in enumerate(chunks, start=1):
                report(f"Распознавание {index}/{len(chunks)}")
                chunk_path = work / f"chunk-{index:04d}.wav"
                write_chunk(wav_path, start, end, chunk_path)
                words.extend(transcriber.transcribe_chunk(chunk_path, offset=start))
        finally:
            transcriber.close()
            del transcriber
            gc.collect()

    phrases = group_phrases(words)
    if with_speakers:
        phrases = assign_speakers(phrases, turns)
    return render_txt(phrases, with_timestamps=with_timestamps)
