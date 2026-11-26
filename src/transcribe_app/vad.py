from pathlib import Path

import soundfile as sf
import torch

MAX_CHUNK_SECONDS = 22.0
SAMPLE_RATE = 16000


def pack_segments(
    segments: list[tuple[float, float]],
    max_duration: float = MAX_CHUNK_SECONDS,
) -> list[tuple[float, float]]:
    """Pack speech intervals into non-overlapping chunks no longer than max_duration."""
    cleaned: list[tuple[float, float]] = []
    for start, end in sorted(segments):
        if end <= start:
            continue
        if cleaned and start < cleaned[-1][1]:
            start = cleaned[-1][1]
        if end - start <= 0.05:
            continue
        cleaned.append((start, end))

    if not cleaned:
        return []

    packed: list[tuple[float, float]] = []
    chunk_start, chunk_end = cleaned[0]
    for start, end in cleaned[1:]:
        if end - chunk_start <= max_duration:
            chunk_end = end
            continue
        packed.extend(_split(chunk_start, chunk_end, max_duration))
        chunk_start, chunk_end = start, end
    packed.extend(_split(chunk_start, chunk_end, max_duration))
    return packed


def fixed_windows(duration: float, max_duration: float = MAX_CHUNK_SECONDS) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 0.05:
        end = min(duration, start + max_duration)
        windows.append((start, end))
        start = end
    return windows


def _split(start: float, end: float, max_duration: float) -> list[tuple[float, float]]:
    pieces: list[tuple[float, float]] = []
    cursor = start
    while end - cursor > max_duration:
        pieces.append((cursor, cursor + max_duration))
        cursor += max_duration
    if end - cursor > 0.05:
        pieces.append((cursor, end))
    return pieces


class SpeechSegmenter:
    def __init__(self) -> None:
        from silero_vad import get_speech_timestamps, load_silero_vad

        self._get_speech_timestamps = get_speech_timestamps
        self._model = load_silero_vad(onnx=False)

    def chunks(self, wav_path: Path) -> list[tuple[float, float]]:
        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz wav, got {sample_rate}")
        waveform = torch.from_numpy(audio)
        stamps = self._get_speech_timestamps(
            waveform,
            self._model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            speech_pad_ms=0,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
            max_speech_duration_s=MAX_CHUNK_SECONDS,
        )
        segments = [(float(item["start"]), float(item["end"])) for item in stamps]
        packed = pack_segments(segments)
        if packed:
            return packed
        return fixed_windows(duration=len(audio) / SAMPLE_RATE)
