import subprocess
import tempfile
from pathlib import Path

import soundfile as sf


def convert_to_wav(source: Path, destination: Path) -> None:
    """Convert any ffmpeg-readable media to 16 kHz mono PCM wav."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-vn",
                "-f",
                "wav",
                str(destination),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip()
        tail = detail[-400:] if detail else "нет подробностей"
        raise RuntimeError(f"ffmpeg не смог прочитать файл.\n{tail}") from exc


def write_chunk(source_wav: Path, start: float, end: float, destination: Path) -> None:
    audio, sample_rate = sf.read(source_wav, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    start_sample = max(0, int(round(start * sample_rate)))
    end_sample = min(len(audio), int(round(end * sample_rate)))
    if end_sample <= start_sample:
        raise ValueError(f"Empty audio slice {start:.3f}-{end:.3f}")
    sf.write(destination, audio[start_sample:end_sample], sample_rate)


def duration_seconds(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return float(info.duration)


def temporary_directory() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="transcribe-")
