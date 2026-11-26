import os
from dataclasses import dataclass
from pathlib import Path

from transcribe_app.paths import ROOT, configure_cache

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


class DiarizationUnavailable(RuntimeError):
    pass


def diarization_status() -> tuple[bool, str]:
    try:
        _prepare_torchaudio()
        import pyannote.audio  # noqa: F401
    except Exception as exc:
        return False, f"Диаризация недоступна: pyannote не загружается ({exc.__class__.__name__}: {exc})."
    if _token() is None:
        return False, "Диаризация недоступна: нет HF_TOKEN в окружении или в .env."
    return True, ""


def _prepare_torchaudio() -> None:
    """pyannote 3.x expects torchaudio APIs removed in the 2.11 wheel."""
    import torchaudio

    if hasattr(torchaudio, "AudioMetaData") and hasattr(torchaudio, "list_audio_backends"):
        return

    @dataclass
    class AudioMetaData:
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = "PCM_S"

    def info(uri, backend=None):
        import soundfile as sf

        data = sf.info(str(uri))
        return AudioMetaData(
            sample_rate=data.samplerate,
            num_frames=data.frames,
            num_channels=data.channels,
            bits_per_sample=16,
            encoding="PCM_S",
        )

    torchaudio.AudioMetaData = AudioMetaData
    torchaudio.list_audio_backends = lambda: ["soundfile"]
    torchaudio.info = info


def diarize(wav_path: Path) -> list[tuple[float, float, str]]:
    available, reason = diarization_status()
    if not available:
        raise DiarizationUnavailable(reason)
    configure_cache()
    _prepare_torchaudio()
    import torch
    from pyannote.audio import Pipeline

    token = _token()
    try:
        try:
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
    except Exception as exc:
        raise DiarizationUnavailable(
            "Не удалось загрузить pyannote/speaker-diarization-community-1. "
            "Примите условия модели на Hugging Face и проверьте HF_TOKEN. "
            f"{exc}"
        ) from exc
    if pipeline is None:
        raise DiarizationUnavailable(
            "Не удалось загрузить pyannote/speaker-diarization-community-1. "
            "Примите условия модели на Hugging Face и проверьте токен."
        )
    pipeline.to(torch.device("cpu"))
    import soundfile as sf

    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    annotation = getattr(output, "speaker_diarization", output)
    turns: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))
    turns.sort(key=lambda item: (item[0], item[1]))
    return turns


def _token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value.strip()
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, raw = stripped.split("=", 1)
        if name.strip() in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
            token = raw.strip().strip('"').strip("'")
            if token:
                return token
    return None
