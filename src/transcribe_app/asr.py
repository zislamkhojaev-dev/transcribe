import os
from pathlib import Path

import torch

from transcribe_app.paths import configure_cache
from transcribe_app.phrases import Word

MODEL_ID = "ai-sage/GigaAM-v3"
DEFAULT_REVISION = "e2e_rnnt"
SAMPLE_RATE = 16000


def resolve_device(requested: str | None = None) -> torch.device:
    choice = requested or os.environ.get("TRANSCRIBE_DEVICE")
    if not choice:
        choice = "mps" if torch.backends.mps.is_available() else "cpu"
    if choice == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS недоступен на этой машине.")
    return torch.device(choice)


class Transcriber:
    def __init__(self, revision: str = DEFAULT_REVISION, device: str | None = None) -> None:
        configure_cache()
        self.revision = revision
        self.device = resolve_device(device)
        self._model = self._load()

    def close(self) -> None:
        self._model = None
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def transcribe_chunk(self, wav_path: Path, offset: float) -> list[Word]:
        asr = self._model.model
        wav, length = asr.prepare_wav(str(wav_path))
        if int(length.item()) > 25 * SAMPLE_RATE:
            raise ValueError("Фрагмент длиннее 25 секунд. Нарезка VAD должна быть короче.")
        text, token_ids, token_frames, seq_len, n_samples = _decode_rnnt(asr, wav, length)
        if not text.strip() or seq_len <= 0:
            return []
        frame_shift = (n_samples / SAMPLE_RATE) / seq_len
        words = _frames_to_words(asr.decoding.tokenizer, token_ids, token_frames, frame_shift)
        if not words and text.strip():
            duration = n_samples / SAMPLE_RATE
            return [Word(text.strip(), offset, offset + duration)]
        return [Word(word.text, word.start + offset, word.end + offset) for word in words]

    def _load(self):
        from transformers import AutoModel

        local_only = _weights_cached(self.revision)
        if local_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
        model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=self.revision,
            trust_remote_code=True,
            local_files_only=local_only,
            use_safetensors=False,
        )
        model.to(self.device)
        model.eval()
        return model


def _weights_cached(revision: str) -> bool:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(MODEL_ID, revision=revision, local_files_only=True)
        return True
    except LocalEntryNotFoundError:
        return False


@torch.inference_mode()
def _decode_rnnt(asr, wav: torch.Tensor, length: torch.Tensor):
    encoded, encoded_len = asr.forward(wav, length)
    decoding = asr.decoding
    head = asr.head
    sequence = encoded.transpose(1, 2)[0]
    seq_len = int(encoded_len[0].item())
    n_samples = int(length[0].item())
    hypothesis: list[int] = []
    token_frames: list[int] = []
    decoder_state = None
    last_label = None
    blank_id = decoding.blank_id
    for frame_index in range(seq_len):
        features = sequence[frame_index].view(1, 1, -1)
        emitted = 0
        while emitted < decoding.max_symbols:
            prediction, hidden = head.decoder.predict(last_label, decoder_state)
            token_id = int(head.joint.joint(features, prediction)[0, 0, 0, :].argmax().item())
            if token_id == blank_id:
                break
            hypothesis.append(token_id)
            token_frames.append(frame_index)
            decoder_state = hidden
            last_label = torch.tensor([[token_id]], device=sequence.device)
            emitted += 1
    text = decoding.tokenizer.decode(hypothesis)
    return text, hypothesis, token_frames, seq_len, n_samples


def _piece(tokenizer, token_id: int) -> str:
    if tokenizer.charwise:
        return tokenizer.vocab[token_id]
    return tokenizer.model.id_to_piece(token_id).replace("▁", " ")


def _frames_to_words(tokenizer, token_ids: list[int], token_frames: list[int], frame_shift: float) -> list[Word]:
    words: list[Word] = []
    chars: list[str] = []
    frames: list[int] = []

    def commit() -> None:
        if not chars:
            return
        text = "".join(chars).strip()
        chars.clear()
        if not text or not frames:
            frames.clear()
            return
        start = frames[0] * frame_shift
        end = (frames[-1] + 1) * frame_shift
        frames.clear()
        words.append(Word(text=text, start=start, end=end))

    for token_id, frame in zip(token_ids, token_frames):
        piece = _piece(tokenizer, token_id)
        if piece == " ":
            commit()
            continue
        if piece.startswith(" "):
            commit()
            piece = piece[1:]
        if not piece:
            continue
        chars.append(piece)
        frames.append(frame)
    commit()
    return words
