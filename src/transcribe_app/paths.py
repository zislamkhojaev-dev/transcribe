import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache"
HF_HOME = CACHE / "huggingface"
OUTPUTS = CACHE / "outputs"


def configure_cache() -> Path:
    """Keep model files inside the project and disable Hub telemetry."""
    HF_HOME.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    return HF_HOME
