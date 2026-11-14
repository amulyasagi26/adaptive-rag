"""Runtime configuration helpers for local and Google Colab execution."""

from pathlib import Path


def select_device(preferred: str = "auto") -> str:
    """Return ``cuda`` when available, unless CPU is explicitly requested."""
    if preferred in {"cpu", "cuda"}:
        return preferred
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def artifact_paths(root: str | Path) -> dict[str, Path]:
    """Create and return persistent directories for indexes, caches, and results."""
    root = Path(root)
    paths = {name: root / name for name in ("indexes", "results", "hf_cache")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
