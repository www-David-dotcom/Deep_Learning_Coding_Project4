from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def prepare_output_path(path: str | Path) -> Path:
    """Resolve an output path and create its parent directory if needed."""

    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
