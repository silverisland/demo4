"""Resolve local parquet inputs consistently for training and evaluation."""

from glob import glob
from pathlib import Path


def expand_paths(source):
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    paths = []
    for item in sources:
        path = Path(item).expanduser()
        matches = (
            sorted(path.glob("*.parquet"))
            if path.is_dir()
            else sorted(Path(p) for p in glob(str(path)))
        )
        matches = [p.resolve() for p in matches if p.is_file()]
        if not matches:
            raise ValueError(f"No parquet files matched: {item}")
        paths.extend(matches)
    if not paths:
        raise ValueError("No parquet inputs supplied")
    return list(dict.fromkeys(str(p) for p in paths))
