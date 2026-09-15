from collections.abc import Iterable
from pathlib import Path


def resolve_slides_source(
    uris: Iterable[str] | None,
    paths: Iterable[str | Path] | None,
    use_paths: bool,
) -> dict[str, Iterable[str] | Iterable[str | Path]]:
    """Picks either `uris` or `paths` to load slides/tiles metadata from.

    Unlike the underlying loader (which concatenates both when given
    together), here exactly one source is used: if both are provided,
    `use_paths` decides which one wins.
    """
    if uris is None and paths is None:
        raise ValueError("Either `uris` or `paths` must be provided.")

    if paths is not None and (uris is None or use_paths):
        return {"paths": paths}

    assert uris is not None
    return {"uris": uris}
