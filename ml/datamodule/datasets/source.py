from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict


class SlidesSource(TypedDict, total=False):
    uris: Iterable[str]
    paths: Iterable[str | Path]


def resolve_slides_source(
    uris: Iterable[str] | None,
    paths: Iterable[str | Path] | None,
    use_paths: bool,
) -> SlidesSource:
    """Picks either `uris` or `paths` to load slides/tiles metadata from.

    Unlike the underlying loader (which concatenates both when given
    together), here exactly one source is used: if both are provided,
    `use_paths` decides which one wins.

    Hydra configs always pass `paths` as a list -- e.g.
    `[${oc.select:data.some_path, null}]` -- so a data config without a path
    version resolves to `[None]` rather than `None`. Such placeholder `None`
    entries are filtered out before checking whether any real path was given.
    """
    if paths is not None:
        paths = [p for p in paths if p is not None]
        if not paths:
            paths = None

    if uris is None and paths is None:
        raise ValueError("Either `uris` or `paths` must be provided.")

    if paths is not None and (uris is None or use_paths):
        return {"paths": paths}

    assert uris is not None
    return {"uris": uris}
