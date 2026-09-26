import os
from pathlib import Path

import datasets.config


def configure_hf_cache(path: str | Path | None) -> None:
    """Points the Hugging Face datasets cache to `path` (no-op if empty).

    `datasets` reads `HF_DATASETS_CACHE` once, at import time, which is before a
    Hydra config is available. Its builders however read `datasets.config.HF_DATASETS_CACHE`
    each time a dataset is loaded, so setting it here is enough. The environment
    variable is set as well, for subprocesses / spawned workers that import
    `datasets` afresh.

    Cached datasets are looked up by the path and mtime of the source files, not
    by the cache location, so the cache can be moved as long as the data isn't.
    """
    if not path:
        return
    os.environ["HF_DATASETS_CACHE"] = str(path)
    datasets.config.HF_DATASETS_CACHE = Path(path)
    Path(path).mkdir(parents=True, exist_ok=True)
