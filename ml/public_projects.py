"""Remaps slide paths under `/mnt/projects` to `/mnt/public_projects`.

In the secure environment, the public `/mnt/projects` (where e.g. the
stain-normalized slides live) is mounted at `/mnt/public_projects`. Set
`USE_PUBLIC_PROJECTS=1` there, so that slide paths stored in the tiling
parquets still resolve. The flag is an environment variable (not a Hydra
option) so it reaches every dataset, including dataloader workers, without
plumbing it through all the datamodule configs.
"""

import os
from pathlib import Path


PROJECTS_ROOT = Path("/mnt/projects")
PUBLIC_PROJECTS_ROOT = Path("/mnt/public_projects")


def use_public_projects() -> bool:
    return os.environ.get("USE_PUBLIC_PROJECTS", "0").lower() in {"1", "true"}


def resolve_slide_path(path: str | Path) -> Path:
    path = Path(path)
    if use_public_projects() and path.is_relative_to(PROJECTS_ROOT):
        return PUBLIC_PROJECTS_ROOT / path.relative_to(PROJECTS_ROOT)
    return path
