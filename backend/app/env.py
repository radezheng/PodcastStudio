from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def _find_env_file() -> Path | None:
    """Find .env within this PodcastStudio project.

    Order:
    1) PodcastStudio/.env
    2) PodcastStudio/code/.env (legacy)
    3) Any parent .env (fallback)
    """

    here = Path(__file__).resolve()
    project_root = here.parents[2]  # PodcastStudio/

    root_env = project_root / ".env"
    if root_env.exists():
        return root_env

    legacy = project_root / "code" / ".env"
    if legacy.exists():
        return legacy

    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate

    return None


def load_project_env() -> Path | None:
    env_path = _find_env_file()
    if env_path is not None:
        load_dotenv(env_path, override=False)
    return env_path
