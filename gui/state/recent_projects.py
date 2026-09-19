"""Small persisted list of recently opened project folders (app-level
settings, independent of any single project's own metadata file)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings

MAX_RECENT = 8


def _settings() -> QSettings:
    return QSettings("ITFVP", "ITFVP-GUI")


def get_recent_projects() -> list[Path]:
    raw = _settings().value("recent_projects", [])
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    return [p for p in (Path(s) for s in raw) if p.is_dir()]


def add_recent_project(path: Path) -> None:
    settings = _settings()
    current = [str(p) for p in get_recent_projects()]
    path_str = str(path)
    if path_str in current:
        current.remove(path_str)
    current.insert(0, path_str)
    settings.setValue("recent_projects", current[:MAX_RECENT])
