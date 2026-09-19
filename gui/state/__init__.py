"""Application state: AppState (the project's persisted data model) and
RenderSettings (its shared cosmetic-render sub-state) — see app_state.py.
recent_projects.py is a separate, small QSettings-backed store (the
app-level list of recently opened project folders), accessed as a
submodule: `from ..state import recent_projects`.
"""

from .app_state import AppState, RenderSettings, is_existing_project, natural_key

__all__ = ["AppState", "RenderSettings", "is_existing_project", "natural_key"]
