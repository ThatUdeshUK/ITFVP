"""Shared, mutable application state for the GUI's pages, persisted as a
small JSON metadata file in the project directory.

Import copies source TIFFs into the project directory rather than touching
them in place — the raw microscopy data is never modified or moved, per
this repo's convention. All renaming/reordering below acts only on those
project-local copies.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from .. import pipeline

_NUM_RE = re.compile(r"(\d+)")

PROJECT_FILE_NAME = "itfvp_project.json"
SCHEMA_VERSION = 1


def natural_key(path: Path):
    parts = _NUM_RE.split(path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def is_existing_project(path: Path) -> bool:
    return (path / PROJECT_FILE_NAME).exists()


def _unique_dest(dest_dir: Path, name: str) -> Path:
    dest = dest_dir / name
    if not dest.exists():
        return dest
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def _noop_progress(done: int, total: int, message: str) -> None:
    pass


def _copy_files(
    sources: list[Path], dest_dir: Path, progress: pipeline.ProgressFn = _noop_progress,
) -> list[Path]:
    """Pure (no AppState access) so it can safely run on a worker thread —
    the caller applies the resulting paths to AppState back on the main
    thread. See import_files_job / apply_imported_files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = len(sources)
    dests = []
    for i, src in enumerate(sources):
        dest = _unique_dest(dest_dir, src.name)
        shutil.copy2(src, dest)
        dests.append(dest)
        progress(i + 1, n, f"Copied {src.name} ({i + 1}/{n})")
    return dests


def _rename_files(
    files: list[Path], pattern: str, progress: pipeline.ProgressFn = _noop_progress,
) -> list[Path]:
    """Pure — see batch_rename_job / apply_batch_rename."""
    n = len(files)
    # Rename through unique temp names first so a pattern that collides
    # with current names (e.g. re-applying the same pattern) can't clash.
    tmp_paths = []
    for i, f in enumerate(files):
        tmp = f.with_name(f".__renaming_{i}{f.suffix}")
        f.rename(tmp)
        tmp_paths.append(tmp)
        progress(i + 1, n, f"Preparing rename ({i + 1}/{n})")
    new_files = []
    for i, tmp in enumerate(tmp_paths, start=1):
        stem = pattern.format(index=i)
        new_path = tmp.with_name(f"{stem}{tmp.suffix}")
        tmp.rename(new_path)
        new_files.append(new_path)
        progress(i, n, f"Renamed {i}/{n}")
    return new_files


class RenderSettings(QObject):
    """Cosmetic render options shared — and kept in sync — across the PIV,
    traction, and combined tabs' "Render properties" panels. Each panel
    edits this same instance and listens to `changed`, so a change in one
    tab is reflected in the others (see gui/render_options.py)."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self.reset_to_defaults()

    def reset_to_defaults(self) -> None:
        self.background_mode = "white"     # "white" | "black" | "channel"
        self.unit_mode = "px"              # "px" | "um"
        self.scale_um_per_px = 1.0
        self.arrow_width = pipeline.RenderStyle().arrow_width
        self.arrow_length = pipeline.RenderStyle().arrow_length
        self.arrow_color_mode = "rainbow"  # "rainbow" | "solid"
        self.arrow_color = "#000000"

    def to_dict(self) -> dict:
        return {
            "background_mode": self.background_mode,
            "unit_mode": self.unit_mode,
            "scale_um_per_px": self.scale_um_per_px,
            "arrow_width": self.arrow_width,
            "arrow_length": self.arrow_length,
            "arrow_color_mode": self.arrow_color_mode,
            "arrow_color": self.arrow_color,
        }

    def load_dict(self, data: dict) -> None:
        defaults = pipeline.RenderStyle()
        self.background_mode = data.get("background_mode", "white")
        self.unit_mode = data.get("unit_mode", "px")
        self.scale_um_per_px = data.get("scale_um_per_px", 1.0)
        self.arrow_width = data.get("arrow_width", defaults.arrow_width)
        self.arrow_length = data.get("arrow_length", defaults.arrow_length)
        self.arrow_color_mode = data.get("arrow_color_mode", "rainbow")
        self.arrow_color = data.get("arrow_color", "#000000")

    def px_to_um(self) -> float | None:
        return self.scale_um_per_px if self.unit_mode == "um" and self.scale_um_per_px else None

    def snapshot(self, include_unit: bool = True) -> pipeline.RenderStyle:
        """include_unit=False forces px_to_um to None regardless of the
        shared unit setting — used by the traction tab, since traction
        stress is unit-independent of pixel size and doesn't expose its own
        axis-unit control (see gui/render_options.py, show_unit)."""
        return pipeline.RenderStyle(
            background_mode=self.background_mode,
            px_to_um=self.px_to_um() if include_unit else None,
            arrow_width=self.arrow_width,
            arrow_length=self.arrow_length,
            arrow_color_mode=self.arrow_color_mode,
            arrow_color=self.arrow_color,
        )


class AppState(QObject):
    # Emitted after every mutation; pages listen and re-render themselves,
    # and it's how project metadata gets persisted to disk (see _persist).
    changed = Signal()
    # Emitted specifically when a project is opened/switched (in addition
    # to `changed`) — pages use this, not `changed`, to re-populate form
    # fields (PIV preset, window sizes, Young's modulus, ...) from the
    # project's saved options, since re-applying those on *every* `changed`
    # would stomp on an in-progress edit the user hasn't submitted yet.
    project_loaded = Signal()

    def __init__(self):
        super().__init__()
        self.project_dir: Path | None = None
        self._created_at: str | None = None
        # Cosmetic render preferences, shared across PIV/traction/combined
        # tabs (see RenderSettings) — project-scoped like everything else
        # here, reset/restored in _reset_fields/_load.
        self.render_settings = RenderSettings()
        self._reset_fields()
        self.changed.connect(self._persist)

    def _reset_fields(self) -> None:
        self.micro_files: list[Path] = []
        self.bg_files: list[Path] = []
        self.stabilized_files: list[Path] = []
        self.stabilized_bg_files: list[Path] | None = None
        self.piv_result: pipeline.PIVResult | None = None
        self.traction_result: pipeline.TractionResult | None = None
        # key -> {"frames_dir": Path, "video": Path}, key in {"piv", "traction", "combined"}
        self.render_outputs: dict[str, dict[str, Path]] = {}
        # The options actually used to produce piv_result/traction_result
        # (None until computed at least once) — restored into the PIV/
        # Traction tabs' form fields on project_loaded, so reopening a
        # project shows what actually produced its cached results.
        self.piv_meta: dict | None = None
        self.traction_meta: dict | None = None
        self.stabilize_meta: dict | None = None
        # Collapsed/expanded state of each "properties" panel (see
        # gui/collapsible_group.py), keyed e.g. "piv.computation",
        # "traction.render" — per-project, absent key means collapsed.
        self.group_expanded: dict[str, bool] = {}
        self.render_settings.reset_to_defaults()

    # -- project lifecycle -------------------------------------------------

    def project_file(self) -> Path:
        return self.project_dir / PROJECT_FILE_NAME

    def open_project(self, path: Path) -> None:
        """Point the app at path: load its itfvp_project.json if present,
        otherwise start fresh (a brand-new project) and write one."""
        path.mkdir(parents=True, exist_ok=True)
        self.project_dir = path
        meta_path = self.project_file()
        if meta_path.exists():
            self._load(meta_path)
        else:
            self._reset_fields()
            self._created_at = datetime.now(timezone.utc).isoformat()
        self.changed.emit()
        self.project_loaded.emit()

    def _rel(self, path: Path) -> str:
        return str(Path(path).relative_to(self.project_dir))

    def _persist(self) -> None:
        if self.project_dir is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        data = {
            "schema_version": SCHEMA_VERSION,
            "created_at": self._created_at or now,
            "updated_at": now,
            "micro_files": [self._rel(f) for f in self.micro_files],
            "bg_files": [self._rel(f) for f in self.bg_files],
            "stabilized_files": [self._rel(f) for f in self.stabilized_files],
            "stabilized_bg_files": (
                [self._rel(f) for f in self.stabilized_bg_files]
                if self.stabilized_bg_files is not None else None
            ),
            "stabilize": self.stabilize_meta,
            "piv": self.piv_meta,
            "traction": self.traction_meta,
            "render_outputs": {
                key: {"frames_dir": self._rel(info["frames_dir"]), "video": self._rel(info["video"])}
                for key, info in self.render_outputs.items()
            },
            "group_expanded": self.group_expanded,
            "render_settings": self.render_settings.to_dict(),
        }
        tmp = self.project_file().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.project_file())

    def _load(self, meta_path: Path) -> None:
        try:
            data = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            self._reset_fields()
            self._created_at = datetime.now(timezone.utc).isoformat()
            return

        self._created_at = data.get("created_at")
        self.micro_files = [self.project_dir / p for p in data.get("micro_files", [])]
        self.bg_files = [self.project_dir / p for p in data.get("bg_files", [])]
        self.stabilized_files = [self.project_dir / p for p in data.get("stabilized_files", [])]
        sbg = data.get("stabilized_bg_files")
        self.stabilized_bg_files = [self.project_dir / p for p in sbg] if sbg else None

        self.stabilize_meta = data.get("stabilize")

        self.piv_meta = data.get("piv")
        self.piv_result = self._load_piv_fields(self.piv_meta)

        self.traction_meta = data.get("traction")
        self.traction_result = self._load_traction_fields(self.traction_meta)

        self.render_outputs = {}
        for key, info in data.get("render_outputs", {}).items():
            frames_dir = self.project_dir / info["frames_dir"]
            video = self.project_dir / info["video"]
            if frames_dir.exists() and video.exists():
                self.render_outputs[key] = {"frames_dir": frames_dir, "video": video}

        self.group_expanded = dict(data.get("group_expanded", {}))
        self.render_settings.load_dict(data.get("render_settings", {}))

    def _load_piv_fields(self, meta: dict | None) -> pipeline.PIVResult | None:
        if not meta or not meta.get("computed"):
            return None
        fields_path = self.project_dir / meta["fields_path"]
        if not fields_path.exists():
            return None
        cached = np.load(fields_path)
        return pipeline.PIVResult(
            x=cached["x"], y=cached["y"], u=cached["u"], v=cached["v"],
            u_smooth=cached["u_smooth"], v_smooth=cached["v_smooth"],
        )

    def _load_traction_fields(self, meta: dict | None) -> pipeline.TractionResult | None:
        if not meta or not meta.get("computed"):
            return None
        fields_path = self.project_dir / meta["fields_path"]
        if not fields_path.exists():
            return None
        cached = np.load(fields_path)
        return pipeline.TractionResult(
            tx=cached["tx"], ty=cached["ty"],
            tx_smooth=cached["tx_smooth"], ty_smooth=cached["ty_smooth"],
        )

    # -- import ------------------------------------------------------------

    def _channel_dir(self, channel: str) -> Path:
        if self.project_dir is None:
            raise ValueError("Choose a project directory first")
        name = "input_micro" if channel == "micro" else "input_bg"
        d = self.project_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def files_for(self, channel: str) -> list[Path]:
        return self.micro_files if channel == "micro" else self.bg_files

    def import_files(self, channel: str, sources: list[Path]) -> None:
        """Synchronous convenience wrapper. GUI code should prefer
        import_files_job + apply_imported_files so the copy runs off the
        main thread behind a progress dialog (see gui/progress.py)."""
        self.apply_imported_files(channel, _copy_files(sources, self._channel_dir(channel)))

    def import_files_job(self, channel: str, sources: list[Path]):
        dest_dir = self._channel_dir(channel)
        return lambda progress: _copy_files(sources, dest_dir, progress)

    def apply_imported_files(self, channel: str, dest_paths: list[Path]) -> None:
        self.files_for(channel).extend(dest_paths)
        self._invalidate_downstream()
        self.changed.emit()

    def remove_file(self, channel: str, index: int) -> None:
        del self.files_for(channel)[index]
        self._invalidate_downstream()
        self.changed.emit()

    def clear_channel(self, channel: str) -> None:
        self.files_for(channel).clear()
        self._invalidate_downstream()
        self.changed.emit()

    def move_file(self, channel: str, index: int, delta: int) -> None:
        files = self.files_for(channel)
        new_index = index + delta
        if 0 <= new_index < len(files):
            files[index], files[new_index] = files[new_index], files[index]
            self._invalidate_downstream()
            self.changed.emit()

    def rename_file(self, channel: str, index: int, new_stem: str) -> None:
        files = self.files_for(channel)
        old_path = files[index]
        new_path = old_path.with_name(f"{new_stem}{old_path.suffix}")
        if new_path != old_path and new_path.exists():
            raise ValueError(f"{new_path.name} already exists")
        old_path.rename(new_path)
        files[index] = new_path
        self._invalidate_downstream()
        self.changed.emit()

    def batch_rename(self, channel: str, pattern: str) -> None:
        """Synchronous convenience wrapper — pattern may use {index}
        (1-based), e.g. 'Frame_{index:04d}'. GUI code should prefer
        batch_rename_job + apply_batch_rename so it runs off the main
        thread behind a progress dialog (see gui/progress.py)."""
        files = self.files_for(channel)
        if not files:
            return
        self.apply_batch_rename(channel, _rename_files(list(files), pattern))

    def batch_rename_job(self, channel: str, pattern: str):
        files = list(self.files_for(channel))
        return lambda progress: _rename_files(files, pattern, progress)

    def apply_batch_rename(self, channel: str, new_files: list[Path]) -> None:
        self.files_for(channel)[:] = new_files
        self._invalidate_downstream()
        self.changed.emit()

    def _invalidate_downstream(self) -> None:
        self.stabilized_files = []
        self.stabilized_bg_files = None
        self.stabilize_meta = None
        self.piv_result = None
        self.traction_result = None
        self.render_outputs = {}
        self.piv_meta = None
        self.traction_meta = None

    # -- pipeline results --------------------------------------------------

    def set_stabilized(self, files: list[Path], bg_files: list[Path] | None, skipped: bool = False) -> None:
        self.stabilize_meta = {"skipped": skipped}
        self.stabilized_files = files
        self.stabilized_bg_files = bg_files
        self.piv_result = None
        self.traction_result = None
        self.render_outputs = {}
        self.piv_meta = None
        self.traction_meta = None
        self.changed.emit()

    def set_piv_result(
        self, result: pipeline.PIVResult,
        use_gapped_settings: bool, window_sizes: tuple[int, ...] | None,
    ) -> None:
        self.piv_result = result
        self.traction_result = None
        self.render_outputs = {}
        self.traction_meta = None
        self.piv_meta = {"computed": True, "use_gapped_settings": use_gapped_settings,
                           "window_sizes": list(window_sizes) if window_sizes else None}
        if self.project_dir is not None:
            fields_path = self.project_dir / "results" / "piv_fields.npz"
            fields_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(fields_path, x=result.x, y=result.y, u=result.u, v=result.v,
                      u_smooth=result.u_smooth, v_smooth=result.v_smooth)
            self.piv_meta["fields_path"] = self._rel(fields_path)
        self.changed.emit()

    def set_traction_result(
        self, result: pipeline.TractionResult,
        youngs_modulus: float, poisson_ratio: float,
    ) -> None:
        self.traction_result = result
        self.render_outputs = {}
        self.traction_meta = {"computed": True, "youngs_modulus": youngs_modulus,
                                "poisson_ratio": poisson_ratio}
        if self.project_dir is not None:
            fields_path = self.project_dir / "results" / "traction_fields.npz"
            fields_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(fields_path, tx=result.tx, ty=result.ty,
                      tx_smooth=result.tx_smooth, ty_smooth=result.ty_smooth)
            self.traction_meta["fields_path"] = self._rel(fields_path)
        self.changed.emit()

    def set_render_output(self, key: str, frames_dir: Path, video_path: Path) -> None:
        self.render_outputs[key] = {"frames_dir": frames_dir, "video": video_path}
        self.changed.emit()

    # -- UI state ------------------------------------------------------

    def is_group_expanded(self, key: str) -> bool:
        """Collapsed (False) unless this project previously had it expanded."""
        return self.group_expanded.get(key, False)

    def set_group_expanded(self, key: str, expanded: bool) -> None:
        if self.group_expanded.get(key) == expanded:
            return
        self.group_expanded[key] = expanded
        # Persist directly rather than emitting `changed` — this is a UI
        # preference, not data the rest of the app needs to react to, and
        # `changed` would trigger every tab's full refresh() (thumbnail
        # reloads included) just for a collapse/expand click.
        self._persist()
