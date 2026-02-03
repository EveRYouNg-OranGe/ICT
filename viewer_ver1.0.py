import os
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image, ImageOps, ImageDraw, ImageFont
import math
from pathlib import Path
APP_DIR = Path(__file__).resolve().parent
import re
import shutil
from pathlib import Path


# Try PyQt6, fallback PyQt5
try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPixmap, QImage
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
        QGridLayout, QCheckBox, QComboBox, QPushButton, QMessageBox, QSizePolicy,
        QFileDialog, QInputDialog
    )
    QT6 = True
except ImportError:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap, QImage
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
        QGridLayout, QCheckBox, QComboBox, QPushButton, QMessageBox, QSizePolicy,
        QFileDialog, QInputDialog
    )
    QT6 = False


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def normalize_scene_name(name: str) -> str:
    """
    归一化场景名：去掉 gt/blur 这类前缀/中缀，避免同场景分散
    例：gtcozy2room -> cozy2room
        blur_pool -> pool
        blurwine -> wine
    """
    s = name.strip()

    # 常见：前缀 gt / blur / gt_ / blur- / gtblur 等
    s = re.sub(r'^(?:gt|blur)+[_\-]*', '', s, flags=re.IGNORECASE)

    # 如果还有中间夹着的 _gt_ / _blur_ 之类，也去掉（可选但更稳）
    s = re.sub(r'[_\-]*(?:gt|blur)[_\-]*', '', s, flags=re.IGNORECASE)

    return s

@dataclass(frozen=True)
class ItemKey:
    scene: str
    filename: str


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def scan_root(
    root: Path,
    no_scene_as_default: bool = False,
) -> Tuple[List[str], List[str], Dict[str, Dict[str, Dict[str, Path]]]]:
    """
    Returns:
      models: [model_name...]
      scenes: [scene...]
      index: model -> scene -> filename -> path
    """
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    index: Dict[str, Dict[str, Dict[str, Path]]] = {}
    models: List[str] = []
    scenes_set = set()

    for model_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        model = model_dir.name
        # scenes are subdirs that contain images; if none and enabled, treat root images as a single "default" scene
        scene_dirs = [p for p in model_dir.iterdir() if p.is_dir()]

        # no subdirs => optional default scene
        if not scene_dirs:
            if not no_scene_as_default:
                continue
            files = {}
            for fp in model_dir.iterdir():
                if is_image(fp):
                    files[fp.name] = fp
            if not files:
                continue
            index[model] = {"default": files}
            models.append(model)
            scenes_set.add("default")
            continue

        model_map: Dict[str, Dict[str, Path]] = {}
        has_any = False
        for sdir in scene_dirs:
            scene_raw = sdir.name
            scene = normalize_scene_name(scene_raw)
            files = {}
            for fp in sdir.iterdir():
                if is_image(fp):
                    files[fp.name] = fp
            
            if files:
                # 关键：同一个 canonical scene 可能来自不同 raw 文件夹名（gt/blur）
                if scene not in model_map:
                    model_map[scene] = {}
                model_map[scene].update(files)

                scenes_set.add(scene)
                has_any = True

        if has_any:
            index[model] = model_map
            models.append(model)

    scenes = sorted(scenes_set)
    if not models:
        raise RuntimeError(f"No model folders with images found under: {root}")
    if not scenes:
        raise RuntimeError(f"No scenes with images found under: {root}")

    return models, scenes, index


def scan_saved_groups_root(
    root: Path,
    scene_name: str = "default",
) -> Tuple[List[str], List[str], Dict[str, Dict[str, Dict[str, Path]]]]:
    """\
    查看模式：加载“保存的图组”。

    兼容由本 viewer 的 “Save Group” 产生的目录结构：

      saved_root/
        <group_dir_1>/
          <model>_<orig_filename>.(png/jpg/...)
          ...
        <group_dir_2>/
          ...

    在查看模式下，我们把每个 group_dir 视为一张“图片”（一个 key）。
    - scene 固定为 scene_name（默认 "default"）
    - filename 使用 group_dir 的名字
    - 每个模型的图片路径来自 group_dir 里的文件

    注意：这里用文件名的第一个 '_' 之前作为 model 名。
    因此 *不建议* model 名里包含 '_'（否则会被截断）。
    """
    if not root.exists():
        raise FileNotFoundError(f"Saved root not found: {root}")

    index: Dict[str, Dict[str, Dict[str, Path]]] = {}
    models_set = set()
    groups: List[str] = []

    group_dirs = [p for p in root.iterdir() if p.is_dir()]
    group_dirs.sort(key=lambda p: p.name)

    for gdir in group_dirs:
        # collect images in this group dir
        imgs = [p for p in gdir.iterdir() if is_image(p)]
        if not imgs:
            continue

        group_name = gdir.name
        groups.append(group_name)

        for p in imgs:
            # parse model from '<model>_<rest>'
            stem = p.name
            if "_" not in stem:
                # ignore files not following the naming
                continue
            model, _rest = stem.split("_", 1)
            models_set.add(model)
            if model not in index:
                index[model] = {scene_name: {}}
            if scene_name not in index[model]:
                index[model][scene_name] = {}
            # one key per group
            index[model][scene_name][group_name] = p

    models = sorted(models_set)
    scenes = [scene_name]

    if not models:
        raise RuntimeError(f"No models found under saved groups root: {root}")
    if not groups:
        raise RuntimeError(f"No saved groups found under: {root}")

    return models, scenes, index


def build_keys(
    models: List[str],
    scenes: List[str],
    index: Dict[str, Dict[str, Dict[str, Path]]],
    mode: str = "union",
    reference_model: Optional[str] = None,
    filter_scene: Optional[str] = None,
) -> List[ItemKey]:
    """
    mode:
      - union: union of all filenames across models for each scene
      - intersection: intersection across models for each scene
      - reference: use filenames from reference_model only
    """
    keys: List[ItemKey] = []
    scenes_iter = [filter_scene] if (filter_scene and filter_scene != "__ALL__") else scenes

    for scene in scenes_iter:
        per_model_sets = []
        for m in models:
            files = index.get(m, {}).get(scene, {})
            per_model_sets.append(set(files.keys()))
        if mode == "intersection":
            fnames = set.intersection(*per_model_sets) if per_model_sets else set()
        elif mode == "reference":
            ref = reference_model if reference_model in models else models[0]
            fnames = set(index.get(ref, {}).get(scene, {}).keys())
        else:  # union
            fnames = set.union(*per_model_sets) if per_model_sets else set()

        for fn in sorted(fnames):
            keys.append(ItemKey(scene=scene, filename=fn))

    return keys


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, w, h, 3 * w, QImage.Format.Format_RGB888 if QT6 else QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def make_missing_tile(text: str, size: Tuple[int, int]) -> Image.Image:
    img = Image.new("RGB", size, (30, 30, 30))
    draw = ImageDraw.Draw(img)
    # keep it dependency-free: default font
    msg = f"MISSING\n{text}"
    draw.multiline_text((20, 20), msg, fill=(240, 240, 240), spacing=6)
    return img


class TileLabel(QLabel):
    """A QLabel that can notify on double-click and drag (pan)."""

    def __init__(self, on_double_click=None, on_drag=None, parent=None):
        super().__init__(parent)
        self._on_double_click = on_double_click
        self._on_drag = on_drag
        self._dragging = False
        self._last_xy = None
        self.setMouseTracking(True)

    def mouseDoubleClickEvent(self, e):
        if callable(self._on_double_click):
            self._on_double_click()
        return super().mouseDoubleClickEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._dragging = True
            p = e.pos()
            self._last_xy = (p.x(), p.y())
            e.accept()
            return
        return super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._dragging and self._last_xy is not None:
            p = e.pos()
            x, y = p.x(), p.y()
            lx, ly = self._last_xy
            dx, dy = x - lx, y - ly
            self._last_xy = (x, y)
            if callable(self._on_drag) and (dx != 0 or dy != 0):
                self._on_drag(dx, dy)
            e.accept()
            return
        return super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._dragging = False
            self._last_xy = None
            e.accept()
            return
        return super().mouseReleaseEvent(e)


class CompareViewer(QWidget):
    def __init__(
        self,
        root: Path,
        thumb_h: int = 360,
        grid_cols: int = 3,
        key_mode: str = "union",
        prefer_models: Optional[List[str]] = None,
        default_enabled: Optional[List[str]] = None,
        no_scene_as_default: bool = False,
        view_mode: str = "normal",  # normal | saved
        saved_scene_name: str = "default",
    ):
        super().__init__()
        self.root = root
        self.thumb_h = thumb_h
        self.grid_cols = max(1, grid_cols)
        self.key_mode = key_mode
        self.use_label_bg = False   # 是否使用半透明底

        # zoom: two levels (1x, 2x). double-click any tile toggles and applies to all models.
        self.zoom_levels = [1.0, 2.0]
        self.zoom_idx = 0
        self.zoom_factor = self.zoom_levels[self.zoom_idx]

        # Pan offset for zoomed-in view (in cell pixels). Shared across all tiles/models.
        self.pan_dx = 0
        self.pan_dy = 0

        # Cache current grid so pan/zoom can update pixmaps without rebuilding widgets.
        self._grid_cache = None  # (enabled_models_tuple, cols, rows, cell_w, cell_h)
        self._tile_labels = {}   # model_name -> TileLabel
        self._last_layout = None # (enabled_models, cols, rows, cell_w, cell_h)


        self.view_mode = view_mode
        self.saved_scene_name = saved_scene_name

        if self.view_mode == "saved":
            # In saved view mode, `root` is the folder containing saved groups.
            self.models, self.scenes, self.index = scan_saved_groups_root(
                root,
                scene_name=self.saved_scene_name,
            )
        else:
            self.models, self.scenes, self.index = scan_root(
                root,
                no_scene_as_default=no_scene_as_default,
            )

        # Reorder models if prefer_models provided (others appended)
        if prefer_models:
            ordered = []
            s = set(self.models)
            for m in prefer_models:
                if m in s and m not in ordered:
                    ordered.append(m)
            for m in self.models:
                if m not in ordered:
                    ordered.append(m)
            self.models = ordered

        self.scene_choice = "__ALL__"
        self.reference_model = self.models[0]
        self.keys: List[ItemKey] = build_keys(
            self.models, self.scenes, self.index,
            mode=self.key_mode,
            reference_model=self.reference_model,
            filter_scene=self.scene_choice
        )
        self.pos = 0

        self.enabled: Dict[str, bool] = {}
        for m in self.models:
            self.enabled[m] = True
        if default_enabled:
            ds = set(default_enabled)
            for m in self.models:
                self.enabled[m] = (m in ds)

        self._build_ui()
        self._refresh()

        # ---- Keyboard shortcuts (focus-independent) ----
        try:
            from PyQt6.QtGui import QShortcut, QKeySequence
        except Exception:
            from PyQt5.QtWidgets import QShortcut
            from PyQt5.QtGui import QKeySequence

        self.sc_next = QShortcut(QKeySequence("Right"), self)
        self.sc_prev = QShortcut(QKeySequence("Left"), self)
        self.sc_next.activated.connect(lambda: self._step(+1))
        self.sc_prev.activated.connect(lambda: self._step(-1))

        # 加速：Shift+左右
        self.sc_next_fast = QShortcut(QKeySequence("Shift+Right"), self)
        self.sc_prev_fast = QShortcut(QKeySequence("Shift+Left"), self)
        self.sc_next_fast.activated.connect(lambda: self._step(+10))
        self.sc_prev_fast.activated.connect(lambda: self._step(-10))

        self.sc_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.sc_save.activated.connect(self._save_current_group_dialog)

        self.sc_goto = QShortcut(QKeySequence("Ctrl+G"), self)
        self.sc_goto.activated.connect(self._goto_dialog)

        try:
            from PyQt6.QtCore import QTimer
        except Exception:
            from PyQt5.QtCore import QTimer

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._refresh)


    def _build_ui(self):
        mode_tag = "SAVED" if getattr(self, "view_mode", "normal") == "saved" else "NORMAL"
        self.setWindowTitle(f"Derf Compare Viewer [{mode_tag}] - {self.root}")
        self.setMinimumSize(1100, 700)

        outer = QVBoxLayout(self)

        # Top bar: scene dropdown + key mode + navigation
        top = QHBoxLayout()

        top.addWidget(QLabel("Scene:"))
        self.scene_combo = QComboBox()
        self.scene_combo.addItem("__ALL__")
        for s in self.scenes:
            self.scene_combo.addItem(s)
        self.scene_combo.currentTextChanged.connect(self._on_scene_changed)
        top.addWidget(self.scene_combo)

        top.addSpacing(10)
        top.addWidget(QLabel("KeyMode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["union", "intersection", "reference"])
        self.mode_combo.setCurrentText(self.key_mode)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        top.addWidget(self.mode_combo)

        top.addSpacing(10)
        top.addWidget(QLabel("Reference:"))
        self.ref_combo = QComboBox()
        self.ref_combo.addItems(self.models)
        self.ref_combo.setCurrentText(self.reference_model)
        self.ref_combo.currentTextChanged.connect(self._on_ref_changed)
        top.addWidget(self.ref_combo)

        top.addStretch(1)

        self.prev_btn = QPushButton("← Prev")
        self.next_btn = QPushButton("Next →")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(+1))
        top.addWidget(self.prev_btn)
        top.addWidget(self.next_btn)

        self.save_btn = QPushButton("Save Group")
        self.save_btn.clicked.connect(self._save_current_group_dialog)
        top.addWidget(self.save_btn)

        self.goto_btn = QPushButton("Go…")
        self.goto_btn.clicked.connect(self._goto_dialog)
        top.addWidget(self.goto_btn)


        outer.addLayout(top)

        # Model toggles
        toggles = QHBoxLayout()
        toggles.addWidget(QLabel("Models:"))
        self.checkboxes: Dict[str, QCheckBox] = {}
        for m in self.models:
            cb = QCheckBox(m)
            cb.setChecked(self.enabled[m])
            cb.stateChanged.connect(self._on_toggle_changed)
            self.checkboxes[m] = cb
            toggles.addWidget(cb)
        toggles.addStretch(1)
        outer.addLayout(toggles)
        
        # title bg color
        self.bg_checkbox = QCheckBox("Label BG")
        self.bg_checkbox.setChecked(False)
        self.bg_checkbox.stateChanged.connect(self._on_bg_toggle)
        toggles.addWidget(self.bg_checkbox)


        # Info line
        self.info = QLabel("")
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse if QT6 else Qt.TextSelectableByMouse)
        outer.addWidget(self.info)

        # Scroll area with grid
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(2)
        self.grid.setContentsMargins(2, 2, 2, 2)
        self.scroll.setWidget(self.grid_host)
        outer.addWidget(self.scroll, stretch=1)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # --- Order controls (simple) ---
        order_bar = QHBoxLayout()
        order_bar.addWidget(QLabel("Order:"))

        self.order_combo = QComboBox()
        self.order_combo.addItems(self.models)
        order_bar.addWidget(self.order_combo)

        self.up_btn = QPushButton("↑")
        self.down_btn = QPushButton("↓")
        self.up_btn.clicked.connect(lambda: self._move_model(-1))
        self.down_btn.clicked.connect(lambda: self._move_model(+1))
        order_bar.addWidget(self.up_btn)
        order_bar.addWidget(self.down_btn)

        order_bar.addStretch(1)
        outer.addLayout(order_bar)

    def _rebuild_keys(self):
        self.keys = build_keys(
            self.models, self.scenes, self.index,
            mode=self.key_mode,
            reference_model=self.reference_model,
            filter_scene=self.scene_choice
        )
        self.pos = min(self.pos, max(0, len(self.keys) - 1))

    def _on_scene_changed(self, val: str):
        self.scene_choice = val
        self._rebuild_keys()
        self._refresh()

    def _on_mode_changed(self, val: str):
        self.key_mode = val
        self._rebuild_keys()
        self._refresh()

    def _on_ref_changed(self, val: str):
        self.reference_model = val
        if self.key_mode == "reference":
            self._rebuild_keys()
            self._refresh()

    def _on_toggle_changed(self, _):
        for m, cb in self.checkboxes.items():
            self.enabled[m] = cb.isChecked()
        self._refresh()

    def _on_bg_toggle(self, _):
        self.use_label_bg = self.bg_checkbox.isChecked()
        self._refresh()

    def _step(self, delta: int):
        if not self.keys:
            return
        self.pos = max(0, min(len(self.keys) - 1, self.pos + delta))
        self._refresh()

    def _goto_dialog(self):
        """Jump to a specific image by index (1-based) or by filename substring."""
        if not self.keys:
            return

        cur = self.keys[self.pos]
        default_text = str(self.pos + 1)
        text, ok = QInputDialog.getText(
            self,
            "Go to",
            "Enter index (1-based) or filename substring:\n"
            f"Current: [{self.pos+1}/{len(self.keys)}] {cur.scene}/{cur.filename}",
            text=default_text,
        )
        if not ok:
            return

        q = (text or "").strip()
        if not q:
            return

        # 1) numeric => absolute index
        if q.isdigit():
            idx = int(q) - 1
            if 0 <= idx < len(self.keys):
                self.pos = idx
                self._refresh()
            else:
                QMessageBox.information(self, "Go to", f"Index out of range: {q}")
            return

        # 2) search by substring in filename (case-insensitive), within current key list
        q_low = q.lower()
        for i, k in enumerate(self.keys):
            if q_low in k.filename.lower():
                self.pos = i
                self._refresh()
                return

        QMessageBox.information(self, "Go to", f"No match for: {q}")


    def _ensure_grid_layout(self, enabled_models: List[str], cols: int, rows: int, cell_w: int, cell_h: int):
        """Create/reuse TileLabel widgets so pan/zoom can update pixmaps smoothly."""
        cache = (tuple(enabled_models), int(cols), int(rows), int(cell_w), int(cell_h))
        if self._grid_cache == cache:
            return

        # Rebuild grid/widgets
        self._clear_grid()
        self._tile_labels = {}
        self._grid_cache = cache
        self._last_layout = cache

        row = col = 0
        for m in enabled_models:
            lab = TileLabel(on_double_click=self._toggle_zoom, on_drag=self._pan_by)
            lab.setFixedSize(int(cell_w), int(cell_h))
            lab.setScaledContents(True)
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter if QT6 else Qt.AlignCenter)
            lab.setMinimumSize(cell_w, cell_h)

            self._tile_labels[m] = lab
            self.grid.addWidget(lab, row, col)

            col += 1
            if col >= cols:
                col = 0
                row += 1

    def _update_tiles_pixmap(self, key: ItemKey, enabled_models: List[str], cell_w: int, cell_h: int):
        """Update pixmaps for existing labels without rebuilding widgets."""
        for m in enabled_models:
            lab = self._tile_labels.get(m)
            if lab is None:
                continue
            path = self.index.get(m, {}).get(key.scene, {}).get(key.filename, None)
            pix = self._load_tile(path, m, cell_w, cell_h, self.zoom_factor, self.pan_dx, self.pan_dy)
            lab.setPixmap(pix)

    def _toggle_zoom(self):
        """Toggle between two zoom levels (1x, 2x). Applies to all tiles."""
        self.zoom_idx = (self.zoom_idx + 1) % len(self.zoom_levels)
        self.zoom_factor = self.zoom_levels[self.zoom_idx]

        # Reset pan whenever zoom level changes (keeps behavior predictable).
        self.pan_dx = 0
        self.pan_dy = 0

        # Update existing tiles without rebuilding the grid (keeps dragging smooth).
        if self._last_layout is not None and self.keys:
            enabled_models, cols, rows, cell_w, cell_h = self._last_layout
            key = self.keys[self.pos]
            self._update_tiles_pixmap(key, list(enabled_models), cell_w, cell_h)
        else:
            self._refresh()

    def _pan_by(self, dx: int, dy: int):
        """Pan the zoomed-in crop window. Shared across all tiles/models."""
        if self.zoom_factor <= 1.0:
            return
        self.pan_dx += int(dx)
        self.pan_dy += int(dy)

        # Update pixmaps only; do NOT rebuild widgets, otherwise drag will stop after the first move.
        if self._last_layout is not None and self.keys:
            enabled_models, cols, rows, cell_w, cell_h = self._last_layout
            key = self.keys[self.pos]
            self._update_tiles_pixmap(key, list(enabled_models), cell_w, cell_h)
        else:
            self._refresh()

    def keyPressEvent(self, e):
        key = e.key()
        # Shift accelerates
        step = 10 if (e.modifiers() & (Qt.KeyboardModifier.ShiftModifier if QT6 else Qt.ShiftModifier)) else 1
        if key in (Qt.Key.Key_Right if QT6 else Qt.Key_Right, Qt.Key.Key_D if QT6 else Qt.Key_D):
            self._step(+step)
        elif key in (Qt.Key.Key_Left if QT6 else Qt.Key_Left, Qt.Key.Key_A if QT6 else Qt.Key_A):
            self._step(-step)
        else:
            super().keyPressEvent(e)

    def _clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _load_tile(self, path: Optional[Path], title: str, cell_w: int, cell_h: int, zoom_factor: float = 1.0, pan_dx: int = 0, pan_dy: int = 0) -> QPixmap:
        cell_w = max(80, int(cell_w))
        cell_h = max(80, int(cell_h))

        if path is None or (not path.exists()):
            img = make_missing_tile(title, (cell_w, cell_h))
        else:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img).convert("RGB")

            z = 1.0 if (zoom_factor is None) else float(zoom_factor)
            if z <= 1.0:
                # 1x：等比例缩放到能塞进 cell（不裁剪）+ padding 到刚好 cell 大小
                fitted = ImageOps.contain(img, (cell_w, cell_h), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (cell_w, cell_h), (0, 0, 0))
                x = (cell_w - fitted.size[0]) // 2
                y = (cell_h - fitted.size[1]) // 2
                canvas.paste(fitted, (x, y))
                img = canvas
            else:
                # 2x：在“每个小窗(cell)”内部做放大：先做 cover 到 (cell*z)，再裁中心回 (cell)
                tw = max(1, int(round(cell_w * z)))
                th = max(1, int(round(cell_h * z)))
                zoomed = ImageOps.fit(img, (tw, th), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
                left0 = (zoomed.size[0] - cell_w) // 2
                top0 = (zoomed.size[1] - cell_h) // 2

                # Apply shared pan (in cell pixels), clamp to valid crop range
                left = left0 + int(pan_dx)
                top = top0 + int(pan_dy)

                max_left = max(0, zoomed.size[0] - cell_w)
                max_top = max(0, zoomed.size[1] - cell_h)
                left = max(0, min(left, max_left))
                top = max(0, min(top, max_top))

                img = zoomed.crop((left, top, left + cell_w, top + cell_h))

            # 直接把 title 叠在图上（不额外占用高度）
            draw = ImageDraw.Draw(img)

            # 更薄的条（少占空间）
            bar_h = max(16, cell_h // 28)

            # =======================
            # 模型名叠加（大字体 + 可选半透明底）
            # =======================
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            draw = ImageDraw.Draw(img)

            # 字体大小：随 cell 高度自适应（关键）
            # 大幅提高：2x2 / 3x3 时非常明显
            font_size = 20
            try:
                # Linux / WSL 通常有
                font_path = APP_DIR / "unispace bd.ttf"
                font = ImageFont.truetype(str(font_path), font_size)
            except Exception:
                print("font load failed!")
                font = ImageFont.load_default()

            text = title
            text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]

            pad_x = int(font_size * 0.5)
            pad_y = int(font_size * 0.3)
            bar_h = text_h + pad_y * 2

            # 可选：半透明背景
            if self.use_label_bg:
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                odraw = ImageDraw.Draw(overlay)
                odraw.rectangle(
                    [0, 0, cell_w, bar_h],
                    fill=(0, 0, 0, 140)   # 半透明黑底
                )
                img = Image.alpha_composite(img, overlay)
                draw = ImageDraw.Draw(img)

            # 亮绿色文字（更亮一点）
            GREEN = (0, 255, 120, 255)

            # 黑色描边（保证任何背景都清楚）
            x, y = pad_x, pad_y
            for dx, dy in [(-0.75,0),(0.75,0),(0,-0.75),(0,0.75),(-0.75,-0.75),(0.75,-0.75),(-0.75,0.75),(0.75,0.75)]:
                draw.text((x+dx, y+dy), text, font=font, fill=(0, 0, 0, 255))

            draw.text((x, y), text, font=font, fill=GREEN)

            img = img.convert("RGB")


        return pil_to_qpixmap(img)
    
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._resize_timer.start(60)  # 60ms 合并一次刷新

    def _refresh(self):
        # Full refresh is used for navigation / enabling / resizing, etc.
        if not self.keys:
            self._clear_grid()
            self.info.setText("No items found (try KeyMode=union or choose another scene).")
            self._grid_cache = None
            self._tile_labels = {}
            self._last_layout = None
            return

        key = self.keys[self.pos]
        enabled_models = [m for m in self.models if self.enabled.get(m, False)]
        if not enabled_models:
            self._clear_grid()
            self.info.setText(f"[{self.pos+1}/{len(self.keys)}] scene={key.scene} file={key.filename}  (No model enabled)")
            self._grid_cache = None
            self._tile_labels = {}
            self._last_layout = None
            return

        self.info.setText(
            f"[{self.pos+1}/{len(self.keys)}]  scene={key.scene}   file={key.filename}   zoom={self.zoom_factor:g}x"
        )

        # Compute layout (AUTO GRID)
        viewport = self.scroll.viewport().size()
        avail_w = max(1, viewport.width() - 20)
        avail_h = max(1, viewport.height() - 20)

        n = len(enabled_models)
        cols, rows = self._pick_grid(n, avail_w, avail_h)
        cell_w = max(80, avail_w // cols)
        cell_h = max(80, avail_h // rows)

        # Ensure grid widgets exist (rebuild only if layout/models changed)
        self._ensure_grid_layout(enabled_models, cols, rows, cell_w, cell_h)
        self._last_layout = (tuple(enabled_models), cols, rows, cell_w, cell_h)

        # Update pixmaps for current key
        self._update_tiles_pixmap(key, enabled_models, cell_w, cell_h)

    def _rebuild_order_combo(self, keep: Optional[str] = None):
        cur = keep or self.order_combo.currentText()
        self.order_combo.blockSignals(True)
        self.order_combo.clear()
        self.order_combo.addItems(self.models)
        # 尽量保持之前选中项
        if cur in self.models:
            self.order_combo.setCurrentText(cur)
        self.order_combo.blockSignals(False)

    def _move_model(self, delta: int):
        m = self.order_combo.currentText()
        if not m or m not in self.models:
            return
        i = self.models.index(m)
        j = i + delta
        if j < 0 or j >= len(self.models):
            return

        # swap in models order
        self.models[i], self.models[j] = self.models[j], self.models[i]

        # 同步 UI
        self._rebuild_order_combo(keep=m)
        self._refresh()
    
    def _pick_grid(self, n: int, avail_w: int, avail_h: int) -> tuple[int, int]:
        """
        更聪明：
        - 目标是让单元格尽量大（基础项）
        - 惩罚空格（unused）
        - 惩罚极端长条布局（skinny）
        - 惩罚与窗口宽高比不匹配（ratio）
        """
        if n <= 0:
            return 1, 1

        # 允许的列数范围：避免 n 很大时无脑遍历
        max_cols = min(n, 8)  # 你也可以设成 10/12
        target_ratio = avail_w / max(1, avail_h)

        best_cols, best_rows = 1, n
        best_score = -1e30

        for cols in range(1, max_cols + 1):
            rows = math.ceil(n / cols)

            cell_w = avail_w / cols
            cell_h = avail_h / rows
            cell_area = cell_w * cell_h

            unused = cols * rows - n

            grid_ratio = cols / rows
            ratio_penalty = abs(math.log((grid_ratio + 1e-9) / (target_ratio + 1e-9)))  # 0最好

            skinny_penalty = (max(cols, rows) / max(1, min(cols, rows)))  # 越接近1越好

            # 评分：面积是主项，其它是惩罚项
            # 这些权重你可以按体验微调
            score = (
                cell_area
                - unused * (cell_area * 0.08)            # 空格惩罚（按面积比例）
                - ratio_penalty * (cell_area * 0.12)     # 比例不匹配惩罚
                - (skinny_penalty - 1.0) * (cell_area * 0.18)  # 极端长条惩罚
            )

            # 额外硬规则：尽量避免 1×N / N×1（除非 n<=2）
            if n >= 3 and (cols == 1 or rows == 1):
                score -= cell_area * 0.35

            if score > best_score:
                best_score = score
                best_cols, best_rows = cols, rows

        return best_cols, best_rows

    def _save_current_group_dialog(self):
        if not self.keys:
            QMessageBox.information(self, "Save Group", "No items to save.")
            return

        # 选择输出目录
        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not out_dir:
            return
        out_dir = Path(out_dir)

        key = self.keys[self.pos]
        scene = key.scene
        img_name = key.filename
        img_stem = Path(img_name).stem

        # 子文件夹：场景_图片序号（这里用 stem 当“序号”）
        subdir = out_dir / f"{scene}_{img_stem}"
        subdir.mkdir(parents=True, exist_ok=True)

        saved = 0
        missing = 0

        for m in self.models:  # 所有模型，不管是否显示
            # 如果你之前实现了 _find_by_basename，就用它更稳：
            if hasattr(self, "_find_by_basename"):
                src = self._find_by_basename(m, scene, img_name)
            else:
                src = self.index.get(m, {}).get(scene, {}).get(img_name, None)

            if not src or (not Path(src).exists()):
                missing += 1
                continue

            src = Path(src)
            dst = subdir / f"{m}_{img_name}"
            try:
                shutil.copy2(src, dst)
                saved += 1
            except Exception:
                missing += 1

        QMessageBox.information(
            self,
            "Save Group",
            f"Saved: {saved}\nMissing/Failed: {missing}\nFolder:\n{subdir}"
        )


def main():
    ap = argparse.ArgumentParser(description="Compare same-image outputs across multiple model folders.")
    ap.add_argument("root", type=str, help="Root folder, e.g. /mnt/f/Learn/3_TestIMG/Derf对比")
    ap.add_argument("--thumb_h", type=int, default=360, help="Thumbnail height per tile")
    ap.add_argument("--cols", type=int, default=3, help="Grid columns")
    ap.add_argument("--key_mode", choices=["union", "intersection", "reference"], default="union")
    ap.add_argument("--prefer", type=str, default="", help="Comma-separated model order preference")
    ap.add_argument("--default_on", type=str, default="", help="Comma-separated models enabled by default")
    ap.add_argument(
        "--no_scene_as_default",
        action="store_true",
        help="If a model folder has no scene subfolders, treat images in root as a single 'default' scene."
    )
    ap.add_argument(
        "--view_mode",
        choices=["normal", "saved"],
        default="normal",
        help="Viewer mode. 'normal': scan model/scene folders. 'saved': open a folder of saved groups (from 'Save Group')."
    )
    ap.add_argument(
        "--saved_scene_name",
        type=str,
        default="default",
        help="Scene name used in --view_mode saved (usually keep as 'default')."
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    prefer = [x.strip() for x in args.prefer.split(",") if x.strip()]
    default_on = [x.strip() for x in args.default_on.split(",") if x.strip()]

    app = QApplication(sys.argv)
    w = CompareViewer(
        root=root,
        thumb_h=args.thumb_h,
        grid_cols=args.cols,
        key_mode=args.key_mode,
        prefer_models=prefer if prefer else None,
        default_enabled=default_on if default_on else None,
        no_scene_as_default=args.no_scene_as_default,
        view_mode=args.view_mode,
        saved_scene_name=args.saved_scene_name,
    )
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
