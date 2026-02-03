# visual_one_batch_v2.py
# ROI crop grid compare viewer (PyQt6/PyQt5 + Pillow)
# - Load all images from a folder (same size expected)
# - Left: pick one image + interactive ROI rectangle (move/resize, free aspect)
# - Right: show ROI crops from selected/visible images in adjustable grid (rows/cols) with zoom factor
# - Save: exports (1) ROI crop (zoomed) for each image, (2) left full image with ROI overlay
#
# deps: pip install PyQt6 pillow
# fallback: pip install PyQt5 pillow

import sys
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image, ImageOps, ImageDraw

# Try PyQt6, fallback PyQt5
try:
    from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer
    from PyQt6.QtGui import QPixmap, QImage, QPainter, QBrush, QPen, QKeySequence, QShortcut
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
        QLabel, QComboBox, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox,
        QScrollArea, QGroupBox, QFormLayout, QMessageBox,
        QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsRectItem,
        QListWidget, QListWidgetItem, QCheckBox
    )
    QT6 = True
except ImportError:
    from PyQt5.QtCore import Qt, QRectF, QPointF, QTimer
    from PyQt5.QtGui import QPixmap, QImage, QPainter, QBrush, QPen, QKeySequence
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
        QLabel, QComboBox, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox,
        QScrollArea, QGroupBox, QFormLayout, QMessageBox,
        QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsRectItem,
        QListWidget, QListWidgetItem, QShortcut, QCheckBox
    )
    QT6 = False


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    data = img.tobytes("raw", "RGB")
    if QT6:
        qimg = QImage(data, w, h, 3 * w, QImage.Format.Format_RGB888)
    else:
        qimg = QImage(data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class ROIItem(QGraphicsRectItem):
    """
    A resizable/movable ROI rectangle with 8 handles.
    Free aspect ratio.
    Calls on_changed() on move/resize.
    """
    HANDLE_SIZE = 10.0
    TL, TM, TR, ML, MR, BL, BM, BR = range(8)

    def __init__(self, rect: QRectF, bounds_rect: QRectF, on_changed=None):
        super().__init__(rect)
        self.setZValue(10)
        self.bounds_rect = bounds_rect
        self.on_changed = on_changed

        # style
        self.setPen(Qt.GlobalColor.green if QT6 else Qt.green)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
        )

        self._drag_mode = None  # "move" or handle index
        self._press_pos_scene = QPointF()
        self._press_rect = QRectF()

        self.lock_aspect = False
        self.aspect = 1.0  # w/h

        # handles as child items
        self.handles: List[QGraphicsRectItem] = []
        for _ in range(8):
            h = QGraphicsRectItem(self)
            h.setBrush(Qt.GlobalColor.green if QT6 else Qt.green)
            h.setPen(QPen(Qt.PenStyle.NoPen))
            h.setZValue(11)
            self.handles.append(h)
        self._update_handles()

        self.setAcceptHoverEvents(True)

    def rect_in_bounds(self, r: QRectF) -> QRectF:
        min_w, min_h = 5.0, 5.0
        if r.width() < min_w:
            r.setWidth(min_w)
        if r.height() < min_h:
            r.setHeight(min_h)

        bx, by, bw, bh = self.bounds_rect.x(), self.bounds_rect.y(), self.bounds_rect.width(), self.bounds_rect.height()
        x = clamp(r.x(), bx, bx + bw - r.width())
        y = clamp(r.y(), by, by + bh - r.height())
        r.moveTo(x, y)

        max_w = bx + bw - r.x()
        max_h = by + bh - r.y()
        r.setWidth(clamp(r.width(), min_w, max_w))
        r.setHeight(clamp(r.height(), min_h, max_h))
        return r

    def _update_handles(self):
        r = self.rect()
        s = self.HANDLE_SIZE
        pts = [
            QPointF(r.left(),  r.top()),
            QPointF(r.center().x(), r.top()),
            QPointF(r.right(), r.top()),
            QPointF(r.left(),  r.center().y()),
            QPointF(r.right(), r.center().y()),
            QPointF(r.left(),  r.bottom()),
            QPointF(r.center().x(), r.bottom()),
            QPointF(r.right(), r.bottom()),
        ]
        for i, p in enumerate(pts):
            self.handles[i].setRect(p.x() - s/2, p.y() - s/2, s, s)

    def _hit_test_handle(self, pos: QPointF) -> Optional[int]:
        for i, h in enumerate(self.handles):
            if h.rect().contains(pos):
                return i
        return None

    def set_aspect_lock(self, locked: bool, aspect: float):
        self.lock_aspect = bool(locked)
        if aspect and aspect > 0:
            self.aspect = float(aspect)

    def hoverMoveEvent(self, event):
        pos = event.pos()
        hi = self._hit_test_handle(pos)
        if hi is not None:
            cursor = Qt.CursorShape.SizeAllCursor
            if hi in (self.TL, self.BR):
                cursor = Qt.CursorShape.SizeFDiagCursor
            elif hi in (self.TR, self.BL):
                cursor = Qt.CursorShape.SizeBDiagCursor
            elif hi in (self.TM, self.BM):
                cursor = Qt.CursorShape.SizeVerCursor
            elif hi in (self.ML, self.MR):
                cursor = Qt.CursorShape.SizeHorCursor
            self.setCursor(cursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos_scene = event.scenePos()
            self._press_rect = QRectF(self.rect())
            if self._press_rect.height() > 0:
                self.aspect = float(self._press_rect.width() / self._press_rect.height())


            hi = self._hit_test_handle(event.pos())
            self._drag_mode = hi if hi is not None else "move"
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_mode is None:
            return super().mouseMoveEvent(event)

        delta = event.scenePos() - self._press_pos_scene
        r = QRectF(self._press_rect)

        if self._drag_mode == "move":
            r.moveTo(r.x() + delta.x(), r.y() + delta.y())
        else:
            hi = self._drag_mode
            x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
            dx, dy = delta.x(), delta.y()

            if hi in (self.TL, self.TM, self.TR):
                y1 += dy
            if hi in (self.BL, self.BM, self.BR):
                y2 += dy
            if hi in (self.TL, self.ML, self.BL):
                x1 += dx
            if hi in (self.TR, self.MR, self.BR):
                x2 += dx

            nx1, nx2 = sorted([x1, x2])
            ny1, ny2 = sorted([y1, y2])
            r = QRectF(QPointF(nx1, ny1), QPointF(nx2, ny2))

            # ---- aspect lock ----
            if self.lock_aspect and self.aspect > 0:
                hi = self._drag_mode
                a = self.aspect

                # anchor is the opposite corner (based on press_rect)
                pr = self._press_rect
                ax = pr.right() if hi in (self.TL, self.TM, self.ML, self.BL, self.BM) else pr.left()
                ay = pr.bottom() if hi in (self.TL, self.TM, self.TR, self.ML, self.MR) else pr.top()

                # use tentative width as driver for corners; height as driver for top/bottom handles
                tw, th = r.width(), r.height()
                if hi in (self.TM, self.BM):      # vertical handle => drive by height
                    th = max(1.0, th)
                    tw = th * a
                elif hi in (self.ML, self.MR):    # horizontal handle => drive by width
                    tw = max(1.0, tw)
                    th = tw / a
                else:                              # corners => drive by width (simple & stable)
                    tw = max(1.0, tw)
                    th = tw / a

                # rebuild rect keeping anchor fixed
                if hi in (self.TL, self.TM, self.ML):  # resizing from left/top-ish
                    x1 = ax - tw
                    x2 = ax
                else:
                    x1 = ax
                    x2 = ax + tw

                if hi in (self.TL, self.TM, self.TR):  # resizing from top-ish
                    y1 = ay - th
                    y2 = ay
                else:
                    y1 = ay
                    y2 = ay + th

                nx1, nx2 = sorted([x1, x2])
                ny1, ny2 = sorted([y1, y2])
                r = QRectF(QPointF(nx1, ny1), QPointF(nx2, ny2))
            # ---- aspect lock end ----

        r = self.rect_in_bounds(r)
        self.setRect(r)

        if self.on_changed:
            self.on_changed()  # debounced by app

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_mode = None
        super().mouseReleaseEvent(event)

    def setRect(self, rect: QRectF):
        super().setRect(rect)
        self._update_handles()


class ImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform if QT6 else QPainter.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing if QT6 else QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # Slightly lighter background reduces the perception of "black bars"
        self.setBackgroundBrush(Qt.GlobalColor.darkGray if QT6 else Qt.darkGray)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter if QT6 else Qt.AlignCenter)

    def wheelEvent(self, event):
        # Ctrl + wheel => zoom view
        if (event.modifiers() & Qt.KeyboardModifier.ControlModifier) if QT6 else (event.modifiers() & Qt.ControlModifier):
            angle = event.angleDelta().y()
            factor = 1.15 if angle > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)


class ROICropGridApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI Crop Grid Compare")
        self.setMinimumSize(1400, 780)
        self._right_rot_k = 0

        self.folder: Optional[Path] = None
        self.paths: List[Path] = []
        self.pil_images: List[Image.Image] = []
        self.img_size: Optional[Tuple[int, int]] = None

        self.scene = QGraphicsScene(self)
        self.view = ImageView()
        self.view.setScene(self.scene)

        self.pix_item = None
        self.roi_item: Optional[ROIItem] = None

        # visibility mask for right grid
        self.visible_mask: List[bool] = []

        # debounced refresh (fix "flicker/popups" / heavy refresh while dragging fast)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh_grid_now)
        self._refresh_interval_ms = 33  # ~30fps
        self._last_tile_size: Tuple[int, int] = (0, 0)

        # Right grid
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(6)
        self.grid.setContentsMargins(8, 8, 8, 8)
        self.scroll.setWidget(self.grid_host)

        self._roi_updating_from_ui = False
        self._roi_last_user_edit = None  # "w" / "h" / "x" / "y"

        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        # -------- Top controls (two-row, compact) --------
        top = QVBoxLayout()
        top.setSpacing(8)
        outer.addLayout(top, stretch=0)

        # Row 1: main actions
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        top.addLayout(row1)

        self.btn_open = QPushButton("Open")
        self.btn_open.setFixedHeight(30)
        self.btn_open.clicked.connect(self.open_folder)
        row1.addWidget(self.btn_open)

        row1.addWidget(QLabel("Image:"))
        self.combo_img = QComboBox()
        self.combo_img.setMinimumWidth(320)
        self.combo_img.currentIndexChanged.connect(self.on_select_image)
        row1.addWidget(self.combo_img, stretch=1)

        self.btn_reset_roi = QPushButton("Reset ROI")
        self.btn_reset_roi.setFixedHeight(30)
        self.btn_reset_roi.clicked.connect(self.reset_roi)
        row1.addWidget(self.btn_reset_roi)

        self.btn_save = QPushButton("Save (S)")
        self.btn_save.setFixedHeight(30)
        self.btn_save.clicked.connect(self.save_outputs)
        row1.addWidget(self.btn_save)

        # Row 2: compact numeric controls + visible list
        row2 = QHBoxLayout()
        row2.setSpacing(12)
        top.addLayout(row2)

        ctrl = QGroupBox("Grid / Zoom")
        ctrl.setMaximumWidth(360)
        form = QFormLayout(ctrl)
        form.setContentsMargins(10, 10, 10, 10)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

        self.spin_cols = QSpinBox(); self.spin_cols.setRange(1, 20); self.spin_cols.setValue(3)
        self.spin_rows = QSpinBox(); self.spin_rows.setRange(1, 50); self.spin_rows.setValue(3)
        self.spin_zoom = QDoubleSpinBox(); self.spin_zoom.setRange(0.1, 50.0); self.spin_zoom.setSingleStep(0.25); self.spin_zoom.setValue(4.0)
        self.spin_tile_max = QSpinBox(); self.spin_tile_max.setRange(64, 4096); self.spin_tile_max.setValue(640); self.spin_tile_max.setSingleStep(64)

        for w in (self.spin_cols, self.spin_rows, self.spin_zoom):
            w.setFixedWidth(90)
        self.spin_tile_max.setFixedWidth(110)

        self.spin_cols.valueChanged.connect(self.schedule_refresh_grid)
        self.spin_rows.valueChanged.connect(self.schedule_refresh_grid)
        self.spin_zoom.valueChanged.connect(self.schedule_refresh_grid)
        self.spin_tile_max.valueChanged.connect(self.schedule_refresh_grid)

        form.addRow("Cols", self.spin_cols)
        form.addRow("Rows", self.spin_rows)
        form.addRow("Zoom", self.spin_zoom)
        form.addRow("Max", self.spin_tile_max)
        row2.addWidget(ctrl, stretch=0)

        roi_box = QGroupBox("ROI (Left)")
        roi_box.setMaximumWidth(360)
        roi_form = QFormLayout(roi_box)
        roi_form.setContentsMargins(10, 10, 10, 10)
        roi_form.setHorizontalSpacing(10)
        roi_form.setVerticalSpacing(6)

        self.spin_roi_w = QSpinBox(); self.spin_roi_w.setRange(1, 999999); self.spin_roi_w.setFixedWidth(110)
        self.spin_roi_h = QSpinBox(); self.spin_roi_h.setRange(1, 999999); self.spin_roi_h.setFixedWidth(110)
        self.spin_roi_x = QSpinBox(); self.spin_roi_x.setRange(0, 999999); self.spin_roi_x.setFixedWidth(110)
        self.spin_roi_y = QSpinBox(); self.spin_roi_y.setRange(0, 999999); self.spin_roi_y.setFixedWidth(110)

        self.chk_lock_aspect = QCheckBox("Lock aspect")
        self.chk_lock_aspect.setChecked(False)

        roi_form.addRow("W", self.spin_roi_w)
        roi_form.addRow("H", self.spin_roi_h)
        roi_form.addRow("X (left)", self.spin_roi_x)
        roi_form.addRow("Y (top)", self.spin_roi_y)
        roi_form.addRow(self.chk_lock_aspect)

        self.spin_roi_w.valueChanged.connect(lambda _: self._on_roi_ui_changed("w"))
        self.spin_roi_h.valueChanged.connect(lambda _: self._on_roi_ui_changed("h"))
        self.spin_roi_x.valueChanged.connect(lambda _: self._on_roi_ui_changed("x"))
        self.spin_roi_y.valueChanged.connect(lambda _: self._on_roi_ui_changed("y"))
        self.chk_lock_aspect.stateChanged.connect(self._on_lock_aspect_changed)

        row2.addWidget(roi_box, stretch=0)

        vis = QGroupBox("Visible (Right)")
        vis.setMinimumWidth(360)
        vis_lay = QVBoxLayout(vis)
        vis_lay.setContentsMargins(10, 10, 10, 10)
        vis_lay.setSpacing(6)

        btns = QHBoxLayout(); btns.setSpacing(6)
        self.btn_all = QPushButton("All")
        self.btn_none = QPushButton("None")
        self.btn_all.setFixedWidth(60)
        self.btn_none.setFixedWidth(60)
        self.btn_all.clicked.connect(self.select_all_visible)
        self.btn_none.clicked.connect(self.select_none_visible)
        btns.addWidget(self.btn_all)
        btns.addWidget(self.btn_none)
        btns.addStretch(1)
        vis_lay.addLayout(btns)

        self.list_visible = QListWidget()
        self.list_visible.setMinimumHeight(110)
        self.list_visible.itemChanged.connect(self.on_visible_item_changed)
        vis_lay.addWidget(self.list_visible)
        row2.addWidget(vis, stretch=1)

        # -------- Bottom: left view / right grid --------
        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        outer.addLayout(bottom, stretch=1)

        # Bottom-left: image view (no extra black border feeling)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(6)
        bottom.addLayout(left_panel, stretch=2)
        left_panel.addWidget(self.view, stretch=1)

        hint = QLabel("Tips: drag ROI to move; drag corners to resize. Ctrl+Wheel zooms view. Press S to save.")
        hint.setStyleSheet("color: #E0E0E0;")
        left_panel.addWidget(hint, stretch=0)

        # Bottom-right: grid + status
        right_panel = QVBoxLayout()
        right_panel.setSpacing(6)
        bottom.addLayout(right_panel, stretch=3)
        right_panel.addWidget(self.scroll, stretch=1)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #EDEDED;")
        right_panel.addWidget(self.status, stretch=0)

    def _bind_shortcuts(self):
        # "S" to save, Ctrl+S also
        sc1 = QShortcut(QKeySequence("S"), self)
        sc1.activated.connect(self.save_outputs)
        sc2 = QShortcut(QKeySequence.StandardKey.Save, self) if QT6 else QShortcut(QKeySequence("Ctrl+S"), self)
        try:
            sc2.activated.connect(self.save_outputs)
        except Exception:
            pass

        # "R": rotate right-grid crops by +90° each press
        sc3 = QShortcut(QKeySequence("R"), self)
        sc3.activated.connect(self.toggle_right_rotation)

    # ---------- data loading ----------
    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not d:
            return
        folder = Path(d)
        paths = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if not paths:
            QMessageBox.warning(self, "No images", "No supported image files found in this folder.")
            return

        pil_images: List[Image.Image] = []
        size0 = None
        for p in paths:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img).convert("RGB")
            if size0 is None:
                size0 = img.size
            elif img.size != size0:
                QMessageBox.warning(
                    self, "Size mismatch",
                    f"Image sizes are not identical.\nFirst: {size0}\nNow: {p.name} -> {img.size}\n\n(Program expects same size.)"
                )
                return
            pil_images.append(img)

        self.folder = folder
        self.paths = paths
        self.pil_images = pil_images
        self.img_size = size0

        # populate combobox
        self.combo_img.blockSignals(True)
        self.combo_img.clear()
        self.combo_img.addItems([p.name for p in self.paths])
        self.combo_img.blockSignals(False)
        self.combo_img.setCurrentIndex(0)

        # populate visibility list
        self.visible_mask = [True] * len(self.paths)
        self.list_visible.blockSignals(True)
        self.list_visible.clear()
        for p in self.paths:
            it = QListWidgetItem(p.name)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable if QT6 else it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if QT6 else Qt.Checked)
            self.list_visible.addItem(it)
        self.list_visible.blockSignals(False)

        self.load_left_image(0)

    def load_left_image(self, idx: int):
        if not (0 <= idx < len(self.pil_images)):
            return
        img = self.pil_images[idx]
        pix = pil_to_qpixmap(img)

        self.scene.clear()
        self.pix_item = self.scene.addPixmap(pix)
        self.pix_item.setZValue(0)

        w, h = img.size
        self.scene.setSceneRect(0, 0, w, h)

        # fit view
        self.view.resetTransform()
        # Fill the view as much as possible (minimize perceived borders).
        # Note: this may crop a small part of the image if the widget aspect differs.
        self.view.fitInView(
            self.scene.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding if QT6 else Qt.KeepAspectRatioByExpanding,
        )

        bounds = QRectF(0, 0, w, h)
        roi = QRectF(w * 0.25, h * 0.25, w * 0.25, h * 0.25)
        self.roi_item = ROIItem(roi, bounds_rect=bounds, on_changed=self.schedule_refresh_grid)
        self._sync_roi_spin_ranges()
        self._sync_roi_ui_from_item()
        self.roi_item.set_aspect_lock(self.chk_lock_aspect.isChecked(), self.roi_item.aspect)

        self.scene.addItem(self.roi_item)

        self.schedule_refresh_grid()

    def on_select_image(self, idx: int):
        if idx < 0:
            return
        self.load_left_image(idx)

    # ---------- ROI / refresh ----------
    def reset_roi(self):
        if not self.img_size or not self.roi_item:
            return
        w, h = self.img_size
        roi = QRectF(w * 0.25, h * 0.25, w * 0.25, h * 0.25)
        roi = self.roi_item.rect_in_bounds(roi)
        self.roi_item.setRect(roi)
        self.schedule_refresh_grid()

    def _sync_roi_spin_ranges(self):
        if not self.img_size:
            return
        W, H = self.img_size
        for s in (self.spin_roi_w, self.spin_roi_h, self.spin_roi_x, self.spin_roi_y):
            s.blockSignals(True)
        self.spin_roi_w.setRange(1, W)
        self.spin_roi_h.setRange(1, H)
        self.spin_roi_x.setRange(0, W - 1)
        self.spin_roi_y.setRange(0, H - 1)
        for s in (self.spin_roi_w, self.spin_roi_h, self.spin_roi_x, self.spin_roi_y):
            s.blockSignals(False)

    def _sync_roi_ui_from_item(self):
        """Called when ROI moved/resized by mouse: update spinboxes (x,y,w,h)."""
        if self._roi_updating_from_ui:
            return
        if not self.roi_item or not self.img_size:
            return
        r = self.roi_item.rect()
        x = int(round(r.left()))
        y = int(round(r.top()))
        w = int(round(r.width()))
        h = int(round(r.height()))
        for s in (self.spin_roi_w, self.spin_roi_h, self.spin_roi_x, self.spin_roi_y):
            s.blockSignals(True)
        self.spin_roi_x.setValue(max(0, x))
        self.spin_roi_y.setValue(max(0, y))
        self.spin_roi_w.setValue(max(1, w))
        self.spin_roi_h.setValue(max(1, h))
        for s in (self.spin_roi_w, self.spin_roi_h, self.spin_roi_x, self.spin_roi_y):
            s.blockSignals(False)

    def _on_lock_aspect_changed(self, _state):
        if not self.roi_item:
            return
        r = self.roi_item.rect()
        aspect = float(r.width() / max(1.0, r.height()))
        locked = self.chk_lock_aspect.isChecked()
        self.roi_item.set_aspect_lock(locked, aspect)

        # 若锁定开启，立刻把 UI 的 w/h 调整到当前比例（保持当前 rect 不动）
        self._sync_roi_ui_from_item()
        self.schedule_refresh_grid()

    def _on_roi_ui_changed(self, which: str):
        """User edits spinboxes -> update ROI rect (and keep aspect if locked)."""
        if not self.roi_item or not self.img_size:
            return
        self._roi_updating_from_ui = True
        try:
            W, H = self.img_size
            x = int(self.spin_roi_x.value())
            y = int(self.spin_roi_y.value())
            w = int(self.spin_roi_w.value())
            h = int(self.spin_roi_h.value())

            locked = self.chk_lock_aspect.isChecked()
            if locked:
                # lock to roi_item.aspect (keeps stable while user edits)
                a = float(self.roi_item.aspect if self.roi_item.aspect > 0 else (w / max(1, h)))

                if which == "w":
                    h = max(1, int(round(w / a)))
                    self.spin_roi_h.blockSignals(True)
                    self.spin_roi_h.setValue(h)
                    self.spin_roi_h.blockSignals(False)
                elif which == "h":
                    w = max(1, int(round(h * a)))
                    self.spin_roi_w.blockSignals(True)
                    self.spin_roi_w.setValue(w)
                    self.spin_roi_w.blockSignals(False)

            # clamp to image bounds
            w = max(1, min(w, W))
            h = max(1, min(h, H))
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            if x + w > W:
                x = W - w
            if y + h > H:
                y = H - h

            nr = QRectF(float(x), float(y), float(w), float(h))
            nr = self.roi_item.rect_in_bounds(nr)
            self.roi_item.setRect(nr)

            # keep ROIItem lock/aspect in sync
            self.roi_item.set_aspect_lock(locked, float(nr.width() / max(1.0, nr.height())) if locked else self.roi_item.aspect)

            self.schedule_refresh_grid()
        finally:
            self._roi_updating_from_ui = False


    def schedule_refresh_grid(self):
        # debounce: restart timer so heavy redraw won't happen for every mouse move
        # self._refresh_timer.start(self._refresh_interval_ms)
        self._sync_roi_ui_from_item()
        self._refresh_timer.start(self._refresh_interval_ms)


    def toggle_right_rotation(self):
        """Rotate the right-grid crop display by +90° each press (0/90/180/270)."""
        self._right_rot_k = (self._right_rot_k + 1) % 4
        self.schedule_refresh_grid()

    def _clear_grid(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _get_roi_box_int(self) -> Optional[Tuple[int, int, int, int]]:
        if not self.roi_item or not self.img_size:
            return None
        r = self.roi_item.rect()
        x1 = int(round(r.left()))
        y1 = int(round(r.top()))
        x2 = int(round(r.right()))
        y2 = int(round(r.bottom()))
        W, H = self.img_size
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(1, min(W, x2))
        y2 = max(1, min(H, y2))
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return None
        return (x1, y1, x2, y2)

    def _compute_tile_size(self, roi_w: int, roi_h: int) -> Tuple[int, int]:
        zoom = float(self.spin_zoom.value())
        max_edge = int(self.spin_tile_max.value())

        tile_w = max(32, int(round(roi_w * zoom)))
        tile_h = max(32, int(round(roi_h * zoom)))
        scale = min(max_edge / max(tile_w, 1), max_edge / max(tile_h, 1), 1.0)
        tile_w = int(tile_w * scale)
        tile_h = int(tile_h * scale)
        return tile_w, tile_h

    def _refresh_grid_now(self):
        if not self.pil_images or not self.img_size:
            return
        roi = self._get_roi_box_int()
        if roi is None:
            return

        x1, y1, x2, y2 = roi
        rw, rh = (x2 - x1), (y2 - y1)
        cols = int(self.spin_cols.value())
        rows = int(self.spin_rows.value())
        cap = cols * rows

        # visible indices
        vis_ids = [i for i, ok in enumerate(self.visible_mask) if ok]
        show_ids = vis_ids[:cap]

        # If rotated by 90/270, the crop's width/height swap for display.
        eff_w, eff_h = (rh, rw) if (self._right_rot_k % 2 == 1) else (rw, rh)
        tile_w, tile_h = self._compute_tile_size(eff_w, eff_h)
        self._last_tile_size = (tile_w, tile_h)

        self._clear_grid()

        zoom = float(self.spin_zoom.value())

        for k, i in enumerate(show_ids):
            img = self.pil_images[i]
            crop = img.crop((x1, y1, x2, y2))
            if self._right_rot_k:
                # PIL rotates counter-clockwise for positive angles; use negative for clockwise.
                crop = crop.rotate(-90 * self._right_rot_k, expand=True)
            crop = crop.resize((tile_w, tile_h), Image.Resampling.NEAREST if zoom >= 6 else Image.Resampling.LANCZOS)
            pix = pil_to_qpixmap(crop)

            cell = QWidget()
            v = QVBoxLayout(cell)
            v.setContentsMargins(2, 2, 2, 2)
            v.setSpacing(2)

            lab_img = QLabel()
            lab_img.setPixmap(pix)
            lab_img.setFixedSize(tile_w, tile_h)
            lab_img.setScaledContents(True)
            lab_img.setStyleSheet("background: #111; border: 1px solid #333;")
            v.addWidget(lab_img)

            # Show short model tag: part before the first underscore
            short = self.paths[i].stem.split("_", 1)[0]
            lab_txt = QLabel(short)
            lab_txt.setStyleSheet("color: #00FF66; font-weight: 700;")
            lab_txt.setWordWrap(False)
            lab_txt.setFixedWidth(tile_w)
            v.addWidget(lab_txt)

            r = k // cols
            c = k % cols
            self.grid.addWidget(cell, r, c)

        rot_deg = (self._right_rot_k * 90) % 360
        self.status.setText(
            f"Loaded: {len(self.pil_images)} images | Visible: {len(vis_ids)} | "
            f"ROI: ({x1},{y1})-({x2},{y2}) size={rw}x{rh} | "
            f"Tile: {tile_w}x{tile_h} | Rot: {rot_deg}° | "
            f"Grid: {rows}x{cols} (show {len(show_ids)}/{len(vis_ids)})"
        )

    # ---------- visibility controls ----------
    def on_visible_item_changed(self, item: QListWidgetItem):
        row = self.list_visible.row(item)
        if 0 <= row < len(self.visible_mask):
            checked = (item.checkState() == (Qt.CheckState.Checked if QT6 else Qt.Checked))
            self.visible_mask[row] = bool(checked)
            self.schedule_refresh_grid()

    def select_all_visible(self):
        if not self.paths:
            return
        self.visible_mask = [True] * len(self.paths)
        self.list_visible.blockSignals(True)
        for i in range(self.list_visible.count()):
            self.list_visible.item(i).setCheckState(Qt.CheckState.Checked if QT6 else Qt.Checked)
        self.list_visible.blockSignals(False)
        self.schedule_refresh_grid()

    def select_none_visible(self):
        if not self.paths:
            return
        self.visible_mask = [False] * len(self.paths)
        self.list_visible.blockSignals(True)
        for i in range(self.list_visible.count()):
            self.list_visible.item(i).setCheckState(Qt.CheckState.Unchecked if QT6 else Qt.Unchecked)
        self.list_visible.blockSignals(False)
        self.schedule_refresh_grid()

    # ---------- save ----------
    def save_outputs(self):
        """
        Save:
          1) ROI crop (zoomed to current tile size) for each visible image
          2) Left full image with ROI overlay (scene rendered at original resolution)
        """
        if not self.folder or not self.pil_images or not self.img_size or not self.roi_item:
            return

        roi = self._get_roi_box_int()
        if roi is None:
            QMessageBox.warning(self, "ROI invalid", "ROI is invalid (too small).")
            return
        x1, y1, x2, y2 = roi
        rw, rh = (x2 - x1), (y2 - y1)

        k = getattr(self, "_right_rot_k", 0)
        # >>> ADD: rotation-aware effective size
        eff_w, eff_h = (rh, rw) if (k % 2 == 1) else (rw, rh)
        # <<< ADD

        tile_w, tile_h = self._compute_tile_size(eff_w, eff_h)

        # output folder (timestamp)
        out_dir = self.folder / "roi_exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        # unique run subfolder
        import datetime
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = out_dir / f"export_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # ---- write ROI info txt ----
        locked = bool(self.chk_lock_aspect.isChecked()) if hasattr(self, "chk_lock_aspect") else False
        aspect = (rw / rh) if rh > 0 else 0.0
        info_txt = run_dir / "roi_info.txt"
        info_txt.write_text(
            "\n".join([
                f"x1={x1}",
                f"y1={y1}",
                f"w={rw}",
                f"h={rh}",
                f"aspect={aspect:.8f}",
                f"lock_aspect={int(locked)}",
                f"right_rot_deg={(getattr(self, '_right_rot_k', 0) * 90) % 360}",
            ]) + "\n",
            encoding="utf-8"
        )
        # ---- end ----

        # Save the left full image with a clean ROI rectangle (no resize handles).
        # idx = max(0, self.combo_img.currentIndex())
        # idx = min(idx, len(self.pil_images) - 1)
        # left_img = self.pil_images[idx].copy()
        # draw = ImageDraw.Draw(left_img)
        # # A clean rectangle looks nicer than the interactive ROI item with handles
        # draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(0, 255, 0), width=4)

        # left_name = self.paths[idx].stem if 0 <= idx < len(self.paths) else "left"
        # left_path = run_dir / f"{left_name}__left_with_roi.png"
        # left_img.save(left_path)

        # === NEW: save ALL images with the same ROI overlay ===
        for i, img in enumerate(self.pil_images):
            img_with_roi = img.copy()
            draw = ImageDraw.Draw(img_with_roi)
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(0, 255, 0), width=4)

            name = self.paths[i].stem
            save_path = run_dir / f"{name}__with_roi.png"
            img_with_roi.save(save_path)

        # save roi crops for each visible image
        vis_ids = [i for i, ok in enumerate(self.visible_mask) if ok]
        zoom = float(self.spin_zoom.value())
        resample = Image.Resampling.NEAREST if zoom >= 6 else Image.Resampling.LANCZOS

        crops_dir = run_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        for i in vis_ids:
            img = self.pil_images[i]
            crop = img.crop((x1, y1, x2, y2)).resize((tile_w, tile_h), resample)

            k = getattr(self, "_right_rot_k", 0)
            if k:
                crop = crop.rotate(-90 * k, expand=True)
            crop = crop.resize((tile_w, tile_h), Image.Resampling.NEAREST if zoom >= 6 else Image.Resampling.LANCZOS)

            # save_path = crops_dir / f"{self.paths[i].stem}__roi_{x1}_{y1}_{x2}_{y2}__x{zoom:.2f}.png"
            save_path = crops_dir / f"{self.paths[i].stem}.png"
            crop.save(save_path)

        QMessageBox.information(
            self, "Saved",
            f"Saved to:\n{run_dir}\n\n"
            f"- Left view w/ ROI: {len(self.pil_images)}\n"
            f"- ROI crops: {len(vis_ids)} images in {crops_dir.name}/"
        )


def main():
    app = QApplication(sys.argv)
    w = ROICropGridApp()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
