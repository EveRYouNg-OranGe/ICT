import argparse
import json
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(in_dir: Path) -> List[Path]:
    return sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def pil_open_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return im


def pil_to_bgr(im: Image.Image) -> np.ndarray:
    arr = np.array(im)  # RGB
    return arr[:, :, ::-1].copy()  # BGR


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb)


def pad_to_size(im: Image.Image, target_w: int, target_h: int, bg=(0, 0, 0)) -> Image.Image:
    """Center-pad (no crop) to exact size."""
    w, h = im.size
    out = Image.new("RGB", (target_w, target_h), bg)
    x0 = (target_w - w) // 2
    y0 = (target_h - h) // 2
    out.paste(im, (x0, y0))
    return out


def rotated_rect_crop_bgr(
    img_bgr: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_deg: float,
) -> np.ndarray:
    """
    Crop a rotated rectangle (center cx,cy; size w,h; angle in degrees CCW) from BGR image.
    Returns an axis-aligned patch of size (h,w) in BGR.

    解释：把整图绕 (cx,cy) 旋转 angle_deg，使得“旋转框选区域”变成水平/竖直，
    然后用 getRectSubPix 取出 w*h 的内容。这就等价于“框选内容旋转过来显示”。
    """
    H, W = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rot = cv2.warpAffine(
        img_bgr, M, (W, H),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    patch = cv2.getRectSubPix(rot, (int(round(w)), int(round(h))), (cx, cy))
    return patch


def build_strip(
    img_pil: Image.Image,
    img_bgr: np.ndarray,
    roi1: Tuple[float, float, float, float],  # x,y,w,h (edit bbox)
    roi2: Tuple[float, float, float, float],
    rot90_roi1: bool,
    rot90_roi2: bool,
    cover_h: int,
    split_x: int,
    bg_mode: str = "black",
) -> Image.Image:
    """
    Bottom strip: exact size (W, cover_h), with two panes:
      pane1 = (split_x, cover_h)
      pane2 = (W-split_x, cover_h)

    每个 pane 的显示规则：
      1) 从原图取 ROI（轴对齐矩形，允许超出边界，cv2.getRectSubPix 会补黑边）
      2) 若 rot90=True，则将 ROI patch 旋转 90 度（把“竖起来的选框内容”横过来显示）
      3) 用 contain 缩放到 pane 内（不裁内容）
      4) pad 到刚好 pane 大小（letterbox），保证严丝合缝
    """
    W, H = img_pil.size

    if bg_mode == "orig":
        bg = img_pil.crop((0, H - cover_h, W, H)).copy()
    else:
        bg = Image.new("RGB", (W, cover_h), (0, 0, 0))

    pane1_w = clamp(int(split_x), 1, W - 1)
    pane2_w = W - pane1_w
    pane_h = clamp(int(cover_h), 1, H)

    def extract_and_fit(roi, rot90: bool, pane_w: int):
        x, y, rw, rh = roi
        cx = clamp(x + rw / 2.0, 0.0, W - 1.0)
        cy = clamp(y + rh / 2.0, 0.0, H - 1.0)
        rw = clamp(rw, 2.0, W)
        rh = clamp(rh, 2.0, H)

        # axis-aligned crop (with padding if out-of-bounds)
        patch_bgr = cv2.getRectSubPix(img_bgr, (int(round(rw)), int(round(rh))), (float(cx), float(cy)))

        if rot90:
            # Rotate patch to make it horizontal for display panes
            patch_bgr = cv2.rotate(patch_bgr, cv2.ROTATE_90_CLOCKWISE)

        patch_pil = bgr_to_pil(patch_bgr)

        fitted = ImageOps.contain(patch_pil, (pane_w, pane_h), Image.Resampling.LANCZOS)
        final = pad_to_size(fitted, pane_w, pane_h, bg=(0, 0, 0))
        return final

    p1 = extract_and_fit(roi1, rot90_roi1, pane1_w)
    p2 = extract_and_fit(roi2, rot90_roi2, pane2_w)

    bg.paste(p1, (0, 0))
    bg.paste(p2, (pane1_w, 0))
    return bg


def overlay_strip(img: Image.Image, strip: Image.Image, cover_h: int) -> Image.Image:
    out = img.copy()
    W, H = img.size
    out.paste(strip, (0, H - cover_h))
    return out

    return out

def draw_save_overlays(
    img: Image.Image,
    roi1: Tuple[float, float, float, float],
    roi2: Tuple[float, float, float, float],
    cover_h: int,
    split_x: int,
    thickness: int = 4,
):
    """
    Draw overlays for saved images:
      - ROI1 bbox (red) and ROI2 bbox (green) on the top area
      - Bottom panes border: left pane red, right pane green
    NOTE: ROI boxes here are axis-aligned editing boxes (same as UI boxes).
    """
    W, H = img.size
    y0 = H - int(cover_h)
    x_split = int(split_x)

    draw = ImageDraw.Draw(img)

    # Colors in RGB
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)

    def rect_outline(x0, y0, x1, y1, color):
        # Pillow doesn't have thickness on older versions for rectangle outline reliably,
        # so draw multiple rectangles expanded inward/outward.
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        for t in range(thickness):
            draw.rectangle([x0 + t, y0 + t, x1 - t, y1 - t], outline=color)

    # ROI rectangles
    x, y, w, h = roi1
    rect_outline(x, y, x + w, y + h, RED)
    x, y, w, h = roi2
    rect_outline(x, y, x + w, y + h, GREEN)

    # Bottom pane borders
    rect_outline(0, y0, x_split, H - 1, RED)
    rect_outline(x_split, y0, W - 1, H - 1, GREEN)

    return img


# ---------------- GUI interaction ----------------

class DragMode:
    NONE = 0
    DRAG_HLINE = 1
    DRAG_VLINE = 2
    DRAG_RECT_MOVE = 3
    DRAG_RECT_RESIZE = 4


def point_in_rect(px, py, rect):
    x, y, w, h = rect
    return (x <= px <= x + w) and (y <= py <= y + h)


def near_corner(px, py, rect, tol=12):
    x, y, w, h = rect
    corners = [
        (x, y, "tl"),
        (x + w, y, "tr"),
        (x, y + h, "bl"),
        (x + w, y + h, "br"),
    ]
    best = None
    best_d = 10**9
    for cx, cy, tag in corners:
        d = (px - cx) ** 2 + (py - cy) ** 2
        if d < best_d:
            best_d = d
            best = tag
    if best_d <= tol * tol:
        return best
    return None


def fit_rect_within(rect, bounds):
    x, y, w, h = rect
    bx, by, bw, bh = bounds
    w = clamp(w, 5, bw)
    h = clamp(h, 5, bh)
    x = clamp(x, bx, bx + bw - w)
    y = clamp(y, by, by + bh - h)
    return (x, y, w, h)


def resize_rect_locked_aspect(rect, corner_tag, px, py, aspect_w_over_h: float, min_w=10, min_h=10):
    """
    Resize rect by dragging one corner, keep fixed aspect ratio (w/h = aspect).

    NOTE:
    之前版本用 dirx/diry 乘法在某些象限会让 dx/dy 变成负数，从而被 clamp 到 min_w，
    导致“几乎不能缩放”的 bug。这里改为基于 anchor + 绝对距离的稳健实现。
    """
    x, y, w, h = rect
    x2 = x + w
    y2 = y + h
    aspect = max(1e-6, float(aspect_w_over_h))

    # Choose anchor (opposite corner)
    if corner_tag == "tl":
        ax, ay = x2, y2
    elif corner_tag == "tr":
        ax, ay = x, y2
    elif corner_tag == "bl":
        ax, ay = x2, y
    elif corner_tag == "br":
        ax, ay = x, y
    else:
        return rect

    # raw sizes from mouse
    raw_w = max(float(min_w), abs(px - ax))
    raw_h = max(float(min_h), abs(py - ay))

    # enforce aspect, pick limiting dimension closest to mouse intent
    if raw_w / raw_h > aspect:
        new_h = raw_h
        new_w = new_h * aspect
    else:
        new_w = raw_w
        new_h = new_w / aspect

    # reconstruct rect based on which corner is moving
    if corner_tag == "tl":
        nx, ny = ax - new_w, ay - new_h
    elif corner_tag == "tr":
        nx, ny = ax, ay - new_h
    elif corner_tag == "bl":
        nx, ny = ax - new_w, ay
    else:  # "br"
        nx, ny = ax, ay

    return (nx, ny, new_w, new_h)


class Wizard:
    """
    step 0: horizontal split line (cover area)
    step 1: vertical split line (two panes)
    step 2: move/resize (aspect-locked) + rotate ROI1/ROI2
    """

    def __init__(self, bgr_img: np.ndarray, pil_img: Image.Image):
        self.base_bgr = bgr_img
        self.pil_img = pil_img
        self.H, self.W = bgr_img.shape[:2]

        self.step = 0
        self.selected_roi = 1  # 1 or 2

        self.cover_ratio = 0.5
        self.cover_h = int(round(self.H * self.cover_ratio))
        self.split_y = self.H - self.cover_h
        self.split_x = self.W // 2

        self.roi1 = (int(self.W * 0.10), int(self.split_y * 0.15), int(self.W * 0.35), int(self.split_y * 0.35))
        self.roi2 = (int(self.W * 0.55), int(self.split_y * 0.15), int(self.W * 0.35), int(self.split_y * 0.35))

        self.rot90_roi1 = False
        self.rot90_roi2 = False

        self.drag_mode = DragMode.NONE
        self.drag_dx = 0
        self.drag_dy = 0
        self.drag_corner = None

        self.top_bounds = (0, 0, self.W, max(1, self.split_y))

        self.aspect1 = 1.0
        self.aspect2 = 1.0

    def _recalc_bounds(self):
        self.top_bounds = (0, 0, self.W, max(1, self.split_y))
        self.roi1 = fit_rect_within(self.roi1, self.top_bounds)
        self.roi2 = fit_rect_within(self.roi2, self.top_bounds)

    def lock_aspects_from_panes(self):
        pane1_w = clamp(int(self.split_x), 1, self.W - 1)
        pane2_w = self.W - pane1_w
        pane_h = clamp(int(self.cover_h), 1, self.H)

        # pane aspect (display target aspect)
        pane_aspect1 = pane1_w / float(pane_h)
        pane_aspect2 = pane2_w / float(pane_h)

        def roi_aspect(pane_aspect: float, rot90: bool) -> float:
            # If ROI is "standing up" (rot90), its box in the top area should be the inverse aspect,
            # because we'll rotate the patch 90° before displaying into the (wide) pane.
            pane_aspect = max(1e-6, float(pane_aspect))
            return (1.0 / pane_aspect) if rot90 else pane_aspect

        self.aspect1 = roi_aspect(pane_aspect1, self.rot90_roi1)
        self.aspect2 = roi_aspect(pane_aspect2, self.rot90_roi2)

        def adjust(rect, aspect):
            x, y, w, h = rect
            cx = x + w / 2.0
            cy = y + h / 2.0
            # keep current height, adjust width to match aspect
            new_h = max(10.0, h)
            new_w = max(10.0, new_h * aspect)
            nx = cx - new_w / 2.0
            ny = cy - new_h / 2.0
            return fit_rect_within((nx, ny, new_w, new_h), self.top_bounds)

        self.roi1 = adjust(self.roi1, self.aspect1)
        self.roi2 = adjust(self.roi2, self.aspect2)

    def set_cover_from_split_y(self, y):
        y = clamp(y, 1, self.H - 1)
        self.split_y = y
        self.cover_h = self.H - self.split_y
        self.cover_ratio = self.cover_h / self.H
        self._recalc_bounds()
        if self.step >= 2:
            self.lock_aspects_from_panes()

    def toggle_rot90_selected(self):
        if self.selected_roi == 1:
            self.rot90_roi1 = not self.rot90_roi1
        else:
            self.rot90_roi2 = not self.rot90_roi2
        # toggling changes the aspect-lock rule; re-lock to panes immediately
        if self.step >= 2:
            self.lock_aspects_from_panes()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.step == 0:
                if abs(y - self.split_y) <= 10:
                    self.drag_mode = DragMode.DRAG_HLINE

            elif self.step == 1:
                if y >= self.split_y and abs(x - self.split_x) <= 10:
                    self.drag_mode = DragMode.DRAG_VLINE

            elif self.step == 2:
                r1 = self.roi1
                r2 = self.roi2
                hit1 = point_in_rect(x, y, r1)
                hit2 = point_in_rect(x, y, r2)

                rect = None
                rect_id = None
                if self.selected_roi == 1 and hit1:
                    rect, rect_id = r1, 1
                elif self.selected_roi == 2 and hit2:
                    rect, rect_id = r2, 2
                elif hit1:
                    rect, rect_id = r1, 1
                elif hit2:
                    rect, rect_id = r2, 2

                if rect is not None:
                    self.selected_roi = rect_id
                    corner = near_corner(x, y, rect, tol=14)
                    if corner is not None:
                        self.drag_mode = DragMode.DRAG_RECT_RESIZE
                        self.drag_corner = corner
                    else:
                        self.drag_mode = DragMode.DRAG_RECT_MOVE
                        rx, ry, rw, rh = rect
                        self.drag_dx = x - rx
                        self.drag_dy = y - ry

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_mode = DragMode.NONE
            self.drag_corner = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_mode == DragMode.DRAG_HLINE:
                self.set_cover_from_split_y(y)

            elif self.drag_mode == DragMode.DRAG_VLINE:
                self.split_x = clamp(x, 1, self.W - 1)
                if self.step >= 2:
                    self.lock_aspects_from_panes()

            elif self.drag_mode == DragMode.DRAG_RECT_MOVE and self.step == 2:
                nx = x - self.drag_dx
                ny = y - self.drag_dy
                if self.selected_roi == 1:
                    self.roi1 = fit_rect_within((nx, ny, self.roi1[2], self.roi1[3]), self.top_bounds)
                else:
                    self.roi2 = fit_rect_within((nx, ny, self.roi2[2], self.roi2[3]), self.top_bounds)

            elif self.drag_mode == DragMode.DRAG_RECT_RESIZE and self.step == 2:
                if self.selected_roi == 1:
                    nr = resize_rect_locked_aspect(self.roi1, self.drag_corner, x, y, self.aspect1)
                    self.roi1 = fit_rect_within(nr, self.top_bounds)
                else:
                    nr = resize_rect_locked_aspect(self.roi2, self.drag_corner, x, y, self.aspect2)
                    self.roi2 = fit_rect_within(nr, self.top_bounds)

    def draw(self) -> np.ndarray:
        if self.step >= 2:
            strip = build_strip(
                self.pil_img,
                self.base_bgr,
                self.roi1,
                self.roi2,
                self.rot90_roi1,
                self.rot90_roi2,
                self.cover_h,
                self.split_x,
                bg_mode="black",
            )
            out_pil = overlay_strip(self.pil_img, strip, self.cover_h)
            vis = pil_to_bgr(out_pil)
        else:
            vis = self.base_bgr.copy()

        cv2.line(vis, (0, int(self.split_y)), (self.W - 1, int(self.split_y)), (0, 255, 255), 2)

        if self.step >= 1:
            cv2.line(vis, (int(self.split_x), int(self.split_y)), (int(self.split_x), self.H - 1), (255, 255, 0), 2)

        if self.step >= 2:
            def draw_rect(r, color, thick, rot90: bool):
                x, y, w, h = map(int, r)
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, thick)
                for (cx, cy) in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
                    cv2.circle(vis, (cx, cy), 4, color, -1)
                tag = "ROT90" if rot90 else "0"
                cv2.putText(vis, f"rot={tag}", (x + 5, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            c1 = (0, 0, 255) if self.selected_roi == 1 else (0, 0, 180)
            c2 = (0, 255, 0) if self.selected_roi == 2 else (0, 180, 0)
            draw_rect(self.roi1, c1, 3 if self.selected_roi == 1 else 2, self.rot90_roi1)
            draw_rect(self.roi2, c2, 3 if self.selected_roi == 2 else 2, self.rot90_roi2)

        help_lines = []
        if self.step == 0:
            help_lines = [
                "STEP 1/3: Drag HORIZONTAL line to choose bottom zoom area.",
                "Enter: confirm | Esc: quit",
            ]
        elif self.step == 1:
            help_lines = [
                "STEP 2/3: Drag VERTICAL line (in bottom area) to set pane widths.",
                "Enter: confirm | Esc: quit",
            ]
        else:
            help_lines = [
                "STEP 3/3: ROI boxes are aspect-locked to panes.",
                "Drag inside: move | Drag corners: resize (locked) | 1/2 select ROI",
                "r: toggle ROTATE 90° (stand ROI up) | Drag inside: move | corners: resize | Enter: APPLY | Esc: quit",
            ]
        y0 = 24
        for i, t in enumerate(help_lines):
            cv2.putText(vis, t, (10, y0 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(
            vis,
            f"cover_h={self.cover_h}px split_x={int(self.split_x)} "
            f"roi1_rot90={int(self.rot90_roi1)} roi2_rot90={int(self.rot90_roi2)}",
            (10, self.H - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    def next_step(self):
        self.step = min(2, self.step + 1)
        if self.step == 2:
            self.lock_aspects_from_panes()

    def get_config(self) -> dict:
        return {
            "cover_ratio": float(self.cover_h / self.H),
            "split_x": int(self.split_x),
            "roi1": [float(v) for v in self.roi1],
            "roi2": [float(v) for v in self.roi2],
            "rot90_roi1": bool(self.rot90_roi1),
            "rot90_roi2": bool(self.rot90_roi2),
        }


def process_folder(in_dir: Path, out_dir: Path, cfg: dict, bg_mode="black"):
    imgs = list_images(in_dir)
    if not imgs:
        raise RuntimeError(f"No images found in {in_dir}")

    for p in imgs:
        pil_img = pil_open_rgb(p)
        bgr = pil_to_bgr(pil_img)
        W, H = pil_img.size

        cover_h = int(round(H * cfg["cover_ratio"]))
        cover_h = clamp(cover_h, 1, H)
        split_x = clamp(int(cfg["split_x"]), 1, W - 1)

        roi1 = tuple(cfg["roi1"])
        roi2 = tuple(cfg["roi2"])
        r1 = bool(cfg.get("rot90_roi1", False))
        r2 = bool(cfg.get("rot90_roi2", False))

        strip = build_strip(pil_img, bgr, roi1, roi2, r1, r2, cover_h, split_x, bg_mode=bg_mode)
        out = overlay_strip(pil_img, strip, cover_h)

        out = draw_save_overlays(out, roi1, roi2, cover_h, split_x, thickness=4)

        rel = p.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(out_path)

    print(f"Done. Processed {len(imgs)} images -> {out_dir}")


def resolve_sample(in_dir: Path, imgs: List[Path], sample: int, sample_path: str) -> Path:
    if sample_path:
        sp = Path(sample_path)
        if sp.is_file():
            return sp
        cand = (in_dir / sample_path)
        if cand.is_file():
            return cand
        for p in imgs:
            if p.name == sample_path:
                return p
        raise FileNotFoundError(f"--sample_path not found: {sample_path}")
    idx = clamp(sample, 0, len(imgs) - 1)
    return imgs[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir", type=str)
    ap.add_argument("out_dir", type=str)

    ap.add_argument("--sample", type=int, default=0, help="Sample image index for GUI.")
    ap.add_argument("--sample_path", type=str, default="", help="Sample image path (abs or relative to in_dir).")

    ap.add_argument("--save_cfg", type=str, default="", help="Save config JSON path.")
    ap.add_argument("--load_cfg", type=str, default="", help="Load config JSON and skip GUI.")
    ap.add_argument("--bg_mode", choices=["black", "orig"], default="black")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    imgs = list_images(in_dir)
    if not imgs:
        raise RuntimeError(f"No images found in {in_dir}")

    if args.load_cfg:
        cfg = json.loads(Path(args.load_cfg).read_text(encoding="utf-8"))
        process_folder(in_dir, out_dir, cfg, bg_mode=args.bg_mode)
        return

    sample_path = resolve_sample(in_dir, imgs, args.sample, args.sample_path)

    pil_img = pil_open_rgb(sample_path)
    bgr = pil_to_bgr(pil_img)

    wiz = Wizard(bgr, pil_img)

    win = "ROI Zoom Wizard (aspect-locked + rotated ROI)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, wiz.W), min(950, wiz.H))
    cv2.setMouseCallback(win, wiz.on_mouse)

    while True:
        vis = wiz.draw()
        cv2.imshow(win, vis)
        key = cv2.waitKey(16) & 0xFF

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return

        if key in (13, 10):  # Enter
            if wiz.step < 2:
                wiz.next_step()
            else:
                cfg = wiz.get_config()
                if args.save_cfg:
                    Path(args.save_cfg).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.save_cfg).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                    print(f"Saved cfg to: {args.save_cfg}")
                cv2.destroyAllWindows()
                process_folder(in_dir, out_dir, cfg, bg_mode=args.bg_mode)
                return

        if key in (ord("1"), ord("2")) and wiz.step >= 2:
            wiz.selected_roi = 1 if key == ord("1") else 2

        if wiz.step >= 2:
            if key == ord("r"):
                wiz.toggle_rot90_selected()


if __name__ == "__main__":
    main()
