import os
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def scan_model_images(model_dir: Path) -> Dict[str, Path]:
    """
    Recursively scan images under model_dir, return mapping:
      relpath (posix str) -> absolute Path
    relpath is relative to model_dir.
    """
    out = {}
    for p in model_dir.rglob("*"):
        if is_img(p):
            rel = p.relative_to(model_dir).as_posix()
            out[rel] = p
    return out


def imread_rgb_u8(path: Path) -> np.ndarray:
    """
    Read image as RGB uint8, shape (H,W,3).
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    """
    PSNR between uint8 RGB patches. Returns float, inf if identical.
    """
    # use float32 for speed/precision tradeoff
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def to_gray_f32(rgb: np.ndarray) -> np.ndarray:
    # rgb uint8 -> gray float32
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32)

def sobel_mag(gray_f32: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag

def lap_var(gray_f32: np.ndarray) -> float:
    lap = cv2.Laplacian(gray_f32, cv2.CV_32F, ksize=3)
    return float(lap.var())

def psnr_f32(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float("inf")
    # data range：用两者联合的 max-min，避免不同图像幅值差导致不公平
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    dr = max(hi - lo, 1e-6)
    return 20.0 * math.log10(dr / math.sqrt(mse))


@dataclass
class PatchCand:
    score: float                # advantage = psnr(ours,gt) - max_other_psnr(gt)
    psnr_og: float              # ours vs gt
    best_other_psnr: float      # best(other vs gt)
    worst_other_psnr: float     # worst(other vs gt)
    relpath: str
    y: int
    x: int
    h: int
    w: int
    best_other_name: str
    worst_other_name: str

# -------- multiprocessing globals --------
_G_GT_MAP = None
_G_OURS_MAP = None
_G_OTHER_MAPS = None
_G_OTHER_MODELS = None
_G_PATCH = None
_G_MIN_PSNR_OG = None
_G_MIN_ADV = None
_G_MAX_BEST_OTHER_PSNR = None
_G_BLUR_MAP = None
_G_W_EDGE = None
_G_W_SHARP = None
_G_MIN_SHARP_GAIN = None


def _init_worker(gt_map, ours_map, blur_map, other_maps, other_models,
                 patch, min_psnr_og, min_adv, max_best_other_psnr,
                 w_edge, w_sharp, min_sharp_gain):
    global _G_GT_MAP, _G_OURS_MAP, _G_OTHER_MAPS, _G_OTHER_MODELS
    global _G_PATCH, _G_MIN_PSNR_OG, _G_MIN_ADV, _G_MAX_BEST_OTHER_PSNR
    global _G_BLUR_MAP, _G_W_EDGE, _G_W_SHARP, _G_MIN_SHARP_GAIN
    _G_BLUR_MAP = blur_map
    _G_W_EDGE = w_edge
    _G_W_SHARP = w_sharp
    _G_MIN_SHARP_GAIN = min_sharp_gain
    _G_GT_MAP = gt_map
    _G_OURS_MAP = ours_map
    _G_OTHER_MAPS = other_maps
    _G_OTHER_MODELS = other_models
    _G_PATCH = patch
    _G_MIN_PSNR_OG = min_psnr_og
    _G_MIN_ADV = min_adv
    _G_MAX_BEST_OTHER_PSNR = max_best_other_psnr

    # 避免 OpenCV 自己再开线程导致“线程/进程过度订阅”
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def _process_one_image(rel: str) -> List[PatchCand]:
    """
    视觉化指标版本：
    - 相似性：Edge-PSNR(ours_edge, gt_edge) 越高越像（轮廓一致）
    - 清晰度：Laplacian variance 越高越锐（更“清晰”）
    - 评分：score = w_edge * (edgePSNR(ours,gt) - max_edgePSNR(other,gt))
                 + w_sharp * (sharp(ours) - max_sharp(other))
    - 约束：sharp(ours) - sharp(blur) >= min_sharp_gain（如果提供了 blur）
    - best/worst 只在 other_models 内选（blur 已在 other_models 外部排除）
    """
    gt_path = _G_GT_MAP.get(rel, None)
    ours_path = _G_OURS_MAP.get(rel, None)
    if gt_path is None or ours_path is None:
        return []

    # read gt/ours
    try:
        gt = imread_rgb_u8(gt_path)
        ours = imread_rgb_u8(ours_path)
    except Exception:
        return []

    if gt.shape != ours.shape:
        return []

    H, W = gt.shape[:2]
    coords = tile_coords(H, W, _G_PATCH)
    if not coords:
        return []

    # read blur once (optional)
    blur = None
    if _G_BLUR_MAP is not None:
        blur_path = _G_BLUR_MAP.get(rel, None)
        if blur_path is not None:
            try:
                blur_tmp = imread_rgb_u8(blur_path)
                if blur_tmp.shape == gt.shape:
                    blur = blur_tmp
            except Exception:
                blur = None

    # load other models images lazily (per rel)
    others_img: Dict[str, np.ndarray] = {}
    for m in _G_OTHER_MODELS:
        p = _G_OTHER_MAPS[m].get(rel, None)
        if p is None:
            continue
        try:
            im = imread_rgb_u8(p)
        except Exception:
            continue
        if im.shape == gt.shape:
            others_img[m] = im

    if not others_img:
        return []

    cands: List[PatchCand] = []

    for (y, x) in coords:
        gt_p = gt[y:y+_G_PATCH, x:x+_G_PATCH]
        ours_p = ours[y:y+_G_PATCH, x:x+_G_PATCH]

        # gray & edges
        gt_g = to_gray_f32(gt_p)
        ours_g = to_gray_f32(ours_p)
        gt_e = sobel_mag(gt_g)
        ours_e = sobel_mag(ours_g)

        edge_psnr_og = psnr_f32(ours_e, gt_e)   # "ours close to gt" (on edges)
        sharp_ours = lap_var(ours_g)

        # blur constraint (optional)
        if blur is not None:
            blur_p = blur[y:y+_G_PATCH, x:x+_G_PATCH]
            blur_g = to_gray_f32(blur_p)
            sharp_blur = lap_var(blur_g)
            if _G_MIN_SHARP_GAIN > 0 and (sharp_ours - sharp_blur) < _G_MIN_SHARP_GAIN:
                continue

        # competitors on edge similarity + sharpness
        best_edge = -1.0
        best_name = ""
        best_sharp = -1.0

        worst_edge = float("inf")
        worst_name = ""

        for m, im in others_img.items():
            op = im[y:y+_G_PATCH, x:x+_G_PATCH]
            op_g = to_gray_f32(op)
            op_e = sobel_mag(op_g)

            e = psnr_f32(op_e, gt_e)
            s = lap_var(op_g)

            if e > best_edge:
                best_edge = e
                best_name = m
                best_sharp = s
            if e < worst_edge:
                worst_edge = e
                worst_name = m

        if best_edge < 0:
            continue

        # visual score: edge-consistency advantage + sharpness advantage
        score = _G_W_EDGE * (edge_psnr_og - best_edge) + _G_W_SHARP * (sharp_ours - best_sharp)

        # filters (沿用原参数，但语义变为 edge)
        if _G_MIN_PSNR_OG >= 0 and edge_psnr_og < _G_MIN_PSNR_OG:
            continue
        if _G_MIN_ADV >= 0 and score < _G_MIN_ADV:
            continue
        # "gt must differ from others": best other cannot be too close to gt on edges
        if best_edge > _G_MAX_BEST_OTHER_PSNR:
            continue

        # 复用字段：psnr_og/best_other_psnr/worst_other_psnr 存 edge-psnr
        cands.append(PatchCand(
            score=score,
            psnr_og=edge_psnr_og,
            best_other_psnr=best_edge,
            worst_other_psnr=worst_edge,
            relpath=rel,
            y=y, x=x, h=_G_PATCH, w=_G_PATCH,
            best_other_name=best_name,
            worst_other_name=worst_name,
        ))

    return cands




def tile_coords(H: int, W: int, patch: int) -> List[Tuple[int, int]]:
    """
    Non-overlapping tiles 50x50:
      (y,x) in steps of patch, only full tiles.
    """
    ys = range(0, H - patch + 1, patch)
    xs = range(0, W - patch + 1, patch)
    return [(y, x) for y in ys for x in xs]


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def cat_h(images: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate RGB images horizontally; pad to same height if needed.
    """
    hs = [im.shape[0] for im in images]
    H = max(hs)
    out = []
    for im in images:
        if im.shape[0] != H:
            pad = H - im.shape[0]
            im = np.pad(im, ((0, pad), (0, 0), (0, 0)), mode="constant", constant_values=0)
        out.append(im)
    return np.concatenate(out, axis=1)


def draw_label(im: np.ndarray, text: str) -> np.ndarray:
    """
    Put label on top-left.
    """
    out = im.copy()
    # OpenCV uses BGR; our array is RGB, so convert color accordingly
    # We'll draw in RGB but OpenCV expects BGR; easiest: temporarily treat as BGR by swapping.
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (out.shape[1], 18), (0, 0, 0), thickness=-1)
    cv2.putText(bgr, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="root folder containing subfolders: gt, ours, others...")
    ap.add_argument("--gt", type=str, default="gt", help="gt folder name")
    ap.add_argument("--ours", type=str, default="ours", help="ours folder name")
    ap.add_argument("--blur", type=str, default="blur", help="blur folder name (excluded from worst/best competitors)")
    ap.add_argument("--patch", type=int, default=50, help="patch size (default 50)")
    ap.add_argument("--topk", type=int, default=30, help="top K patches to export")
    ap.add_argument("--min_psnr_og", type=float, default=28.0,
                    help="filter: require PSNR(ours,gt) >= this (default 28). Set -1 to disable.")
    ap.add_argument("--min_adv", type=float, default=1.0,
                    help="filter: require advantage >= this (default 1dB). Set -1 to disable.")
    ap.add_argument("--max_best_other_psnr", type=float, default=35.0,
                    help="filter: require best other PSNR(other,gt) <= this, to enforce 'gt differs from others'. Set 1e9 to disable.")
    ap.add_argument("--out", type=str, default="", help="output dir (default: root/_patch_mining_out)")
    ap.add_argument("--w_edge", type=float, default=1.0, help="weight for edge-psnr advantage")
    ap.add_argument("--w_sharp", type=float, default=0.5, help="weight for sharpness advantage")
    ap.add_argument("--min_sharp_gain", type=float, default=0.0, help="require sharp(ours)-sharp(blur) >= this")
    args = ap.parse_args()


    root = Path(args.root).expanduser().resolve()
    gt_dir = root / args.gt
    ours_dir = root / args.ours
    if not gt_dir.is_dir():
        raise SystemExit(f"GT dir not found: {gt_dir}")
    if not ours_dir.is_dir():
        raise SystemExit(f"OURS dir not found: {ours_dir}")

    # discover model dirs
    model_dirs = [p for p in root.iterdir() if p.is_dir()]
    model_names = [p.name for p in model_dirs]
    other_models = [p.name for p in model_dirs if p.name not in (args.gt, args.ours, args.blur)]

    if not other_models:
        raise SystemExit("No other model folders found besides gt/ours.")

    print(f"[INFO] root={root}")
    print(f"[INFO] models={model_names}")
    print(f"[INFO] other_models={other_models}")

    # scan
    gt_map = scan_model_images(gt_dir)
    ours_map = scan_model_images(ours_dir)
    blur_dir = root / args.blur
    blur_map = scan_model_images(blur_dir) if blur_dir.is_dir() else {}
    print(f"[INFO] blur_dir={blur_dir} exists={blur_dir.is_dir()} matched_blur={len(blur_map)}")

    other_maps: Dict[str, Dict[str, Path]] = {}
    for m in other_models:
        other_maps[m] = scan_model_images(root / m)

    # -------- sanity check: common images across models --------
    all_maps: Dict[str, Dict[str, Path]] = {}
    all_maps[args.gt] = gt_map
    all_maps[args.ours] = ours_map
    if blur_map is not None and len(blur_map) > 0:
        all_maps[args.blur] = blur_map
    for m in other_models:
        all_maps[m] = other_maps[m]

    # each model count
    print("\n[CHECK] image counts per model:")
    for name, mp in all_maps.items():
        print(f"  - {name:12s}: {len(mp)}")

    # intersection across ALL models
    sets = [set(mp.keys()) for mp in all_maps.values()]
    common_all = set.intersection(*sets) if sets else set()
    print(f"\n[CHECK] common images across ALL models ({len(all_maps)} models): {len(common_all)}")

    # intersection across GT + OURS + OTHERS (exclude blur, since blur may be optional)
    maps_no_blur = {k: v for k, v in all_maps.items() if k != args.blur}
    sets_no_blur = [set(mp.keys()) for mp in maps_no_blur.values()]
    common_no_blur = set.intersection(*sets_no_blur) if sets_no_blur else set()
    print(f"[CHECK] common images across GT/OURS/OTHERS ({len(maps_no_blur)} models): {len(common_no_blur)}")

    # missing stats w.r.t GT
    gt_set = set(gt_map.keys())
    print("\n[CHECK] missing vs GT:")
    for name, mp in all_maps.items():
        if name == args.gt:
            continue
        s = set(mp.keys())
        missing = gt_set - s
        extra = s - gt_set
        print(f"  - {name:12s}: missing {len(missing)} | extra {len(extra)}")

    # (optional) show a few missing examples for debugging
    # show_n = 5
    # for name, mp in all_maps.items():
    #     if name == args.gt:
    #         continue
    #     missing = list(gt_set - set(mp.keys()))
    #     if missing:
    #         print(f"    examples missing in {name}: {missing[:show_n]}")
    print()
    # -------- end sanity check --------


    # match keys by intersection (must have gt and ours, at least 1 other)
    keys = sorted(set(gt_map.keys()) & set(ours_map.keys()))
    keys = [k for k in keys if any(k in other_maps[m] for m in other_models)]
    if not keys:
        raise SystemExit("No matched images found across gt/ours/others by relative path.")

    print(f"[INFO] matched images: {len(keys)}")

    patch = int(args.patch)
    cands: List[PatchCand] = []

    # for idx, rel in enumerate(tqdm(keys, desc="Scanning images", total=len(keys))):
    #     gt_path = gt_map[rel]
    #     ours_path = ours_map[rel]

    #     try:
    #         gt = imread_rgb_u8(gt_path)
    #         ours = imread_rgb_u8(ours_path)
    #     except Exception as e:
    #         print(f"[WARN] read failed: {rel} -> {e}")
    #         continue

    #     if gt.shape != ours.shape:
    #         print(f"[WARN] size mismatch ours/gt: {rel} gt={gt.shape} ours={ours.shape} (skip)")
    #         continue

    #     H, W = gt.shape[:2]
    #     coords = tile_coords(H, W, patch)
    #     if not coords:
    #         print(f"[WARN] image too small for patch {patch}: {rel} ({H}x{W})")
    #         continue

    #     # load others lazily per image
    #     others_img: Dict[str, np.ndarray] = {}
    #     for m in other_models:
    #         p = other_maps[m].get(rel, None)
    #         if p is None:
    #             continue
    #         try:
    #             im = imread_rgb_u8(p)
    #         except Exception:
    #             continue
    #         if im.shape == gt.shape:
    #             others_img[m] = im

    #     if len(others_img) == 0:
    #         continue

    #     for (y, x) in coords:
    #         gt_p = gt[y:y+patch, x:x+patch]
    #         ours_p = ours[y:y+patch, x:x+patch]

    #         psnr_og = psnr_u8(ours_p, gt_p)

    #         # compute other psnrs vs gt
    #         best_psnr = -1.0
    #         worst_psnr = float("inf")
    #         best_name = ""
    #         worst_name = ""

    #         for m, im in others_img.items():
    #             op = im[y:y+patch, x:x+patch]
    #             p = psnr_u8(op, gt_p)
    #             if p > best_psnr:
    #                 best_psnr = p
    #                 best_name = m
    #             if p < worst_psnr:
    #                 worst_psnr = p
    #                 worst_name = m

    #         if best_psnr < 0:
    #             continue

    #         # advantage: ours vs the best other (hardest competitor)
    #         # higher => ours clearly better than all others
    #         score = psnr_og - best_psnr

    #         # filters
    #         if args.min_psnr_og >= 0 and psnr_og < args.min_psnr_og:
    #             continue
    #         if args.min_adv >= 0 and score < args.min_adv:
    #             continue
    #         if best_psnr > args.max_best_other_psnr:
    #             continue

    #         cands.append(PatchCand(
    #             score=score,
    #             psnr_og=psnr_og,
    #             best_other_psnr=best_psnr,
    #             worst_other_psnr=worst_psnr,
    #             relpath=rel,
    #             y=y, x=x, h=patch, w=patch,
    #             best_other_name=best_name,
    #             worst_other_name=worst_name,
    #         ))

    #     # if (idx + 1) % 50 == 0 or (idx + 1) == len(keys):
    #     #     print(f"[INFO] processed {idx+1}/{len(keys)} images, cands={len(cands)}")
    
    
    # -------- parallel scan over images --------
    cands: List[PatchCand] = []

    # Windows/WSL/跨平台建议显式控制进程数
    max_workers = min(os.cpu_count() or 8, 16)  # 你也可以改成 8/12/24
    print(f"[INFO] Using {max_workers} processes")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(gt_map, ours_map, blur_map, other_maps, other_models,
          patch, args.min_psnr_og, args.min_adv, args.max_best_other_psnr,
          args.w_edge, args.w_sharp, args.min_sharp_gain)
    ) as ex:
        futures = [ex.submit(_process_one_image, rel) for rel in keys]
        for fu in tqdm(as_completed(futures), total=len(futures), desc="Scanning images"):
            try:
                cands.extend(fu.result())
            except Exception as e:
                # 某张图炸了不影响全局
                # 你也可以 print 更详细的 rel，但需要在 worker 里携带
                continue

    print(f"[INFO] total candidates: {len(cands)}")


    if not cands:
        raise SystemExit("No candidates found. Try relaxing thresholds: --min_psnr_og -1 --min_adv -1 --max_best_other_psnr 1e9")

    # rank by score desc, then psnr_og desc
    cands.sort(key=lambda c: (c.score, c.psnr_og), reverse=True)
    topk = cands[:int(args.topk)]

    out_dir = Path(args.out).expanduser().resolve() if args.out else (root / "_patch_mining_out")
    safe_mkdir(out_dir)
    safe_mkdir(out_dir / "patches")

    # save metadata
    meta_path = out_dir / "top_patches.tsv"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("rank\tscore_adv\tpsnr_ours_gt\tbest_other_psnr\tworst_other_psnr\trelpath\tx\ty\tw\th\tbest_other\tworst_other\n")
        for i, c in enumerate(topk):
            f.write(
                f"{i+1}\t{c.score:.4f}\t{c.psnr_og:.4f}\t{c.best_other_psnr:.4f}\t{c.worst_other_psnr:.4f}\t"
                f"{c.relpath}\t{c.x}\t{c.y}\t{c.w}\t{c.h}\t{c.best_other_name}\t{c.worst_other_name}\n"
            )

    print(f"[INFO] saving top {len(topk)} patches to: {out_dir}")

    # export patches as montage: GT | OURS | BEST_OTHER | WORST_OTHER
    for i, c in enumerate(topk, 1):
        gt = imread_rgb_u8(gt_map[c.relpath])
        ours = imread_rgb_u8(ours_map[c.relpath])
        best_other = imread_rgb_u8(other_maps[c.best_other_name][c.relpath])
        worst_other = imread_rgb_u8(other_maps[c.worst_other_name][c.relpath])

        y, x, p = c.y, c.x, c.w
        gt_p = gt[y:y+p, x:x+p]
        ours_p = ours[y:y+p, x:x+p]
        best_p = best_other[y:y+p, x:x+p]
        worst_p = worst_other[y:y+p, x:x+p]

        gt_p = draw_label(gt_p, "GT")
        ours_p = draw_label(ours_p, f"OURS  EDGEPSNR={c.psnr_og:.2f}")
        best_p = draw_label(best_p, f"BEST:{c.best_other_name}  EDGEPSNR={c.best_other_psnr:.2f}")
        worst_p = draw_label(worst_p, f"WORST:{c.worst_other_name}  EDGEPSNR={c.worst_other_psnr:.2f}")

        montage = cat_h([gt_p, ours_p, best_p, worst_p])

        # filename safe
        rel_safe = c.relpath.replace("/", "__").replace("..", "_")
        fn = out_dir / "patches" / f"{i:02d}__adv_{c.score:.2f}__{rel_safe}__x{c.x}_y{c.y}.png"

        # write back as BGR for cv2
        bgr = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(fn), bgr)

    print(f"[DONE] Exported:\n  - {meta_path}\n  - {out_dir / 'patches'}")


if __name__ == "__main__":
    main()
