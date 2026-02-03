import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ================= 配置区域 =================
# 1. 图片文件夹路径 (请确认这是你要跑的文件夹)
# FOLDER_PATH = r"F:\Learn\3_TestIMG\IMG_viewer\save\default_37_99_1f00c4d5_18_r2_1.2_dy_0"
FOLDER_PATH = r"F:\Learn\3_TestIMG\IMG_viewer\save\default_37_99_1f00053b_03_r2_1.2_dy_1"

# 2. 你的模型名字 (已改为 ex)
TARGET_NAME = "ex" 

# 3. GT 的名字
GT_NAME = "gt"

# 4. 结果保存的子文件夹名字
OUTPUT_DIR_NAME = "Best_Results"

# 5. 区域设置
CORE_W = 64
CORE_H = 36
VIEW_PADDING = 20  # 向外延伸像素
TOP_K = 8          # 找8个备选
# ===========================================

def cv2_imread_win(file_path):
    """解决 Windows 中文路径读取问题"""
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"读取失败: {file_path}")
        return None

def get_advantage_map(images, titles):
    """
    计算【优势图】：(其他模型的平均误差) - (目标模型的误差)
    目标: ex 误差极小，competitors 误差较大
    """
    gt_img = None
    target_img = None
    competitors = []
    
    # 转灰度
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) for img in images]
    
    # 分类图片
    for i, title in enumerate(titles):
        t_lower = title.lower()
        if GT_NAME.lower() in t_lower:
            gt_img = gray_images[i]
        elif TARGET_NAME.lower() == t_lower: # 精确匹配 ex
            target_img = gray_images[i]
        # 兼容文件名类似 ex_... 的情况
        elif title.lower().startswith(TARGET_NAME.lower() + "_"):
            target_img = gray_images[i]
        else:
            competitors.append(gray_images[i])
            
    if gt_img is None:
        raise ValueError(f"未找到 GT 图片 (文件名需包含 '{GT_NAME}')")
    if target_img is None:
        raise ValueError(f"未找到目标模型图片 (文件名需包含 '{TARGET_NAME}')")
    if not competitors:
        raise ValueError("未找到对比模型")

    print(f"  -> GT已锁定")
    print(f"  -> 主角 (Ours): {TARGET_NAME}")
    print(f"  -> 对手 (Competitors): {len(competitors)} 个")

    # 计算误差 (L1 Loss)
    diff_target = np.abs(target_img - gt_img)
    
    diff_others_sum = np.zeros_like(gt_img)
    for comp in competitors:
        diff_others_sum += np.abs(comp - gt_img)
    diff_others_avg = diff_others_sum / len(competitors)
    
    # 优势分 = 别人的平均错误 - 我的错误
    advantage_map = diff_others_avg - diff_target
    
    return advantage_map

def find_best_patches(score_map, patch_h, patch_w, k=5):
    h, w = score_map.shape
    candidates = []
    
    # 均值滤波算区域分
    window_sum = cv2.boxFilter(score_map, -1, (patch_w, patch_h), normalize=False)
    temp_map = window_sum.copy()
    
    mask_h, mask_w = int(patch_h * 1.2), int(patch_w * 1.2)
    
    for _ in range(k):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp_map)
        
        if max_val <= 0:
            break
            
        cx, cy = max_loc
        x = int(cx - patch_w / 2)
        y = int(cy - patch_h / 2)
        
        x = max(0, min(x, w - patch_w))
        y = max(0, min(y, h - patch_h))
        
        candidates.append((x, y, max_val))
        
        # NMS 抑制
        y1 = max(0, y - mask_h // 2)
        y2 = min(h, y + mask_h + mask_h // 2)
        x1 = max(0, x - mask_w // 2)
        x2 = min(w, x + mask_w + mask_w // 2)
        temp_map[y1:y2, x1:x2] = -99999

    return candidates

def plot_details(images, titles, region, padding, rank, save_dir):
    x_core, y_core, score = region
    h_img, w_img, _ = images[0].shape
    
    # 计算裁剪坐标
    x1 = max(0, x_core - padding)
    y1 = max(0, y_core - padding)
    x2 = min(w_img, x_core + CORE_W + padding)
    y2 = min(h_img, y_core + CORE_H + padding)
    
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.05, hspace=0.2, top=0.85)
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            crop = images[i][y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ax.imshow(crop)
            
            t = titles[i]
            t_lower = t.lower()
            
            # 标题着色逻辑
            if GT_NAME.lower() in t_lower:
                color = 'green'
                font_w = 'bold'
                t = "GT (Ref)"
            # 判断是否是 ex
            elif t_lower == TARGET_NAME.lower() or t_lower.startswith(TARGET_NAME.lower() + "_"):
                color = 'red'
                font_w = 'bold'
                t = f"{TARGET_NAME} (Ours)"
            else:
                color = 'black'
                font_w = 'normal'
                
            ax.set_title(t, fontsize=11, fontweight=font_w, color=color)
            
            # 绘制红框 (仅标示核心计算区域)
            rx = x_core - x1
            ry = y_core - y1
            rect = Rectangle((rx, ry), CORE_W, CORE_H, 
                             linewidth=1.5, edgecolor='red', linestyle='--', facecolor='none')
            ax.add_patch(rect)
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.suptitle(f"Rank #{rank}: {TARGET_NAME} vs Others\nAdvantage Score: {score:.0f}", fontsize=16, fontweight='bold')
    
    filename = f'Best_Rank_{rank}.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"  -> 已保存: {filename}")
    plt.close(fig)

def main():
    print(f"正在读取: {FOLDER_PATH}")
    
    # 0. 创建保存目录
    save_dir = os.path.join(FOLDER_PATH, OUTPUT_DIR_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建保存目录: {save_dir}")
    else:
        print(f"保存目录已存在: {save_dir}")

    # 1. 搜索图片
    files = sorted(glob.glob(os.path.join(FOLDER_PATH, "*.[jp][pn]g")))
    if len(files) < 3:
        print("错误: 图片数量不足")
        return

    images = []
    titles = []
    for f in files:
        img = cv2_imread_win(f)
        if img is not None:
            images.append(img)
            # 提取文件名开头作为 title
            titles.append(os.path.basename(f).split('_')[0])

    # 2. 计算优势
    try:
        adv_map = get_advantage_map(images, titles)
    except ValueError as e:
        print(f"错误: {e}")
        return

    # 3. 找点
    candidates = find_best_patches(adv_map, CORE_H, CORE_W, k=TOP_K)
    
    if not candidates:
        print(f"未找到 {TARGET_NAME} 明显优于其他模型的地方。")
        return

    print(f"找到 {len(candidates)} 个最佳区域，生成图片中...")

    # 4. 绘图并保存到新文件夹
    for i, region in enumerate(candidates):
        plot_details(images, titles, region, VIEW_PADDING, i+1, save_dir)

    print(f"\n全部完成！结果已保存在: {save_dir}")

if __name__ == "__main__":
    main()