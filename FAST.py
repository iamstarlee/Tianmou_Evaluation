import os
import re
import numpy as np
import cv2
from rod_event_visualization import event_visualization, event_visualization_td
import matplotlib.pyplot as plt
import re

waitkey = 0
# ---------------- 文件排序 ----------------
def sorted_npy_files(folder):
    """
    返回按 (idx, frame) 排序的 .npy 完整路径列表。
    优先匹配 'idx{idx}_frame{frame}'。如果找不到，使用文件名中最后两个数字作为 (idx, frame)。
    """
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]

    def key_fn(fname):
        b = os.path.basename(fname)
        # 1) 优先匹配显式模式 idxXXX_frameYYY
        m = re.search(r'idx(\d+)_frame(\d+)', b)
        if m:
            idx = int(m.group(1))
            frame = int(m.group(2))
            return (idx, frame, b)   # 加上 b 保证同 idx/frame 时有确定顺序

        # 2) 没有匹配到就提取所有数字，使用最后两个数字作为 (idx, frame)
        nums = re.findall(r'(\d+)', b)
        if len(nums) >= 2:
            idx = int(nums[-2])
            frame = int(nums[-1])
            return (idx, frame, b)
        if len(nums) == 1:
            return (int(nums[0]), -1, b)

        # 3) 没有数字，放到最后
        return (float('inf'), float('inf'), b)

    files.sort(key=key_fn)
    return [os.path.join(folder, f) for f in files]


def filter_keypoints_distance(keypoints, min_distance=5):
    """过滤关键点，确保任意两个关键点距离至少 min_distance"""
    if len(keypoints) == 0:
        return []

    pts = np.array([kp.pt for kp in keypoints])
    mask = np.ones(len(pts), dtype=bool)

    for i in range(len(pts)):
        if not mask[i]:
            continue
        dists = np.linalg.norm(pts - pts[i], axis=1)
        mask = mask & ((dists > min_distance) | (np.arange(len(pts)) == i))

    return [keypoints[i] for i in range(len(pts)) if mask[i]]

# ---------------- FAST 检测 ----------------
def detect_fast_points(img, threshold=50, min_response=20, roi_ratio=0.45):
    """
    FAST 检测关键点
    :param img: 灰度图 uint8
    :param threshold: FAST 对比阈值
    :param min_response: 关键点响应阈值
    :return: keypoints
    """

    h, w = img.shape

    # 创建圆形 ROI
    if roi_ratio:
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2 - 1, h // 2 - 1)
        radius = int(min(h, w) * roi_ratio)
        cv2.circle(mask, center, radius, 255, -1)
    else:
        mask = None

    fast = cv2.FastFeatureDetector_create(
        threshold=threshold,
        nonmaxSuppression=True,
        type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
    )

    keypoints = fast.detect(img, mask)
    keypoints = filter_keypoints_distance(keypoints, min_distance=8)  # 8像素以上
    keypoints = [kp for kp in keypoints if kp.response >= min_response]
    return keypoints

# ---------------- 主程序 ----------------
folder_l = 'multi/1200/Poisson_denoised_out'
name = "FAST"
files_l = sorted_npy_files(folder_l)

corner_counts = []

if "cone" in folder_l:
    step = 2
else:
    step = 25

for idx, file_l in enumerate(files_l[::step]):  
    img = cv2.resize(np.load(file_l), (320, 160))

    # 转 uint8
    if img.dtype != np.uint8:
        if folder_l.endswith("td"):
            vis = event_visualization(img * 10)
            img = ((img + 50) * 2).clip(0, 255).astype(np.uint8)
            # gray = cv2.medianBlur(img, 3)
            gray = img
        else:
            vis = event_visualization(img * 10)
            img = ((img + 50) * 2).clip(0, 255).astype(np.uint8)
            gray = img
        # gray = img

    # 单通道灰度
    if img.ndim == 3:
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (320, 160))
        vis = gray
    # else:
    #     gray = img
    #     vis = img

    # FAST 检测关键点
    keypoints = detect_fast_points(gray, threshold=30, min_response=10, roi_ratio=0.46)
    corner_counts.append(len(keypoints))

    # 绘制关键点
    # out_img = cv2.drawKeypoints(vis, keypoints, None, color=(0, 165 ,255))
    cv2.imshow("FAST Keypoints", vis)
    key = cv2.waitKey(waitkey) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# ---------------- 滑动均值 + 不确定性 ----------------
corner_counts = np.array(corner_counts)
if folder_l.startswith("mu"):
    gt = 17
else:
    gt = 10
frame_indices = np.arange(len(corner_counts))

# 滑动窗口均值和方差
if "cone" in folder_l:  # \b 表示单词边界
    window = 4
else:
    window = 4
    
mean_counts = np.full_like(corner_counts, np.nan, dtype=float)
uncertainty = np.full_like(corner_counts, np.nan, dtype=float)

for i in range(window-1, len(corner_counts)):
    window_values = corner_counts[i-window+1:i+1]
    mean_counts[i] = np.mean(window_values)
    uncertainty[i] = np.std(window_values)

# 统一纵轴范围
y_min = min(np.nanmin(mean_counts - uncertainty), gt) - 1
y_max = max(np.nanmax(mean_counts + uncertainty), gt) + 1

# 绘制
import matplotlib.pyplot as plt
import os

# 绘制
plt.figure(figsize=(4,3), dpi=150)
plt.plot(frame_indices, mean_counts, label="Mean", linewidth=1.2)
plt.fill_between(frame_indices,
                 mean_counts - uncertainty, mean_counts + uncertainty,
                 alpha=0.15, label='±1σ')
plt.hlines(gt, 0, len(corner_counts)-1, colors='orange', linestyles='--', linewidth=1.2, label=f'GT={gt}')

plt.xlabel("Frame", fontsize=8)
plt.ylabel("Corners", fontsize=8)
plt.ylim(y_min, y_max)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
# plt.legend(fontsize=7, frameon=False, loc="best")
plt.tight_layout()

# 保存为 svg
save_name = f"{folder_l.replace('/', '_')}_{name}.svg"
save_name = "result/" + save_name
plt.savefig(save_name, format="svg", dpi=300)

# 显示
plt.show()

print(f"保存图像为: {save_name}")
