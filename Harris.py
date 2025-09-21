import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from rod_event_visualization import event_visualization
from scores_compute import *

def sorted_npy_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    def key_fn(fname):
        b = os.path.basename(fname)
        m = re.search(r'idx(\d+)_frame(\d+)', b)
        if m:
            return (int(m.group(1)), int(m.group(2)), b)
        nums = re.findall(r'(\d+)', b)
        if len(nums) >= 2:
            return (int(nums[-2]), int(nums[-1]), b)
        if len(nums) == 1:
            return (int(nums[0]), -1, b)
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

def detect_keypoints(img, detector_type="HARRIS", use_roi=False, roi_ratio=0.41,
                     harris_params=None, fast_params=None, orb_params=None):
    """
    通用关键点检测 HARRIS / FAST / ORB
    """
    gray = img.copy()
    h, w = gray.shape

    # ROI mask
    mask = None
    if use_roi:
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2 - 1, h // 2 - 1)
        radius = int(min(h, w) * roi_ratio)
        cv2.circle(mask, center, radius, 255, -1)

    keypoints = []

    if detector_type.upper() == "HARRIS":
        if harris_params is None:
            harris_params = dict(maxCorners=100,
                     qualityLevel=0.03,
                     minDistance=10,
                     useHarrisDetector=True,
                     k=0.06)
        corners = cv2.goodFeaturesToTrack(gray, mask=mask, **harris_params)
        if corners is not None:
            for c in corners:
                x, y = c.ravel()
                keypoints.append(cv2.KeyPoint(x=float(x), y=float(y), size=3))

    elif detector_type.upper() == "FAST":
        if fast_params is None:
            fast_params = dict(threshold=25, nonmaxSuppression=True,
                               type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
        fast = cv2.FastFeatureDetector_create(**fast_params)
        keypoints = fast.detect(gray, mask=mask)

    elif detector_type.upper() == "ORB":
        if orb_params is None:
            
            orb_params = dict(nfeatures=200, scaleFactor=1.2, nlevels=8,
                              edgeThreshold=31, firstLevel=0, WTA_K=2,
                              scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31,
                              fastThreshold=40)
        orb = cv2.ORB_create(**orb_params)
        keypoints = orb.detect(gray, mask=mask)

    else:
        raise ValueError(f"未知 detector_type: {detector_type}")
    
    keypoints = filter_keypoints_distance(keypoints, min_distance=8)  # 8像素以上
    if detector_type == "HARRIS":
        # 计算 confidence
        responses, trackable = compute_shitomasi_confidence(
            gray, corners, blockSize=5, ksize=3, ratio_thresh=0.4, normalize=False
        )
        # responses = compute_harris_confidence(gray, corners)
        # keypoints   = [kp for kp, t in zip(keypoints, trackable) if t]
        # responses  = [resp for resp, t in zip(responses, trackable) if t]

    elif detector_type == "FAST":
        keypoints = [kp for kp in keypoints if kp.response >= 20]
        responses = compute_fast_scores(gray, keypoints, threshold=20)

    elif detector_type == "ORB":
        responses = np.array([kp.response * 10 ** 6 for kp in keypoints], dtype=float)

    # 计算数量、均值、方差
    num_kp = len(keypoints)
    if responses is not None and len(responses) > 0:
        mean_conf = float(np.mean(responses))
        var_conf = float(np.var(responses))
    else:
        mean_conf = 0.0
        var_conf = 0.0

    return keypoints, mean_conf, num_kp

# ---------------- 主程序 ----------------
# Poisson_denoised_out, cone_output_cam0, Ixy, denoised_output_cam0/td#
folder_l = 'multi/900/denoised_output_cam0/td'
detector_type = "HARRIS"   # <<< "HARRIS", "FAST", "ORB"
files_l = sorted_npy_files(folder_l)

# ---------------- 初始化列表 ----------------
mean_confs = []
num_kps = []
var_num_kps = []
var_mean_confs = []

corner_counts = []
scores = []

step = 25 if "cone" not in folder_l else 2
waitkey = 0

for idx, file_l in enumerate(files_l[::step]):
    img = cv2.resize(np.load(file_l), (320, 160))

    if img.dtype != np.uint8:
        vis = event_visualization(img * 10)
        gray = ((img + 50) * 2).clip(0, 255).astype(np.uint8)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        vis = gray
    else:
        gray = img
        vis = gray

    # --- 使用通用检测器 ---
    kps, mean_conf, num_kp = detect_keypoints(gray, detector_type=detector_type,
                                              use_roi=True, roi_ratio=0.40)

    # --- 更新列表 ---
    mean_confs.append(mean_conf)
    num_kps.append(num_kp)

    # 这里可以计算滑动窗口方差或单帧方差
    # 我们假设用最近5帧计算方差
    window = 4
    if len(num_kps) >= window:
        var_num_kps.append(np.var(num_kps[-window:]))
        var_mean_confs.append(np.var(mean_confs[-window:]))
    else:
        var_num_kps.append(0.0)
        var_mean_confs.append(0.0)

    # 绘制关键点
    out_img = cv2.drawKeypoints(vis, kps, None, color=(30, 144, 255))
    cv2.imshow(f"{detector_type} Corners", out_img)
    key = cv2.waitKey(waitkey) & 0xFF
    if key == ord('q'):
        break

# ---------------- 写入 txt 文件 ----------------
# 使用 folder_l 名称作为文件名
txt_name = detector_type + "_" + folder_l.replace('/', '_') + ".txt"
os.makedirs("result", exist_ok=True)
txt_path = os.path.join("result", txt_name)

with open(txt_path, 'w') as f:
    f.write("frame_idx\tmean_conf\tnum_kp\tvar_num_kp\tvar_mean_conf\n")
    for i in range(len(mean_confs)):
        f.write(f"{i}\t{mean_confs[i]:.4f}\t{num_kps[i]}\t{var_num_kps[i]:.4f}\t{var_mean_confs[i]:.4f}\n")

print(f"保存结果到: {txt_path}")

# print(f"{detector_type} mean scores: {np.mean(scores)}")

cv2.destroyAllWindows()

# ---------------- 绘制折线图 ----------------
corner_counts = np.array(corner_counts)
gt = 17 if folder_l.startswith("mu") else 10
frame_indices = np.arange(len(corner_counts))

# window = 4
# mean_counts = np.full_like(corner_counts, np.nan, dtype=float)
# uncertainty = np.full_like(corner_counts, np.nan, dtype=float)
# for i in range(window-1, len(corner_counts)):
#     window_values = corner_counts[i-window+1:i+1]
#     mean_counts[i] = np.mean(window_values)
#     uncertainty[i] = np.std(window_values)

# y_min = min(np.nanmin(mean_counts - uncertainty), gt) - 1
# y_max = max(np.nanmax(mean_counts + uncertainty), gt) + 1

# plt.figure(figsize=(4,3), dpi=150)
# plt.plot(frame_indices, mean_counts, label="Mean", linewidth=1.2)
# plt.fill_between(frame_indices, mean_counts - uncertainty, mean_counts + uncertainty, alpha=0.15, label='±1σ')
# plt.hlines(gt, 0, len(corner_counts)-1, colors='orange', linestyles='--', linewidth=1.2, label=f'GT={gt}')
# plt.xlabel("Frame", fontsize=8)
# plt.ylabel("Corners", fontsize=8)
# plt.ylim(y_min, y_max)
# plt.xticks(fontsize=7)
# plt.yticks(fontsize=7)
# plt.tight_layout()

# save_name = f"result/{folder_l.replace('/', '_')}_{detector_type}.svg"
# plt.savefig(save_name, format="svg", dpi=300)
# plt.show()

# print(f"保存图像为: {save_name}")
