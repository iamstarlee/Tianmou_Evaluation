import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment  # 匈牙利算法
from rod_event_visualization import event_visualization
import re

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

def compute_keypoint_matches(gt_keypoints, input_keypoints, max_distance=30):
    distances = np.linalg.norm(gt_keypoints[:, np.newaxis, :] - input_keypoints[np.newaxis, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(distances)
    matches, distances_matched = [], []
    for row, col in zip(row_ind, col_ind):
        distance = distances[row, col]
        if distance <= max_distance:
            matches.append((row, col))
            distances_matched.append(distance)
    return matches, distances_matched

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

def extract_keypoints_from_npy_with_mask(npy_path, roi_ratio=0.3, method="Harris"):
    """
    从 npy 图像中提取关键点，使用圆形 ROI 掩码
    :param npy_path: npy 文件路径
    :param roi_ratio: ROI 半径占图像最小边的比例
    :param method: 检测方法 ("Harris", "FAST", "ORB")
    :return: keypoints (n x 2), 图像 (BGR)
    """
    if "gt" in npy_path:
        img = cv2.resize(np.load(npy_path), (160, 160))
    else:
        img = cv2.resize(np.load(npy_path), (320, 160))

    if "gt" in npy_path:
        img = cv2.flip(img, 1)

        # 高斯模糊 + LoG
        blurred = cv2.GaussianBlur(img, (5, 5), 0.0)
        log_img = cv2.Laplacian(blurred, cv2.CV_64F)
        log_img = cv2.convertScaleAbs(log_img)
        img = cv2.bitwise_not(log_img)

    if img is None:
        print(f"无法加载图像：{npy_path}")
        return None, None

    # 转为 uint8
    if img.dtype != np.uint8:
        vis = event_visualization(img * 10)
        gray = ((img + 50) * 2).clip(0, 255).astype(np.uint8)

    # 灰度图
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        vis = img
    # else:
    #     gray = img
    #     vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape

    # 创建圆形掩码
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = int(min(h, w) * roi_ratio)
    cv2.circle(mask, center, radius, 255, -1)

    # vis_with_roi = vis.copy() if vis.ndim == 3 else cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    # cv2.circle(vis_with_roi, center, radius, (0, 255, 0), 2)

    # cv2.imshow("ROI Visualization", vis_with_roi)
    # cv2.waitKey(0)

    # 检测关键点
    keypoints = []
    if method == "Harris":
        harris_params = dict(maxCorners=100,
                     qualityLevel=0.03,
                     minDistance=10,
                     useHarrisDetector=True,
                     k=0.06)
        keypoints = cv2.goodFeaturesToTrack(gray, mask=mask, **harris_params)
        if keypoints is not None:
            keypoints = [cv2.KeyPoint(float(x), float(y), 3) for [[x, y]] in keypoints]

    elif method == "FAST":
        fast_params = dict(threshold=30, nonmaxSuppression=True,
                               type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
        fast = cv2.FastFeatureDetector_create(**fast_params)
        keypoints = fast.detect(gray, mask=mask)

    elif method == "ORB":
        orb_params = dict(
            nfeatures=250,         # 提取更多特征点，避免遗漏
            scaleFactor=1.2,
            nlevels=8,             # 减少金字塔层数，小图不需要太多层
            edgeThreshold=9,       # 缩小边缘跳过范围
            patchSize=31,          # 小图用更小的 patch
            fastThreshold=27        # 降低 FAST 阀值
        )
        orb = cv2.ORB_create(**orb_params)
        keypoints = orb.detect(gray, mask=mask)
    keypoints = filter_keypoints_distance(keypoints, 10)

    # 转换为 numpy (n x 2)
    if keypoints is not None and len(keypoints) > 0:
        keypoints_global = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        return keypoints_global, vis
    else:
        return np.array([], dtype=np.float32).reshape(0, 2), vis


def draw_matches(gt_image, input_image, gt_keypoints, input_keypoints, matches, distances, threshold=10):
    """
    可视化匹配的关键点：
      - TP (True Positive): 橙色 (GT 和 Input 成功匹配，距离 <= 阈值)
      - FP (False Positive): 蓝色 (Input 中多出来的点，没有对应 GT)
      - FN (False Negative): 红色 (GT 中有，但 Input 没有匹配到)
    """
    h1, w1, _ = gt_image.shape
    h2, w2 = input_image.shape[:2]
    height = max(h1, h2)
    width = w1 + w2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:h1, :w1] = gt_image
    canvas[:h2, w1:w1+w2] = input_image

    # 显示关键点数量
    # cv2.putText(canvas, f"GT KPs: {len(gt_keypoints)}", (10, 30), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    # cv2.putText(canvas, f"Input KPs: {len(input_keypoints)}", (w1+10, 30), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    matched_gt = set()
    matched_input = set()
    TP, FP, FN = 0, 0, 0

    # 匹配的点对
    for (gt_idx, input_idx), dist in zip(matches, distances):
        gt_point = tuple(map(int, gt_keypoints[gt_idx]))
        input_point = tuple(map(int, input_keypoints[input_idx]))
        input_point_shifted = (input_point[0] + w1, input_point[1])

        if dist <= threshold:
            color = (0, 165, 255)  # 橙色 → TP
            TP += 1
        else:
            color = (0, 0, 255)    # 红色 → 错误匹配视作 FN
            FN += 1

        matched_gt.add(gt_idx)
        matched_input.add(input_idx)

        cv2.circle(canvas, gt_point, 4, color, -1)
        cv2.circle(canvas, input_point_shifted, 4, color, -1)
        # cv2.line(canvas, gt_point, input_point_shifted, color, 1)

    # FP = Input 中未匹配的点
    for idx, pt in enumerate(input_keypoints):
        if idx not in matched_input:
            pt_shifted = (int(pt[0]) + w1, int(pt[1]))
            cv2.circle(canvas, pt_shifted, 4, (255, 0, 0), -1)  # 蓝色
            FP += 1

    # FN = GT 中未匹配的点
    for idx, pt in enumerate(gt_keypoints):
        if idx not in matched_gt:
            cv2.circle(canvas, tuple(map(int, pt)), 3, (0, 0, 255), -1)  # 红色
            FN += 1

    # 显示统计信息
    stats = f"TP: {TP}  FP: {FP}  FN: {FN}"
    cv2.putText(canvas, stats, (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0,255,0), 2)

    # 可视化
    cv2.namedWindow("Keypoint Matches", cv2.WINDOW_NORMAL)
    cv2.imshow("Keypoint Matches", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return TP, FP, FN


def main(gt_npy_path, input_npy_folder, M_folder, av):
    # 提取 GT npy 
    method = "Harris"
    gt_keypoints, gt_image = extract_keypoints_from_npy_with_mask(
        gt_npy_path, roi_ratio=0.42, method=method
    )

    if gt_keypoints is None or len(gt_keypoints) == 0:
        print("GT 图像没有检测到关键点")
        return

    # 输出目录：warp_results/multi/150/ （和原来分开）
    base_name = os.path.basename(os.path.normpath(input_npy_folder))
    base_name = method + "_" + base_name
    stats_dir = os.path.join("warp_results", "multi", f"{av}")
    os.makedirs(stats_dir, exist_ok=True)
    output_txt = os.path.join(stats_dir, f"{base_name}.txt")

    with open(output_txt, "w") as f:
        f.write("filename,TP,FP,FN,avg_distance\n")

        # 遍历输入 npy 图像
        # 遍历 npy 文件
        if "cone" in input_npy_folder:
            step = 2
            st_step = 27
        else:
            step = 25

        if "Ixy" in input_npy_folder:
            st_step = 14

        if "td" in input_npy_folder:
            st_step = 34
        
        if "Poisson" in input_npy_folder:
            st_step = 31

        if "1200" in input_npy_folder:
            st_step += 1

        for filename in sorted_npy_files(input_npy_folder)[::step][2:10]:
            if not filename.lower().endswith(".npy"):
                continue
            input_path = filename

            # 对应的 M 文件
            name_no_ext = os.path.splitext(filename)[0][st_step:]
            M_path = os.path.join(M_folder, name_no_ext + "_M.npy")
            if not os.path.exists(M_path):
                print(f"未找到 M 文件: {M_path}")
                continue
            M = np.load(M_path)

            print(f"处理图像: {input_path}，使用矩阵 {M_path}")

            # 1. 提取原图关键点
            input_keypoints, input_image = extract_keypoints_from_npy_with_mask(
                input_path, roi_ratio=0.40, method=method
            )
            if input_keypoints is None or len(input_keypoints) == 0:
                print(f"输入图像 {input_path} 没有检测到关键点")
                continue

            h, w = 160, 160   # warp 输出尺寸（和 GT 对齐）
            warped_image = cv2.warpAffine(
                input_image, 
                M, 
                (w, h), 
                flags=cv2.INTER_LINEAR,          # 插值方式
                borderMode=cv2.BORDER_CONSTANT,  # 常数填充
                borderValue=(255, 255, 255)      # 填充为白色 (三通道)
            )


            input_keypoints_warped = cv2.transform(
                input_keypoints.reshape(-1, 1, 2), M
            ).reshape(-1, 2)

            # 3. 匹配
            matches, distances = compute_keypoint_matches(gt_keypoints, input_keypoints_warped)
            print(f"匹配的关键点数量: {len(matches)}")

            for i, ((gt_idx, input_idx), dist) in enumerate(zip(matches, distances)):
                print(f"匹配 {i+1}: GT 点 {gt_idx} -> 输入点 {input_idx}, 距离: {dist:.2f}")

            # 4. 可视化 & 统计
            TP, FP, FN = draw_matches(
                gt_image, warped_image, gt_keypoints, input_keypoints_warped,
                matches, distances, threshold=10
            )
            avg_distance = np.mean(distances) if len(distances) > 0 else -1

            # 写入结果
            f.write(f"{filename},{TP},{FP},{FN},{avg_distance:.2f}\n")

    print(f"统计结果已保存到 {output_txt}")


if __name__ == "__main__":
    av = 300
    gt_npy_path = "multi_gt.npy"  
    input_npy_folder = f"multi/{av}/Ixy"  # warp 前原始图像
    M_folder = f"warp_M/multi/{av}/Ixy"  # 保存 M.npy 的目录
    main(gt_npy_path, input_npy_folder, M_folder, av)

    # Poisson_denoised_out, cone_output_cam0, Ixy, denoised_output_cam0/td#

