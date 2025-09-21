import cv2
import numpy as np
import os
import re
from rod_event_visualization import event_visualization

# 全局变量
points_img2 = []
num_points = 3  # 仿射变换需要的点数
scale_factor = 0.05

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

def select_points(event, x, y, flags, param):
    """鼠标回调函数，用于选点"""
    global points_img2
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_img2) < num_points:
            points_img2.append((x, y))
            print(f"Image 2 - Point {len(points_img2)}: ({x}, {y})")

def main():
    global points_img2

    # 设置输入文件夹、基准图像、输出文件夹
    input_folder = "multi/900/denoised_output_cam0/td"  # npy 文件夹
    input_folder = sorted_npy_files(input_folder)
    reference_image_path = "multi_gt.JPG"  # JPG 文件
    output_folder = "warp_M/"

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 加载基准图像 (JPG)
    img2 = cv2.imread(reference_image_path)
    height, width, _ = img2.shape
    # img2 = cv2.flip(cv2.resize(img2, (max(1, int(width*scale_factor)), max(1, int(height*scale_factor)))), 1)
    img2 = cv2.flip((cv2.resize(img2, (160, 160))), 1)
    if img2 is None:
        print(f"Error: Could not load reference image '{reference_image_path}'.")
        return

    # 显示基准图像并选点
    cv2.namedWindow("Reference Image")
    cv2.setMouseCallback("Reference Image", select_points)

    while len(points_img2) < num_points:
        temp_img2 = img2.copy()
        for pt in points_img2:
            cv2.circle(temp_img2, pt, 5, (0, 0, 255), -1)
        cv2.imshow("Reference Image", temp_img2)
        cv2.waitKey(1)

    cv2.destroyWindow("Reference Image")

    # 设置起始索引，比如从第 10 张开始
    start_idx = 0

    # 遍历 npy 文件
    if "cone" in input_folder[0]:
        step = 2
        start_idx *= 2
    else:
        start_idx *= 25
        step = 25

    # 确保 start_idx 不超过文件数量
    start_idx = min(start_idx, len(input_folder) - 1)

    for file_name in input_folder[start_idx::step]:
        if not file_name.lower().endswith('.npy'):
            continue
        input_path = file_name

        # 加载当前图像 (npy)
        img1 = cv2.resize(np.load(input_path), (320, 160))
        if img1 is None:
            print(f"Error: Could not load image '{input_path}'. Skipping.")
            continue

        # 如果是单通道或浮点类型，将其转为 uint8 彩色图像用于显示和选点
        if img1.ndim == 2:
            img1_disp = event_visualization(img1 * 10, 1)
            # img1_disp = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            img1_disp = img1.astype(np.uint8)

        # 选择图像1上的三个点
        points_img1 = []
        def select_points_img1(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points_img1) < num_points:
                points_img1.append((x, y))
                print(f"Image 1 - Point {len(points_img1)}: ({x}, {y})")

        cv2.namedWindow("Input Image")
        cv2.setMouseCallback("Input Image", select_points_img1)

        while len(points_img1) < num_points:
            temp_img1 = img1_disp.copy()
            for pt in points_img1:
                cv2.circle(temp_img1, pt, 5, (0, 0, 255), -1)
            cv2.imshow("Input Image", temp_img1)
            cv2.waitKey(1)

        cv2.destroyWindow("Input Image")

        # 计算仿射变换矩阵
        points_img1_np = np.array(points_img1, dtype=np.float32)
        points_img2_np = np.array(points_img2, dtype=np.float32)

        M = cv2.getAffineTransform(points_img1_np, points_img2_np)
        print(f"Computed Affine Transformation Matrix for {file_name}:")
        print(M)

        # ---- 保存仿射矩阵 ----
        rel_path = os.path.splitext(file_name)[0] + "_M.npy"  # 加个后缀避免混淆
        output_path = os.path.join(output_folder, rel_path)

        # 确保子目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        np.save(output_path, M)
        print(f"Saved affine matrix M to '{output_path}'.")

        # # 计算仿射变换矩阵
        # points_img1_np = np.array(points_img1, dtype=np.float32)
        # points_img2_np = np.array(points_img2, dtype=np.float32)

        # M = cv2.getAffineTransform(points_img1_np, points_img2_np)
        # print(f"Computed Affine Transformation Matrix for {file_name}:")
        # print(M)

        # # 应用仿射变换
        # height1, width1 =  img1.shape[:2]
        # warped_img1 = cv2.warpAffine(img1, M, (160, 160))
        # # cv2.imshow("warped", warped_img1)
        # # cv2.waitKey(0)

        # # 保存结果为 npy
        # # 保存结果为 npy
        # rel_path = os.path.splitext(file_name)[0] + ".npy"
        # output_path = os.path.join(output_folder, rel_path)

        # # 确保子目录存在
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # np.save(output_path, warped_img1)
        # print(f"Saved warped image to '{output_path}'.")

    print("All images processed successfully.")

if __name__ == "__main__":
    main()
