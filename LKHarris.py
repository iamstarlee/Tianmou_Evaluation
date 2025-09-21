import numpy as np
import cv2
import time
import re
import os
from rod_event_visualization import event_visualization

def sorted_npy_list(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    def key_fn(x):
        m = re.search(r"idx(\d+)_frame(\d+)", x)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        else:
            return (float('inf'), float('inf'))
    files.sort(key=key_fn)
    return files

image_folder = 'multi_300/denoised_output_cam0/td'

files = sorted_npy_list(image_folder)                 # 返回排序好的文件名（不带路径）
image_files = [os.path.join(image_folder, f) for f in files]  # 再拼成完整路径
video_name = 'rgbLK.mp4'
fps = 120  # 帧率

trajectory_len = 10
detect_interval = 200
trajectories = []
frame_idx = 0

lk_params = dict(winSize  = (17, 17),
                maxLevel = 3,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 100,
                    qualityLevel = 0.03,
                    minDistance = 10,
                    useHarrisDetector=True, 
                    k=0.06)

# 创建视频编写器
video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (320, 160))

for i in range(1, 5000, 1):
    # start time to calculate FPS
    start = time.time()

    # 读取 npy 数据
    prev_data = np.load(image_files[i-1])
    frame_data = np.load(image_files[i])

    prev_data_or = cv2.resize(prev_data, (320, 160))
    frame_data_or = cv2.resize(frame_data, (320, 160))

    # roi_img, mask = apply_circular_roi(prev_data_or, radius_ratio=0.45, offset_x=-15, offset_y=0)

    prev_data = (prev_data_or + 50) * 2
    frame_data = (frame_data_or + 50) * 2
    # prev_data = prev_data.astype(np.uint8)
    # frame_data = frame_data.astype(np.uint8)

    # 转灰度
    if prev_data.ndim == 3 and prev_data.shape[2] == 3:
        prev_gray = cv2.cvtColor(prev_data_or.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        img = prev_data_or.astype(np.uint8)
        img_viz = img
    else:
        prev_gray = prev_data_or.astype(np.uint8)
        img = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)  # 保证画彩色轨迹

    if frame_data.ndim == 3 and frame_data.shape[2] == 3:
        frame_gray = cv2.cvtColor(frame_data_or.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        img_viz = event_visualization(prev_data_or * 20)
        frame_gray = frame_data_or.astype(np.uint8)

    # Calculate optical flow for a sparse feature set using LK
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

        # 一致性检查
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        # 光流速度（欧氏距离）
        flow_dist = np.linalg.norm(p1.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)

        new_trajectories = []
        for trajectory, (x, y), good_flag, dist in zip(trajectories, p1.reshape(-1, 2), good, flow_dist):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)

            # 只有当速度大于 2 px/frame 才画
            cv2.circle(img_viz, (int(x), int(y)), 3, (0, 0, 255), -1)

        trajectories = new_trajectories

        # 画轨迹（也可以只画高速的点）
        cv2.polylines(img_viz, [np.int32(trajectory) for trajectory in trajectories], 
                      False, (30, 144, 255), thickness=1)
        # cv2.putText(img, 'track count: %d' % len(trajectories), 
        #             (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)


    # Update interval - When to update and detect new features
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        last_points = []
        # Lastest point in latest trajectory
        # for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
        #     cv2.circle(mask, (x, y), 5, 0, -1)
        #     last_points.append([x, y])
        
        if len(last_points) > 3:
            mean = np.mean(np.array(last_points), axis=0)
            distances = np.linalg.norm(last_points - mean, axis=1)

            # 定义一个阈值（你可以根据具体情况调整）
            threshold = np.mean(distances) + 0.12 * np.std(distances)

            # 移除距离超过阈值的点（即离群点）
            filtered_points = np.array(last_points)[distances < threshold]
            final_filtered_points = filtered_points[(filtered_points[:, 0] < 310) & (filtered_points[:, 1] < 150)]
            center = np.mean(final_filtered_points, axis=0)
            contains_nan = np.any(np.isnan(center))

            if not contains_nan:
                cv2.circle(img, (int(center[0]), int(center[1])),  5, (255, 165, 0), -1)

        # Detect the good features to track
        h, w = frame_gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2 - 17, h // 2)
        radius = int(min(h, w) * 0.456)
        cv2.circle(mask, center, radius, 255, -1)
        p = cv2.goodFeaturesToTrack(frame_gray, **feature_params, mask=mask)
        if p is not None:
            # If good features can be tracked - add that to the trajectories
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])


    frame_idx += 1
    prev_gray = frame_gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)
    
    # Show Results
    # cv2.putText(img_viz, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.imwrite(f"of_out/{i}.png", img_viz)
    cv2.imshow('Optical Flow', img_viz)

    # cv2.imshow('Mask', mask)

    # 循环读取每张图像并写入视频

    video_writer.write(img_viz)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

video_writer.release()