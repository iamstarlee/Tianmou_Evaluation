import cv2
import os
import time
import re

# 获取文件夹中的所有图片文件名，并按顺序排序
def n_sort(value):
    parts = re.split(r'(\d+)', value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def filter_matches(kp1, kp2, matches, thr):
    filted_mathces = []
    for match in matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        if abs(pt1[1] - pt2[1]) < thr:
            filted_mathces.append(match)
    
    return filted_mathces


image_folder = '/home/mingtao/THU/SLAM_DATA/Big_pose1/cons_r'
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpeg') or f.endswith('.png')]
image_files = sorted(image_files, key=n_sort)

descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=64, use_orientation=False)

# 创建SIFT检测器
harris = cv2.GFTTDetector_create(
                maxCorners=2000, minDistance=3.0, 
                qualityLevel=0.01, useHarrisDetector=False)



# 创建BFMatcher对象
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 读取第一张图片
prev_img = cv2.imread(image_files[0])
prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB) 
prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
kp_pre = harris.detect(prev_gray, None)

prev_kp, prev_des = descriptor_extractor.compute(prev_gray, kp_pre)

# 遍历后续图片进行关键点匹配
for i in range(0, len(image_files), 1):
    st = time.time()
    
    curr_img = cv2.imread(image_files[i])
    # curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    kp_curr = harris.detect(curr_gray, None)

    curr_kp, curr_des = descriptor_extractor.compute(curr_gray, kp_curr)

    if len(curr_kp) < 3:
        continue
    
    matches = bf.match(prev_des, curr_des)

    matches = filter_matches(prev_kp, curr_kp, matches, 5)
    print(len(matches))
    
    # 申请比值测试来去除错误匹
    
    # 绘制匹配结果
    img_matches = cv2.drawMatches(prev_img, prev_kp, curr_img, curr_kp, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    end = time.time()
    print("FPS:", 1/(end-st))
    # 显示结果
    cv2.imshow('Matches', img_matches)
    if cv2.waitKey(0) & 0xFF == 27:  # 按下ESC键退出
        break
    
    # 更新前一帧的关键点和描述子
    prev_img = curr_img
    prev_gray = curr_gray
    prev_kp, prev_des = curr_kp, curr_des

cv2.destroyAllWindows()
