import cv2
import os
import time
import re
import numpy as np
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
        if abs(pt1[1] - pt2[1]) < thr and abs(pt1[0] - pt2[0]) < 400:
            filted_mathces.append(match)
    
    return filted_mathces


image_folder_l = '/home/mingtao/denoise/cone_output_cam0'
image_folder_r = '/home/mingtao/denoise/cone_output_cam1'

image_files_l = [os.path.join(image_folder_l, f) for f in os.listdir(image_folder_l) if f.endswith('.bmp') or f.endswith('.png')]
image_files_l = sorted(image_files_l, key=n_sort)

image_files_r = [os.path.join(image_folder_r, f) for f in os.listdir(image_folder_r) if f.endswith('.bmp') or f.endswith('.png')]
image_files_r = sorted(image_files_r, key=n_sort)

descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)

# 创建SIFT检测器
# harris = cv2.GFTTDetector_create(
#                 maxCorners=1000, minDistance=5.0, 
#                 qualityLevel=0.008, useHarrisDetector=False)

iniThFAST = 5
minThFAST = 1

# 创建 ORB 实例，设置初始阈值
orb = cv2.ORB_create(
    scaleFactor=1.1, 
    nlevels=10, 
    edgeThreshold=19, 
    patchSize=31, 
    WTA_K=3, 
    scoreType=cv2.ORB_HARRIS_SCORE, 
    fastThreshold=iniThFAST
)


# 创建BFMatcher对象
mean_matches = []
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
# 遍历后续图片进行关键点匹配
for i in range(1, len(image_files_l), 1):

    st = time.time()

    prev_img = cv2.imread(image_files_l[i])
    # prev_img = cv2.resize(prev_img, (320, 287 //2), interpolation=cv2.INTER_AREA)
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("g", prev_gray)
    # cv2.waitKey(0)
    # kp_pre = harris.detect(prev_gray, None)

    # prev_kp, prev_des = descriptor_extractor.compute(prev_gray, kp_pre)
    prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)
    curr_img = cv2.imread(image_files_r[i])
    # curr_img = cv2.resize(curr_img, (320, 287 //2), interpolation=cv2.INTER_AREA)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    # curr_gray = cv2.ximgproc.thinning(curr_gray)
    # kp_curr = harris.detect(curr_gray, None)
    # curr_kp, curr_des = descriptor_extractor.compute(curr_gray, kp_curr)
    curr_kp, curr_des = orb.detectAndCompute(curr_gray, None)
    
    matches = bf.match(prev_des, curr_des)

    matches = filter_matches(prev_kp, curr_kp, matches, 5)
    mean_matches.append(len(matches))
    print(np.mean(mean_matches))
    
    # 申请比值测试来去除错误匹
    
    # 绘制匹配结果
    img_matches = cv2.drawMatches(prev_img, prev_kp, curr_img, curr_kp, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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
