import cv2
import numpy as np

# 颜色过滤参数
lower_yellow_green = np.array([35, 0, 0])  # 黄绿色的下界
upper_yellow_green = np.array([70, 255, 255])  # 黄绿色的上界

# 轨迹估计参数
min_contour_area = 150  # 最小轮廓面积，用于筛选小球候选区域

# 打开视频文件
cap = cv2.VideoCapture(0)

# 读取第一帧
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# 初始化轨迹点列表
trajectory = []

while cap.isOpened():# and len(trajectory) < 50:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为HSV色彩空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 颜色过滤
    mask = cv2.inRange(hsv_frame, lower_yellow_green, upper_yellow_green)

    # 轮廓检测
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            print(area)
            # 计算轮廓的外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 计算轮廓的中心坐标
            center_x = x + w // 2
            center_y = y + h // 2
            
            # 将中心坐标添加到轨迹点列表
            trajectory.append((center_x, center_y))
            
    # 绘制轨迹
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i-1], trajectory[i], (0, 0, 255), 2)
        
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
