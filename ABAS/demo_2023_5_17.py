import cv2
import numpy as np

# 运动模糊去除参数
num_frames = 5  # 平均帧数，用于运动模糊去除

# 颜色过滤参数
lower_yellow_green = np.array([20, 100, 100])  # 黄绿色的下界
upper_yellow_green = np.array([60, 255, 255])  # 黄绿色的上界

# 打开视频文件
cap = cv2.VideoCapture(0)

# 读取第一帧
ret, prev_frame = cap.read()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 运动模糊去除
    blurred_frame = np.zeros_like(frame, dtype=np.float32)
    for _ in range(num_frames):
        blurred_frame += frame.astype(np.float32)
        ret, frame = cap.read()
    blurred_frame /= num_frames

    # 颜色过滤
    hsv_frame = cv2.cvtColor(blurred_frame.astype(np.uint8), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_yellow_green, upper_yellow_green)
    masked_frame = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask)

    # 查找轮廓并绘制目标区域
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # 面积阈值，可根据实际情况调整
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(prev_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", prev_frame)
    prev_frame = frame
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
