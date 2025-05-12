import cv2
import numpy as np
import os

# 输入输出目录
input_dir = 'input'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# 遍历 input 文件夹中的图像
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        filepath = os.path.join(input_dir, filename)

        # 读取图像
        image = cv2.imread(filepath)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 1: 边缘检测，输出为黑底白边图像
        edges = cv2.Canny(gray, 100, 200)


        # Step 2: 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Step 3: 在原图上绘制拟合椭圆
        output_image = image.copy()
        for cnt in contours:
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(output_image, ellipse, (0, 0, 255), 2)  # 红色绘制

        # 保存拟合图像
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, output_image)
        print(f'Processed and saved: {output_path}')
