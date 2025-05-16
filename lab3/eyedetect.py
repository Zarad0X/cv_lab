import os
import cv2
import shutil

# 初始化 Haar 检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

input_dir = "./CUHK student dataset/Testing"
output_dir = "./test"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    name = os.path.splitext(filename)[0]
    img_path = os.path.join(input_dir, filename)

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f" 未检测到人脸: {filename}")
        continue

    x, y, w, h = faces[0]
    face_roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_roi_gray)

    if len(eyes) < 2:
        print(f"❗ 未检测到两只眼睛: {filename}")
        continue

    # 取前两个检测到的眼睛
    eyes = sorted(eyes, key=lambda ex: ex[0])[:2]
    eye_coords = []
    for (ex, ey, ew, eh) in eyes:
        cx = x + ex + ew // 2
        cy = y + ey + eh // 2
        eye_coords.append((cx, cy))

    if len(eye_coords) < 2:
        continue

    # 创建子文件夹并保存图像与坐标
    person_dir = os.path.join(output_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    shutil.copy(img_path, os.path.join(person_dir, filename))

    with open(os.path.join(person_dir, f"{name}.txt"), "w") as f:
        f.write(f"{eye_coords[0][0]} {eye_coords[0][1]} {eye_coords[1][0]} {eye_coords[1][1]}\n")

    print(f"✅ 完成: {filename}")

print("🎉 所有图像已完成处理。")
