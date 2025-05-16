import os
import cv2

# 数据路径（原始图像 + txt）
data_dir = "./test"

# 输出路径（可视化后的图像）
vis_dir = "./visualized"
os.makedirs(vis_dir, exist_ok=True)

for person in os.listdir(data_dir):
    person_dir = os.path.join(data_dir, person)
    if not os.path.isdir(person_dir):
        continue

    # 找图像和对应 txt
    img_file = None
    txt_file = None
    for file in os.listdir(person_dir):
        if file.lower().endswith((".jpg", ".png")):
            img_file = file
        elif file.endswith(".txt"):
             
            txt_file = file

    if not img_file or not txt_file:
        continue

    img_path = os.path.join(person_dir, img_file)
    txt_path = os.path.join(person_dir, txt_file)

    img = cv2.imread(img_path)
    with open(txt_path, "r") as f:
        coords = list(map(int, f.readline().strip().split()))

    if len(coords) != 4:
        print(f"⚠️ 坐标格式错误: {txt_path}")
        continue

    x1, y1, x2, y2 = coords

    # 画出红蓝圆表示眼睛中心
    cv2.circle(img, (x1, y1), 5, (0, 0, 255), -1)  # 红色：左眼
    cv2.circle(img, (x2, y2), 5, (255, 0, 0), -1)  # 蓝色：右眼

    # 保存图像到 visualized/person.jpg
    out_path = os.path.join(vis_dir, f"{person}.jpg")
    cv2.imwrite(out_path, img)
    print(f"✅ 已保存可视化图像: {out_path}")

print("🎉 所有图像已保存到 visualized 文件夹。")
