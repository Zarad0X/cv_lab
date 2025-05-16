import cv2
import numpy as np
import sys
import pickle
import os

IMG_SIZE = (100, 100)
EYE_DST = 40

def align_face(image, eye_coords):
    eye1 = np.array(eye_coords[:2])
    eye2 = np.array(eye_coords[2:])
    if eye1[0] > eye2[0]:
        eye1, eye2 = eye2, eye1

    dx, dy = eye2 - eye1
    angle = np.degrees(np.arctan2(dy, dx))
    center = tuple(map(int, ((eye1 + eye2) / 2)))

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, image.shape[1::-1])
    eye1_rot = np.dot(M[:, :2], eye1) + M[:, 2]
    eye2_rot = np.dot(M[:, :2], eye2) + M[:, 2]

    dx = eye2_rot[0] - eye1_rot[0]
    scale = EYE_DST / dx
    M = cv2.getRotationMatrix2D(tuple(eye1_rot), 0, scale)
    scaled = cv2.warpAffine(rotated, M, image.shape[1::-1])

    x = int(eye1_rot[0] * scale - EYE_DST * 0.3)
    y = int(eye1_rot[1] * scale - EYE_DST * 0.4)
    cropped = scaled[y:y+IMG_SIZE[1], x:x+IMG_SIZE[0]]

    if cropped.shape[0] != IMG_SIZE[1] or cropped.shape[1] != IMG_SIZE[0]:
        return None
    return cv2.resize(cropped, IMG_SIZE)

def recognize(test_img_path, model_path):
    txt_path = os.path.splitext(test_img_path)[0] + ".txt"
    if not os.path.exists(txt_path):
        print(" 缺少眼睛坐标文件:", txt_path)
        return

    image = cv2.imread(test_img_path)
    if image is None:
        print("图像读取失败:", test_img_path)
        return
    with open(txt_path, "r") as f:
        coords = list(map(int, f.readline().strip().split()))
    if len(coords) != 4:
        print("眼睛坐标格式错误")
        return

    aligned = align_face(image, coords)
    if aligned is None:
        print(" 对齐失败")
        return

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    mean = model["mean"]
    eigfaces = model["eigfaces"]
    projections = model["projections"]
    labels = model["labels"]

    test_proj = np.dot(flat - mean, eigfaces)

    distances = np.linalg.norm(projections - test_proj, axis=1)
    idx = np.argmin(distances)
    matched_label = labels[idx]
    print(f"识别结果: {matched_label}  (距离: {distances[idx]:.2f})")

    # 加载最相似图像用于显示
    match_img_path = None
    train_dirs = ["train", "./train"]
    for train_dir in train_dirs:
        cand = os.path.join(train_dir, matched_label, matched_label + ".jpg")
        if os.path.exists(cand):
            match_img_path = cand
            break

    # 可视化叠加
    cv2.putText(image, f"Recognized as: {matched_label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Input Image", image)

    if match_img_path:
        matched_img = cv2.imread(match_img_path)
        cv2.imshow("Most Similar in DB", matched_img)
    else:
        print("未找到匹配图像用于显示")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python mytest.py <测试图像路径> <模型文件路径>")
        sys.exit(1)

    test_img = sys.argv[1]
    model_path = sys.argv[2]
    recognize(test_img, model_path)
