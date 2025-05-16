import cv2
import numpy as np
import os
import sys
import pickle

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

def load_images(train_dir):
    images = []
    labels = []
    for person_dir in os.listdir(train_dir):
        person_path = os.path.join(train_dir, person_dir)
        if not os.path.isdir(person_path):
            continue
        img_path = os.path.join(person_path, f"{person_dir}.jpg")
        txt_path = os.path.join(person_path, f"{person_dir}.txt")
        if not os.path.exists(img_path) or not os.path.exists(txt_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue
        with open(txt_path, "r") as f:
            coords = list(map(int, f.readline().strip().split()))
        if len(coords) != 4:
            continue

        aligned = align_face(image, coords)
        if aligned is None:
            continue

        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        images.append(gray.flatten())
        labels.append(person_dir)
    return np.array(images), labels

def compute_pca(images, energy_threshold):
    mean_face = np.mean(images, axis=0)
    A = images - mean_face
    cov = np.dot(A, A.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    total_energy = np.sum(eigvals)
    retained_energy = 0
    k = 0
    for val in eigvals:
        retained_energy += val
        k += 1
        if retained_energy / total_energy >= energy_threshold:
            break

    print(f" 选取前 {k} 个特征脸，保留能量比例: {retained_energy/total_energy:.4f}")
    eignfaces = np.dot(A.T, eigvecs[:, :k])
    eignfaces = eignfaces / np.linalg.norm(eignfaces, axis=0)

    projections = np.dot(A, eignfaces)
    return mean_face, eignfaces, projections, k

def visualize_eigenfaces(eignfaces):
    face_images = []
    for i in range(min(10, eignfaces.shape[1])):
        face = eignfaces[:, i].reshape(IMG_SIZE)
        norm_face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX)
        face_images.append(norm_face.astype(np.uint8))
    vis = cv2.hconcat(face_images)
    cv2.imshow("Top 10 Eigenfaces", vis)
    cv2.imwrite("eigenfaces.jpg", vis)
    print(" 特征脸图像已保存为 eigenfaces.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_model(filename, mean, eigfaces, projections, labels):
    model = {
        "mean": mean,
        "eigfaces": eigfaces,
        "projections": projections,
        "labels": labels
    }
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f" 模型已保存为 {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python mytrain.py <能量百分比> <模型文件名> [训练集目录]")
        sys.exit(1)

    energy = float(sys.argv[1])
    model_file = sys.argv[2]
    train_dir = sys.argv[3] if len(sys.argv) >= 4 else "train"

    print(f" 训练集中读取图像: {train_dir}")
    images, labels = load_images(train_dir)
    print(f" 加载 {len(images)} 张图像")

    mean, eignfaces, projections, k = compute_pca(images, energy)
    save_model(model_file, mean, eignfaces, projections, labels)
    visualize_eigenfaces(eignfaces)
