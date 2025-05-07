import cv2
import os
import sys
import glob
import numpy as np

def resize_frame(frame, size):
    return cv2.resize(frame, size)


#创建片头帧
def create_text_frame(text, size, duration_sec, fps):
    frames = []
    frame = 255 * np.ones((size[1], size[0], 3), dtype=np.uint8)
    cv2.putText(frame, text, (50, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    for _ in range(int(duration_sec * fps)):
        frames.append(frame.copy())
    return frames

def add_footer_text(frame, text):
    h, w = frame.shape[:2]
    cv2.putText(frame, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

def main(input_path):
    import numpy as np

    # 找到视频和图片
    video_files = glob.glob(os.path.join(input_path, "*.avi"))
    image_files = sorted(glob.glob(os.path.join(input_path, "*.jpg")))

    if not video_files:
        print("未找到视频文件（.avi）")
        return
    if len(image_files) < 3:
        print("请至少放入 3 张 .jpg 图片")
        return

    video_path = video_files[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    # 输出视频
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, size)

    #  写片头 
    print("写入片头...")
    title_frames = create_text_frame(" Computer Vision HW1 ", size, duration_sec=2, fps=fps)
    for f in title_frames:
        out.write(add_footer_text(f, "ZhangHan_3230102282"))

    #  写入照片（幻灯片效果）
    print("写入图片幻灯片...")
    for img_path in image_files:
        img = cv2.imread(img_path)
        img_resized = resize_frame(img, size)
        for _ in range(int(fps * 1.5)):  # 每张图显示1.5秒
            out.write(add_footer_text(img_resized.copy(), "ZhangHan_3230102282"))

    #  写入原始视频 
    print("写入原始视频...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = add_footer_text(frame, "ZhangHan_3230102282")
        out.write(frame)

    cap.release()
    out.release()
    print("生成完成，输出文件：output.avi")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python makevideo.py <文件夹路径>")
    else:
        main(sys.argv[1])