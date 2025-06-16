import cv2
import numpy as np
import sys
import os

def help():
    print("\n\n"
          " Calling convention:\n"
          " python combined_script.py mode board_w board_h number_of_boards [checker_image]\n"
          "\n"
          "   WHERE:\n"
          "     mode              -- 'calibrate' for camera calibration, 'birdseye' for bird's eye view\n"
          "     board_w, board_h  -- number of corners along the row and columns respectively\n"
          "     number_of_boards  -- number of chessboard views to collect before calibration\n"
          "     checker_image     -- path to checkerboard image for birdseye mode (optional)\n")

def get_image_files(directory):
    image_extensions = ('.jpg', '.jpeg', '.png')
    return [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

def calibrate_camera(board_w, board_h, n_boards):
    image_dir = "./calibration"
    image_files = get_image_files(image_dir)
    if not image_files:
        print(f"Error: No images found in {image_dir}")
        return None, None

    board_n = board_w * board_h
    board_sz = (board_w, board_h)

    objp = np.zeros((board_n, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    cv2.namedWindow("Calibration")
    successes = 0
    image_idx = 0

    while successes < n_boards and image_idx < len(image_files):
        image_path = os.path.join(image_dir, image_files[image_idx])
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Couldn't load {image_path}")
            image_idx += 1
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_sz,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            cv2.drawChessboardCorners(image, board_sz, corners2, ret)
            objpoints.append(objp)
            imgpoints.append(corners2)
            successes += 1
            print(f"Collected {successes} of {n_boards} chessboard images")
        cv2.imshow("Calibration", image)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('p'):
            while True:
                key = cv2.waitKey(250) & 0xFF
                if key == ord('p') or key == 27:
                    break
        if key == 27:
            break
        image_idx += 1

    if successes < n_boards:
        print(f"Warning: Only collected {successes} of {n_boards} needed images")
        return None, None

    print("\n*** CALIBRATING THE CAMERA...")
    ret, intrinsic, distortion, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Calibration done. Saving Intrinsics.xml and Distortion.xml")

    fs1 = cv2.FileStorage("Intrinsics.xml", cv2.FILE_STORAGE_WRITE)
    fs1.write("intrinsic", intrinsic)
    fs1.release()

    fs2 = cv2.FileStorage("Distortion.xml", cv2.FILE_STORAGE_WRITE)
    fs2.write("distortion", distortion)
    fs2.release()

    return intrinsic, distortion

def birdseye_view(board_w, board_h, intrinsic, distortion, checker_image=None):
    image_dir = "./birdseye"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    image_files = [checker_image] if checker_image else get_image_files(image_dir)
    if not image_files:
        print(f"Error: No images found in {image_dir}")
        return

    board_sz = (board_w, board_h)
    square_size = 50  # 可以根据实际棋盘格子大小设置

    for image_file in image_files:
        image_path = image_file if checker_image else os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Couldn't load {image_path}")
            continue

        undistorted = cv2.undistort(image, intrinsic, distortion)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, board_sz,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FILTER_QUADS)
        if not ret:
            print(f"Could not detect chessboard in {image_path}")
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

        img_pts = np.array([
            corners[0][0],
            corners[board_w - 1][0],
            corners[(board_h - 1) * board_w][0],
            corners[-1][0]
        ], dtype=np.float32)

        obj_pts = np.array([
            [0, 0],
            [(board_w - 1) * square_size, 0],
            [0, (board_h - 1) * square_size],
            [(board_w - 1) * square_size, (board_h - 1) * square_size]
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(img_pts, obj_pts)

        birdseye_width = (board_w - 1) * square_size
        birdseye_height = (board_h - 1) * square_size

        birds_image = cv2.warpPerspective(undistorted, H, (birdseye_width, birdseye_height),
                                          flags=cv2.INTER_LINEAR)

        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"birdseye_{base_name}")
        cv2.imwrite(output_path, birds_image)
        print(f"Saved bird's eye view: {output_path}")

def main():
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print("ERROR: Wrong number of input parameters")
        help()
        return -1

    mode = sys.argv[1].lower()
    board_w = int(sys.argv[2])
    board_h = int(sys.argv[3])
    n_boards = int(sys.argv[4])
    checker_image = sys.argv[5] if len(sys.argv) == 6 else None

    if mode == "calibrate":
        intrinsic, distortion = calibrate_camera(board_w, board_h, n_boards)
        if intrinsic is not None and distortion is not None:
            print("Calibration completed. Proceeding to birdseye view with last image...")
            birdseye_view(board_w, board_h, intrinsic, distortion)
    elif mode == "birdseye":
        fs1 = cv2.FileStorage("Intrinsics.xml", cv2.FILE_STORAGE_READ)
        intrinsic = fs1.getNode("intrinsic").mat()
        fs1.release()

        fs2 = cv2.FileStorage("Distortion.xml", cv2.FILE_STORAGE_READ)
        distortion = fs2.getNode("distortion").mat()
        fs2.release()

        if intrinsic is None or distortion is None:
            print("Error: Intrinsics.xml or Distortion.xml not found. Please run calibrate mode first.")
            return -1

        birdseye_view(board_w, board_h, intrinsic, distortion, checker_image)
    else:
        print("Error: Invalid mode. Use 'calibrate' or 'birdseye'")
        help()
        return -1

if __name__ == "__main__":
    main()
