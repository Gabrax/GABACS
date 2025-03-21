import numpy as np
import cv2

print('Starting Dual-Lens Camera Calibration. Press and hold the space bar to exit.\n')
print('Press (s) to save the images, (c) to continue without saving.')

id_image = 0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

Cam = cv2.VideoCapture(0)

Cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
Cam.set(cv2.CAP_PROP_FPS, 60)

while True:
    ret, frame = Cam.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    
    height, width, _ = frame.shape
    mid = width // 2  # Split point

    left_frame = frame[:, :mid]   # Left half
    right_frame = frame[:, mid:]  # Right half

    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Detect chessboard on both halves
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (9, 6), None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (9, 6), None)

    if ret_left:
        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(gray_left, (9, 6), corners2_left, ret_left)

    if ret_right:
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(gray_right, (9, 6), corners2_right, ret_right)

    cv2.imshow('Left Camera Feed', left_frame)
    cv2.imshow('Right Camera Feed', right_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print(f'Saving images {id_image}...')
        cv2.imwrite(f'left_chessboard-{id_image}.png', left_frame)
        cv2.imwrite(f'right_chessboard-{id_image}.png', right_frame)
        id_image += 1
    elif key == ord(' '):
        break

Cam.release()
cv2.destroyAllWindows()
