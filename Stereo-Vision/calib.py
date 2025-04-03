
import numpy as np
import cv2

print('Starting Dual-Lens Camera Calibration. Press and hold the space bar to exit.\n')
print('Press (s) to save the images, (c) to continue without saving.')

CHESSBOARD_SIZE = (9, 6)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

Cam = cv2.VideoCapture(0)

Cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  
Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
Cam.set(cv2.CAP_PROP_FPS, 60)

id_image = 0  # Image index for saving

while True:
    ret, frame = Cam.read()
    if not ret:
        print("❌ Failed to capture frame. Exiting...")
        break

    height, width, _ = frame.shape
    mid = width // 2  # Middle point for splitting

    left_frame = frame[:, :mid]   # Left camera view
    right_frame = frame[:, mid:]  # Right camera view
    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)



    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHESSBOARD_SIZE, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHESSBOARD_SIZE, None)
    cv2.imshow('imgLeft', left_frame)
    cv2.imshow('imgRight', right_frame)



    if ret_left & ret_right:
        corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
    
        cv2.drawChessboardCorners(gray_right, CHESSBOARD_SIZE, corners2_right, ret_right)
        cv2.drawChessboardCorners(gray_left, CHESSBOARD_SIZE, corners2_left, ret_left)
        cv2.imshow('Left Camera', gray_left)
        cv2.imshow('Right Camera', gray_right)

        if cv2.waitKey(0) & 0xFF == ord('s'): 
            print(f'Saving images {id_image}...')
            cv2.imwrite(f'calib_images/left_chessboard-{id_image}.png', left_frame)
            cv2.imwrite(f'calib_images/right_chessboard-{id_image}.png', right_frame)
            id_image += 1
        else:
            print("Chessboard not detected in both cameras. Skipping save.")

    elif cv2.waitKey(1) & 0xFF == ord(' '):  # Spacebar to exit
        print("Exiting calibration...")
        break

Cam.release()
cv2.destroyAllWindows()

