import numpy as np
import cv2
import time
# from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
from concurrent.futures import ThreadPoolExecutor

# Termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

print('Starting stereo calibration ... ')

for i in range(0, 64):
    t = str(i)
    ChessImgR = cv2.imread('calib_images/right_chessboard-' + t + '.png', 0)
    ChessImgL = cv2.imread('calib_images/left_chessboard-' + t + '.png', 0)
    if ChessImgR is None or ChessImgL is None:
        print(f"⚠️ Warning: Image {t} could not be loaded.")
        continue  # Skip this iteration if loading failed
    retR, cornersR = cv2.findChessboardCorners(ChessImgR, (9, 6), None)
    retL, cornersL = cv2.findChessboardCorners(ChessImgL, (9, 6), None)
    if retR and retL:
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImgR, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(ChessImgL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# Calibration
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, ChessImgR.shape[::-1], None, None)
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, ChessImgL.shape[::-1], None, None)
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, ChessImgR.shape[::-1], 1, ChessImgR.shape[::-1])
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, ChessImgL.shape[::-1], 1, ChessImgL.shape[::-1])

print('Calibration complete')

retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,imgpointsL,imgpointsR,mtxL,distL,mtxR,distR,ChessImgR.shape[::-1],criteria = criteria_stereo,flags = cv2.CALIB_FIX_INTRINSIC)

# StereoRectify function
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS, ChessImgR.shape[::-1], R, T, 0,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, ChessImgR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, ChessImgR.shape[::-1], cv2.CV_16SC2)

# Create StereoSGBM and prepare all parameters
block_size = 7
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,numDisparities = num_disp,blockSize = block_size,uniquenessRatio = 10,speckleWindowSize = 100,speckleRange = 32, disp12MaxDiff = 5,
    P1 = 8*3*block_size**2,
    P2 = 32*3*block_size**2)

# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(80000)
wls_filter.setSigmaColor(1.8)

Cam = cv2.VideoCapture(0)
Cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  
# Cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)  
Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
Cam.set(cv2.CAP_PROP_FPS, 60)


# Filtering
kernel= np.ones((3,3),np.uint8)

executor = ThreadPoolExecutor(max_workers=4)
prev_time = 0

while True:

    current_time = time.time()

    ret, frame = Cam.read()
    if not ret:
        print("❌ Failed to capture frame. Exiting...")
        break

    height, width, _ = frame.shape
    mid = width // 2

    left_frame = executor.submit(lambda: frame[:, :mid])
    right_frame = executor.submit(lambda: frame[:, mid:])
    LFrame = left_frame.result()
    RFrame = right_frame.result()

    # Start remap operations in parallel
    future_Left_nice = executor.submit(cv2.remap, LFrame, Left_Stereo_Map[0], Left_Stereo_Map[1], interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    future_Right_nice = executor.submit(cv2.remap, RFrame, Right_Stereo_Map[0], Right_Stereo_Map[1], interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    Left_nice = future_Left_nice.result()
    Right_nice = future_Right_nice.result()

    # Start grayscale conversion in parallel
    future_gray_left = executor.submit(cv2.cvtColor, Left_nice, cv2.COLOR_BGR2GRAY)
    future_gray_right = executor.submit(cv2.cvtColor, Right_nice, cv2.COLOR_BGR2GRAY)
    Gray_left = future_gray_left.result()
    Gray_right = future_gray_right.result()

    # Start disparity computations in parallel
    future_dispL = executor.submit(stereo.compute, Gray_left, Gray_right)
    future_dispR = executor.submit(stereoR.compute, Gray_right, Gray_left)
    DispL = np.int16(future_dispL.result())
    DispR = np.int16(future_dispR.result())

    disp = ((DispL.astype(np.float32) / 16) - min_disp) / num_disp

    filteredImg = wls_filter.filter(DispL, Gray_left, None, DispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

    # ----- Close Object Detection  ----- #
    _, close_mask = cv2.threshold(filteredImg, 160, 255, cv2.THRESH_BINARY)
    close_mask = cv2.morphologyEx(close_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(close_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)

            roi_disp = disp[y:y + h, x:x + w].astype(np.float32)
            cx, cy = x + w // 2, y + h // 2
            sample_disp = disp[cy - 1:cy + 2, cx - 1:cx + 2].astype(np.float32)
            average_disp = np.mean(sample_disp[sample_disp > 0])

            if average_disp > 0:
                distance = -593.97 * average_disp**3 + 1506.8 * average_disp**2 - 1373.1 * average_disp + 522.06
                distance = np.around(distance * 0.01, decimals=2)

                if distance < 1.0:
                    box_color = (0, 0, 255) if distance < 0.5 else (0, 255, 0)
                    cv2.rectangle(filt_Color, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(filt_Color, f"{distance:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    # ----- Close Object Detection  ----- #

    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(filt_Color, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('Filtered Color Depth', filt_Color)

    #print(fps)

    if cv2.waitKey(1) & 0xFF == 27:
        break

Cam.release()
cv2.destroyAllWindows()
