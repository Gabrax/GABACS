import numpy as np
import cv2
import time
# from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
from concurrent.futures import ThreadPoolExecutor
import lgpio
from collections import deque

chip = lgpio.gpiochip_open(0)
PIN = 17
lgpio.gpio_claim_output(chip,PIN,1)

# Filtering
kernel= np.ones((3,3),np.uint8)

def estimate_distance(disparity_value):
    distance = -593.97 * disparity_value**3 + 1506.8 * disparity_value**2 - 1373.1 * disparity_value + 522.06
    return np.around(distance * 0.01, decimals=2)  # Convert to meters

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

ChessImaR = None
ChessImaL = None

for i in range(0, 64):
    t = str(i)
    ChessImaR = cv2.imread('calib_images/right_chessboard-' + t + '.png', 0)
    ChessImaL = cv2.imread('calib_images/left_chessboard-' + t + '.png', 0)
    if ChessImaR is None or ChessImaL is None:
        print(f"⚠️ Warning: Image {t} could not be loaded.")
        continue  # Skip this iteration if loading failed
    retR, cornersR = cv2.findChessboardCorners(ChessImaR, (9, 6), None)
    retL, cornersL = cv2.findChessboardCorners(ChessImaL, (9, 6), None)
    if retR and retL:
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# Calibration
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, ChessImaR.shape[::-1], None, None)
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, ChessImaL.shape[::-1], None, None)
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, ChessImaR.shape[::-1], 1, ChessImaR.shape[::-1])
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, ChessImaL.shape[::-1], 1, ChessImaL.shape[::-1])

retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,imgpointsL,imgpointsR,mtxL,distL,mtxR,distR,ChessImaR.shape[::-1],criteria = criteria_stereo,flags = cv2.CALIB_FIX_INTRINSIC)

print('Calibration complete')

# StereoRectify function
rectify_scale= 0 # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS, ChessImaR.shape[::-1], R, T, rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped

# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, ChessImaR.shape[::-1], cv2.CV_16SC2)

# Create StereoSGBM and prepare all parameters
window_size = 7
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,numDisparities = num_disp,blockSize = window_size,uniquenessRatio = 10,speckleWindowSize = 100,speckleRange = 32, disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)

# Used for the filtered image
# Create another stereo for right this time
stereoR=cv2.ximgproc.createRightMatcher(stereo)

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

Cam = cv2.VideoCapture(0)
Cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)  
Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
Cam.set(cv2.CAP_PROP_FPS, 60)

prev_time = 0

executor = ThreadPoolExecutor(max_workers=4)

focal_length_px = PL[0, 0]
baseline_m = abs(T[0][0]) / 100  

# --- Configurable Parameters --- #
MIN_DISTANCE_TRIGGER = 0.65
MAX_DISTANCE_RELEASE = 0.7
DISPARITY_RANGE = (1.0, 150.0)
CONTOUR_AREA_THRESHOLD = 500
DISP_AVG_HISTORY = 5

distance_history = deque(maxlen=DISP_AVG_HISTORY)

while True:
    current_time = time.time()
    ret, frame = Cam.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    height, width, _ = frame.shape
    mid = width // 2

    left_frame = frame[:, :mid]
    right_frame = frame[:, mid:]

    # Remap using stereo rectification maps
    Left_nice = executor.submit(cv2.remap, left_frame, Left_Stereo_Map[0], Left_Stereo_Map[1],
                                interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT).result()
    Right_nice = executor.submit(cv2.remap, right_frame, Right_Stereo_Map[0], Right_Stereo_Map[1],
                                 interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT).result()

    # Convert to grayscale
    gray_left = executor.submit(cv2.cvtColor, Left_nice, cv2.COLOR_BGR2GRAY).result()
    gray_right = executor.submit(cv2.cvtColor, Right_nice, cv2.COLOR_BGR2GRAY).result()

    # Compute disparity maps
    dispL = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    dispR = stereoR.compute(gray_right, gray_left).astype(np.float32) / 16.0

    # Apply WLS filter
    filtered_disp = wls_filter.filter(dispL, gray_left, None, dispR)
    disp_vis = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    disp_closed = cv2.morphologyEx(disp_vis, cv2.MORPH_CLOSE, kernel)

    # # --- Define central region of interest (ROI) ---
    # disp_height, disp_width = disp_vis.shape
    # roi_width = disp_width // 3  # 1/3 width
    # roi_height = disp_height // 2  # 1/2 height
    # roi_x = (disp_width - roi_width + 150) // 2
    # roi_y = (disp_height - roi_height) // 2
    # roi_rect = (roi_x, roi_y, roi_width, roi_height)

    # # Visualize ROI
    # cv2.rectangle(color_disp, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 255), 2)

    # Detect close objects via thresholding
    #_, close_mask = cv2.threshold(disp_vis, 160, 255, cv2.THRESH_BINARY)
    #close_mask = cv2.morphologyEx(close_mask, cv2.MORPH_CLOSE, kernel)

    # # Restrict mask to ROI only
    # roi_mask = np.zeros_like(close_mask)
    # roi_mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = close_mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # # Find contours in restricted region
    # contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # current_stop_flag = False

    # for cnt in contours:
        # if cv2.contourArea(cnt) < CONTOUR_AREA_THRESHOLD:
            # continue

        # x, y, w, h = cv2.boundingRect(cnt)
        # cx, cy = x + w // 2, y + h // 2

        # # Make sure coordinates are within image bounds
        # if cy - 1 < 0 or cy + 2 > dispL.shape[0] or cx - 1 < 0 or cx + 2 > dispL.shape[1]:
            # continue

        # roi_disp = dispL[cy - 1:cy + 2, cx - 1:cx + 2]
        # valid_disp = roi_disp[(roi_disp > DISPARITY_RANGE[0]) & (roi_disp < DISPARITY_RANGE[1])]

        # if valid_disp.size == 0:
            # continue

        # avg_disp = np.median(valid_disp)
        # distance = (focal_length_px * baseline_m) / avg_disp
        # distance_history.append(distance)
        # smoothed_distance = np.median(distance_history)

        # print(f"Distance: {smoothed_distance:.2f} m")

        # # Apply hysteresis logic for stability
        # if smoothed_distance < MIN_DISTANCE_TRIGGER:
            # current_stop_flag = True
            # box_color = (0, 0, 255) if smoothed_distance < 0.5 else (0, 255, 0)
            # cv2.rectangle(color_disp, (x, y), (x + w, y + h), box_color, 2)
            # cv2.putText(color_disp, f"{smoothed_distance:.2f} m", (x, y - 10),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            # break

    # # Trigger GPIO only once per frame
    # if current_stop_flag:
        # lgpio.gpio_write(chip, PIN, 0)  # Stop
    # elif len(distance_history) == DISP_AVG_HISTORY and smoothed_distance > MAX_DISTANCE_RELEASE:
        # lgpio.gpio_write(chip, PIN, 1)  # Move forward

    # current_time = time.time()
    # fps = 1 / (current_time - prev_time + 1e-6)  # avoid div by zero
    # prev_time = current_time

    # # Overlay FPS and resolution info
    # cv2.putText(color_disp, f"FPS: {fps:.2f}", (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # cv2.putText(color_disp, f"Size: {disp_width}x{disp_height}", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Filtered Color Depth', disp_vis)

    if cv2.waitKey(1) & 0xFF == 27:
        lgpio.gpio_write(chip,PIN,1)
        lgpio.gpiochip_close(chip)
        break

lgpio.gpiochip_close(chip)
Cam.release()
cv2.destroyAllWindows()
