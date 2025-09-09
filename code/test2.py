"""
stereo_distance_fixed.py

Fixed and improved version of your stereo pipeline:
- calibrates using real checkerboard square size (meters)
- uses Q and cv2.reprojectImageTo3D to get metric distances (meters)
- robust median over small ROI for distance estimate
- tuned numDisparities (multiple of 16)
"""

import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import sys
import os

# -------------------- CONFIG --------------------
# real checkerboard parameters (change to your board)
CHECKER_ROWS = 6      # inner corners (height)
CHECKER_COLS = 9      # inner corners (width)
square_size = 0.025   # meters (e.g., 0.025 for 25 mm squares) <-- CHANGE IF NEEDED

# calibration images folder and filename template
CALIB_DIR = "calib_images"
LEFT_TEMPLATE = "left_chessboard-{}.png"
RIGHT_TEMPLATE = "right_chessboard-{}.png"
NUM_CALIB_IMAGES_TO_TRY = 64

# stereo matcher params
window_size = 7
min_disp = 0
max_expected_disp = 128  # choose sufficiently large for near objects; will be rounded to multiple of 16
num_disp = ((max_expected_disp // 16) + 1) * 16

# WLS params
lmbda = 80000
sigma = 1.8

# ROI / detection params
DISPARITY_MIN = 1.0
DISPARITY_MAX = 500.0
CONTOUR_AREA_THRESHOLD = 100  # you can tune
DISP_MEDIAN_ROI = 3  # use (2*ROi+1) x (2*ROI+1) window around centroid

# Camera capture
CAM_INDEX = 0
CAP_WIDTH = 1100
CAP_HEIGHT = 270
CAP_FPS = 60

# smoothing
DISP_AVG_HISTORY = 5

# -------------------------------------------------

# small helper: check that ximgproc exists
if not hasattr(cv2, "ximgproc"):
    print("Error: OpenCV ximgproc module is required (cv2.ximgproc). Install opencv-contrib-python.")
    sys.exit(1)

# morphological kernel
kernel = np.ones((3, 3), np.uint8)

# termination criteria
term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points (3D points) in meters
objp = np.zeros((CHECKER_ROWS * CHECKER_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKER_COLS, 0:CHECKER_ROWS].T.reshape(-1, 2) * square_size

# arrays to store object points and image points
objpoints = []
imgpointsL = []
imgpointsR = []

print("Starting stereo calibration (loading images)...")

# load chessboard pairs
last_left_shape = None
for i in range(NUM_CALIB_IMAGES_TO_TRY):
    idx = str(i)
    left_path = os.path.join(CALIB_DIR, LEFT_TEMPLATE.format(idx))
    right_path = os.path.join(CALIB_DIR, RIGHT_TEMPLATE.format(idx))

    if not os.path.exists(left_path) or not os.path.exists(right_path):
        # skip missing pairs
        # print(f"Warning: missing pair {idx}")
        continue

    left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    if left_img is None or right_img is None:
        continue

    foundL, cornersL = cv2.findChessboardCorners(left_img, (CHECKER_COLS, CHECKER_ROWS), None)
    foundR, cornersR = cv2.findChessboardCorners(right_img, (CHECKER_COLS, CHECKER_ROWS), None)
    if foundL and foundR:
        objpoints.append(objp.copy())
        cv2.cornerSubPix(left_img, cornersL, (11, 11), (-1, -1), term_criteria)
        cv2.cornerSubPix(right_img, cornersR, (11, 11), (-1, -1), term_criteria)
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)
        last_left_shape = left_img.shape[::-1]  # (width, height)

print(f"Found {len(objpoints)} good pairs for calibration.")

if len(objpoints) < 5:
    print("Not enough calibration pairs found. Need more good chessboard images.")
    sys.exit(1)

# calibrate single cameras
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, last_left_shape, None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, last_left_shape, None, None)

# get optimal new camera matrices (we'll use them for undistort/rectify)
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, last_left_shape, 1, last_left_shape)
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, last_left_shape, 1, last_left_shape)

# stereo calibration (fix intrinsics) to get R, T, E, F
flags = cv2.CALIB_FIX_INTRINSIC
retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    last_left_shape, criteria=term_criteria, flags=flags
)

print("Stereo calibration done.")
print(f"stereoCalibrate RMS: {retS}")
print("Translation vector T (meters because objp used meters):\n", T)
print("Fundamental matrix F:\n", F)

# rectify
rectify_scale = 0  # 0 = crop, 1 = keep all
RL, RR, PL, PR, Q, roiLr, roiRr = cv2.stereoRectify(MLS, dLS, MRS, dRS, last_left_shape, R, T, rectify_scale, (0, 0))
print("Projection matrix PL:\n", PL)
focal_length_px = PL[0, 0]
baseline_m = abs(T[0][0])  # since objp used meters, T is in meters
print(f"Using focal f = {focal_length_px:.3f} px, baseline = {baseline_m:.6f} m")

# init undistort rectify maps
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, last_left_shape, cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, last_left_shape, cv2.CV_16SC2)

# create stereo matcher
# ensure num_disp is multiple of 16
if num_disp <= 0:
    num_disp = 16
num_disp = ((num_disp + 15) // 16) * 16

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=5,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2
)
stereoR = cv2.ximgproc.createRightMatcher(stereo)

# WLS filter
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

# open camera
Cam = cv2.VideoCapture(CAM_INDEX)
Cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
Cam.set(cv2.CAP_PROP_FPS, CAP_FPS)

executor = ThreadPoolExecutor(max_workers=4)
prev_time = time.time()
distance_history = deque(maxlen=DISP_AVG_HISTORY)

print("Starting main loop. Press ESC to exit.")

while True:
    ret, frame = Cam.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    height, width, _ = frame.shape
    mid = width // 2
    left_frame = frame[:, :mid].copy()
    right_frame = frame[:, mid:].copy()

    # rectification remap (use executor for speed)
    left_rect_f = executor.submit(cv2.remap, left_frame, Left_Stereo_Map[0], Left_Stereo_Map[1],
                                  interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    right_rect_f = executor.submit(cv2.remap, right_frame, Right_Stereo_Map[0], Right_Stereo_Map[1],
                                   interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    Left_rect = left_rect_f.result()
    Right_rect = right_rect_f.result()

    # downscale for faster disparity if desired
    Left_small = cv2.resize(Left_rect, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    Right_small = cv2.resize(Right_rect, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    gray_left = cv2.cvtColor(Left_small, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(Right_small, cv2.COLOR_BGR2GRAY)

    # compute disparity maps (SGBM gives 16x scaled disparities)
    dispL = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    dispR = stereoR.compute(gray_right, gray_left).astype(np.float32) / 16.0

    # WLS filter for cleaner disparity
    filtered_disp = wls_filter.filter(dispL, gray_left, None, dispR)

    # normalize for visualization
    disp_vis = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_closed = cv2.morphologyEx(disp_vis, cv2.MORPH_CLOSE, kernel)

    # ROI (you can adapt)
    disp_h, disp_w = disp_vis.shape
    roi_w = disp_w // 3
    roi_h = disp_h // 2
    roi_x = max(0, (disp_w - roi_w + 150) // 2)
    roi_y = max(0, (disp_h - roi_h) // 2)
    cv2.rectangle(disp_closed, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 255), 2)

    # find close objects by thresholding disparity (bigger disparity -> closer)
    _, close_mask = cv2.threshold(disp_vis, 160, 255, cv2.THRESH_BINARY)
    close_mask = cv2.morphologyEx(close_mask, cv2.MORPH_CLOSE, kernel)
    roi_mask = np.zeros_like(close_mask)
    roi_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = close_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_stop_flag = False
    measured_distance = None

    # reproject entire filtered disparity (float32) to 3D points in meters using Q
    # Note: reprojectImageTo3D expects disparity in same scale as used to compute Q; filtered_disp is in pixels (float)
    points_3d = cv2.reprojectImageTo3D(filtered_disp, Q)  # shape (h,w,3), units = meters because objp used meters
    valid_mask = (filtered_disp > DISPARITY_MIN) & (filtered_disp < DISPARITY_MAX) & np.isfinite(points_3d[:,:,2])

    for cnt in contours:
        if cv2.contourArea(cnt) < CONTOUR_AREA_THRESHOLD:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        # make sure indices within bounds
        if cy - DISP_MEDIAN_ROI < 0 or cy + DISP_MEDIAN_ROI >= points_3d.shape[0] or cx - DISP_MEDIAN_ROI < 0 or cx + DISP_MEDIAN_ROI >= points_3d.shape[1]:
            continue

        # extract small ROI around the center
        roi_pts = points_3d[cy - DISP_MEDIAN_ROI: cy + DISP_MEDIAN_ROI + 1,
                            cx - DISP_MEDIAN_ROI: cx + DISP_MEDIAN_ROI + 1].reshape(-1, 3)
        roi_mask_local = valid_mask[cy - DISP_MEDIAN_ROI: cy + DISP_MEDIAN_ROI + 1,
                                    cx - DISP_MEDIAN_ROI: cx + DISP_MEDIAN_ROI + 1].reshape(-1)

        valid_points = roi_pts[roi_mask_local]
        if valid_points.size == 0:
            continue

        # compute robust distance: median of Euclidean norms
        dists = np.linalg.norm(valid_points, axis=1)
        dist_median = float(np.median(dists))
        distance_history.append(dist_median)
        smoothed_distance = float(np.median(list(distance_history))) if len(distance_history) > 0 else dist_median
        measured_distance = smoothed_distance

        # decide detection / visualize
        if smoothed_distance < 0.65:  # your trigger
            current_stop_flag = True
            box_color = (0, 0, 255) if smoothed_distance < 0.5 else (0, 255, 0)
            cv2.rectangle(disp_closed, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(disp_closed, f"{smoothed_distance:.2f} m", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            break

    # show FPS and measured distance
    now = time.time()
    fps = 1.0 / (now - prev_time + 1e-9)
    prev_time = now
    cv2.putText(disp_closed, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    if measured_distance is not None:
        cv2.putText(disp_closed, f"Dist: {measured_distance:.2f} m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Filtered Color Depth", disp_closed)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# cleanup
Cam.release()
cv2.destroyAllWindows()
