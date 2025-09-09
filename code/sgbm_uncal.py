import numpy as np
import cv2
import time
from sklearn.preprocessing import normalize
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# --- User-settable parameters (IMPORTANT) ---
# For metric distance you must know focal length (px) and baseline (meters).
# If unknown, set approximate values or obtain them from calibration.
focal_length_px = 700.0    # <-- set to your camera's focal length in pixels (approx)
baseline_m = 0.12          # <-- set to your stereo baseline in meters (approx)

MIN_DISTANCE_TRIGGER = 0.65
MAX_DISTANCE_RELEASE = 0.7
DISPARITY_RANGE = (1.0, 150.0)
CONTOUR_AREA_THRESHOLD = 500
DISP_AVG_HISTORY = 5

# --- image processing params ---
kernel = np.ones((3, 3), np.uint8)
window_size = 7
min_disp = 2
num_disp = 130 - min_disp

# --- stereo SGBM + WLS as before ---
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=5,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
)
stereoR = cv2.ximgproc.createRightMatcher(stereo)

lmbda = 80000
sigma = 1.8
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

# --- capture setup ---
Cam = cv2.VideoCapture(0)
Cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)
Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
Cam.set(cv2.CAP_PROP_FPS, 60)

executor = ThreadPoolExecutor(max_workers=4)
prev_time = 0.0
distance_history = deque(maxlen=DISP_AVG_HISTORY)

# --- helper: compute homographies using Hartley (uncalibrated) ---
def compute_uncalib_rectification(left_gray, right_gray, debug=False):
    """
    Detect features, match, compute fundamental matrix and obtain H1,H2 via stereoRectifyUncalibrated.
    Returns (H1, H2) or (None, None) if failed.
    """
    # ORB detector
    orb = cv2.ORB_create(5000)
    kps1, des1 = orb.detectAndCompute(left_gray, None)
    kps2, des2 = orb.detectAndCompute(right_gray, None)

    if des1 is None or des2 is None or len(kps1) < 8 or len(kps2) < 8:
        if debug:
            print("Not enough keypoints/descriptors for uncalibrated rectification.")
        return None, None

    # BFMatcher with Hamming (ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test (Lowe)
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        if debug:
            print(f"Not enough good matches: {len(good)}")
        return None, None

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good])

    # Estimate fundamental matrix with RANSAC to get inliers
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None or F.shape != (3, 3):
        if debug:
            print("Failed to estimate Fundamental matrix.")
        return None, None

    # Keep only inlier correspondences
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]

    if len(inliers1) < 8:
        if debug:
            print(f"Not enough inliers after RANSAC: {len(inliers1)}")
        return None, None

    # stereoRectifyUncalibrated expects points in pixel coordinates and F
    imgSize = (left_gray.shape[1], left_gray.shape[0])
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(
        inliers1, inliers2, F, imgSize
    )
    if not retval:
        if debug:
            print("stereoRectifyUncalibrated failed to compute homographies.")
        return None, None

    return H1, H2

# --- main loop ---
while True:
    current_time = time.time()
    ret, frame = Cam.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    height, width, _ = frame.shape
    mid = width // 2
    left_frame = frame[:, :mid].copy()
    right_frame = frame[:, mid:].copy()

    # Convert to grayscale for feature detection / rectification
    gray_left_full = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right_full = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Compute H1,H2 using uncalibrated Hartley approach (once per frame or you may cache temporally)
    H1, H2 = compute_uncalib_rectification(gray_left_full, gray_right_full, debug=False)

    if H1 is not None and H2 is not None:
        # Warp the original color frames using the homographies to get rectified pairs
        # Use same output size as each half-frame
        out_size = (left_frame.shape[1], left_frame.shape[0])
        Left_rect_color = cv2.warpPerspective(left_frame, H1, out_size, flags=cv2.INTER_LINEAR)
        Right_rect_color = cv2.warpPerspective(right_frame, H2, out_size, flags=cv2.INTER_LINEAR)
    else:
        # If rectification failed, fall back to raw halves (warn)
        # (Better: you could reattempt with different feature types or lower thresholds)
        Left_rect_color = left_frame
        Right_rect_color = right_frame
        # optional: print a single-time warning or log
        # print("Warning: Uncalibrated rectification failed; using raw images.")

    # Downscale (same as your original pipeline)
    small_left = executor.submit(lambda: cv2.resize(Left_rect_color, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    small_right = executor.submit(lambda: cv2.resize(Right_rect_color, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    Left_small = small_left.result()
    Right_small = small_right.result()

    # Gray for stereo
    gray_left = executor.submit(cv2.cvtColor, Left_small, cv2.COLOR_BGR2GRAY)
    gray_right = executor.submit(cv2.cvtColor, Right_small, cv2.COLOR_BGR2GRAY)
    Left_gray = gray_left.result()
    Right_gray = gray_right.result()

    # Compute disparity (left and right)
    left_future = executor.submit(lambda: stereo.compute(Left_gray, Right_gray).astype(np.float32) / 16.0)
    right_future = executor.submit(lambda: stereoR.compute(Right_gray, Left_gray).astype(np.float32) / 16.0)
    Left_disp = left_future.result()
    Right_disp = right_future.result()

    # WLS filtering
    filtered_disp = wls_filter.filter(Left_disp, Left_gray, None, Right_disp)

    # Closing filter and visualization normalization
    disp_closed = cv2.morphologyEx(filtered_disp, cv2.MORPH_CLOSE, kernel)
    disp_vis = cv2.normalize(disp_closed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # region of interest (same logic as your original)
    disp_height, disp_width = disp_vis.shape
    roi_width = disp_width // 3
    roi_height = disp_height // 2
    roi_x = (disp_width - roi_width + 150) // 2
    roi_y = (disp_height - roi_height) // 2
    cv2.rectangle(disp_vis, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 255), 2)

    _, close_mask = cv2.threshold(disp_vis, 160, 255, cv2.THRESH_BINARY)
    roi_mask = np.zeros_like(close_mask)
    roi_mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = close_mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_stop_flag = False

    for cnt in contours:
        if cv2.contourArea(cnt) < CONTOUR_AREA_THRESHOLD:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2

        # make sure indices inside disparity map
        if cy - 1 < 0 or cy + 2 > Left_disp.shape[0] or cx - 1 < 0 or cx + 2 > Left_disp.shape[1]:
            continue

        roi_disp = Left_disp[cy - 1:cy + 2, cx - 1:cx + 2]
        valid_disp = roi_disp[(roi_disp > DISPARITY_RANGE[0]) & (roi_disp < DISPARITY_RANGE[1])]
        if valid_disp.size == 0:
            continue

        avg_disp = np.median(valid_disp)
        # distance calculation: requires focal_length_px and baseline_m to be meaningful
        distance = (focal_length_px * baseline_m) / (avg_disp + 1e-6)
        distance_history.append(distance)
        smoothed_distance = np.median(distance_history)

        if smoothed_distance < MIN_DISTANCE_TRIGGER:
            current_stop_flag = True
            box_color = (0, 0, 255) if smoothed_distance < 0.5 else (0, 255, 0)
            cv2.rectangle(disp_vis, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(disp_vis, f"{smoothed_distance:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            break

    # optional GPIO control omitted (same as original)
    current_time = time.time()
    fps = 1 / (current_time - prev_time + 1e-6)
    prev_time = current_time

    cv2.putText(disp_vis, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(disp_vis, f"Size: {disp_width}x{disp_height}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('Filtered Color Depth', disp_vis)

    if cv2.waitKey(1) & 0xFF == 27:
        break

Cam.release()
cv2.destroyAllWindows()
