import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor

# Enable OpenCL
cv2.ocl.setUseOpenCL(True)
print("OpenCL enabled:", cv2.ocl.useOpenCL())

# Filtering
kernel = np.ones((3, 3), np.uint8)

def estimate_distance(disparity_value):
    distance = -593.97 * disparity_value**3 + 1506.8 * disparity_value**2 - 1373.1 * disparity_value + 522.06
    return np.around(distance * 0.01, decimals=2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objpoints, imgpointsR, imgpointsL = [], [], []

print('Starting stereo calibration ...')

for i in range(0, 64):
    t = str(i)
    ChessImaR = cv2.imread(f'calib_images/right_chessboard-{t}.png', 0)
    ChessImaL = cv2.imread(f'calib_images/left_chessboard-{t}.png', 0)
    if ChessImaR is None or ChessImaL is None:
        print(f"⚠️ Warning: Image {t} could not be loaded.")
        continue
    retR, cornersR = cv2.findChessboardCorners(ChessImaR, (9, 6), None)
    retL, cornersL = cv2.findChessboardCorners(ChessImaL, (9, 6), None)
    if retR and retL:
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, ChessImaR.shape[::-1], None, None)
retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, ChessImaL.shape[::-1], None, None)
OmtxR, _ = cv2.getOptimalNewCameraMatrix(mtxR, distR, ChessImaR.shape[::-1], 1, ChessImaR.shape[::-1])
OmtxL, _ = cv2.getOptimalNewCameraMatrix(mtxL, distL, ChessImaL.shape[::-1], 1, ChessImaL.shape[::-1])

print('Calibration complete')

_, MLS, dLS, MRS, dRS, R, T, _, _ = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    ChessImaR.shape[::-1], criteria=criteria_stereo, flags=cv2.CALIB_FIX_INTRINSIC
)

rectify_scale = 0
RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(
    MLS, dLS, MRS, dRS,
    ChessImaR.shape[::-1], R, T,
    rectify_scale, (0, 0)
)

Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, ChessImaR.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, ChessImaR.shape[::-1], cv2.CV_16SC2)

window_size = 7
min_disp = 2
num_disp = 130 - min_disp
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=5,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

stereoR = cv2.ximgproc.createRightMatcher(stereo)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(80000)
wls_filter.setSigmaColor(1.8)

Cam = cv2.VideoCapture(0)
Cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
Cam.set(cv2.CAP_PROP_FPS, 60)

prev_time = 0
executor = ThreadPoolExecutor(max_workers=4)

while True:
    current_time = time.time()
    ret, frame = Cam.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    height, width, _ = frame.shape
    mid = width // 2
    LFrame, RFrame = frame[:, :mid], frame[:, mid:]

    # Convert to UMat for OpenCL acceleration
    LFrame_u = cv2.UMat(LFrame)
    RFrame_u = cv2.UMat(RFrame)

    Left_nice = cv2.remap(LFrame_u, *Left_Stereo_Map, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    Right_nice = cv2.remap(RFrame_u, *Right_Stereo_Map, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

    Gray_left = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
    Gray_right = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)

    DispL = stereo.compute(Gray_left, Gray_right)
    DispR = stereoR.compute(Gray_right, Gray_left)

    dispL_np = DispL.get().astype(np.float32) / 16
    dispR_np = DispR.get().astype(np.float32) / 16

    filteredImg = wls_filter.filter(DispL, Gray_left, None, DispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg.get())

    filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

    # Close object detection
    _, close_mask = cv2.threshold(filteredImg, 160, 255, cv2.THRESH_BINARY)
    close_mask = cv2.morphologyEx(close_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(close_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    disp_norm = (dispL_np - min_disp) / num_disp

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            sample_disp = disp_norm[cy - 1:cy + 2, cx - 1:cx + 2]
            average_disp = np.mean(sample_disp[sample_disp > 0])
            if average_disp > 0:
                distance = estimate_distance(average_disp * num_disp + min_disp)
                if distance < 1.0:
                    box_color = (0, 0, 255) if distance < 0.5 else (0, 255, 0)
                    cv2.rectangle(filt_Color, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(filt_Color, f"{distance:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(filt_Color, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Filtered Color Depth", filt_Color)

    if cv2.waitKey(1) & 0xFF == 27:
        break

Cam.release()
cv2.destroyAllWindows()
