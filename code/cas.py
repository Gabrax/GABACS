import numpy as np
import cv2
import time
from sklearn.preprocessing import normalize  # (left from original, not required now)
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import lgpio

chip = lgpio.gpiochip_open(0)
PIN = 17
lgpio.gpio_claim_output(chip,PIN,1)


# =====================
# Parametry i ustawienia
# =====================
SCALE_FOR_SGBM = 0.5            # współczynnik zmiany rozdzielczości przed StereoSGBM
BILATERAL_DIAM = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

WLS_LAMBDA = 80000
WLS_SIGMA = 1.8

# Histereza decyzji odległości
DISPARITY_RANGE = (1.0, 150.0)  # do filtrowania outlierów
CONTOUR_AREA_THRESHOLD = 500
DISP_AVG_HISTORY = 5

# =====================
# Kalibracja i rektyfikacja
# =====================
print('Rozpoczynanie kalibracji ... ')

# Uwaga: sekcja kalibracji pozostaje jak w oryginale – zakładamy obrazy w 'calib_images/'
kernel = np.ones((3, 3), np.uint8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints, imgpointsR, imgpointsL = [], [], []

ChessImaR = None
ChessImaL = None
for i in range(0, 64):
    t = str(i)
    ChessImaR = cv2.imread('calib_images/right_chessboard-' + t + '.png', 0)
    ChessImaL = cv2.imread('calib_images/left_chessboard-' + t + '.png', 0)
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

retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, ChessImaR.shape[::-1], None, None)
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, ChessImaL.shape[::-1], None, None)
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, ChessImaR.shape[::-1], 1, ChessImaR.shape[::-1])
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, ChessImaR.shape[::-1], 1, ChessImaR.shape[::-1])

retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR,mtxL, distL, mtxR, distR,ChessImaR.shape[::-1], criteria=criteria_stereo,flags=cv2.CALIB_FIX_INTRINSIC)

print('Calibration complete')

# Rektyfikacja
rectify_scale = 0  # 0 = crop, 1 = no crop
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, ChessImaR.shape[::-1], R, T, rectify_scale, (0, 0))

Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, ChessImaR.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, ChessImaR.shape[::-1], cv2.CV_16SC2)

# =====================
# StereoSGBM + WLS
# =====================
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
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2
)
stereoR = cv2.ximgproc.createRightMatcher(stereo)

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(WLS_LAMBDA)
wls_filter.setSigmaColor(WLS_SIGMA)

# =====================
# Kamera i pomocnicze
# =====================
Cam = cv2.VideoCapture(0)
Cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)
Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
Cam.set(cv2.CAP_PROP_FPS, 60)

prev_time = 0
executor = ThreadPoolExecutor(max_workers=4)

# Ogniskowa i baza do przeliczeń (skalowane później o SCALE_FOR_SGBM)
focal_length_px_full = PL[0, 0]
baseline_m = abs(T[0][0]) / 100.0  # założenie: T w cm -> m

# Bufor
distance_history = deque(maxlen=DISP_AVG_HISTORY)

# =====================
# Główna pętla
# =====================
while True:
    ret, frame = Cam.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    height, width, _ = frame.shape
    mid = width // 2

    left_frame = frame[:, :mid]
    right_frame = frame[:, mid:]

    # Rektyfikacja do rozmiaru pełnego
    Left_rectified = executor.submit(cv2.remap, left_frame, Left_Stereo_Map[0], Left_Stereo_Map[1],
                                interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT).result()
    Right_rectified = executor.submit(cv2.remap, right_frame, Right_Stereo_Map[0], Right_Stereo_Map[1],
                                 interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT).result()

    # Skala dla SGBM
    Left_small = cv2.resize(Left_rectified, None, fx=SCALE_FOR_SGBM, fy=SCALE_FOR_SGBM, interpolation=cv2.INTER_AREA)
    Right_small = cv2.resize(Right_rectified, None, fx=SCALE_FOR_SGBM, fy=SCALE_FOR_SGBM, interpolation=cv2.INTER_AREA)

    # Szarość
    Left_gray = cv2.cvtColor(Left_small, cv2.COLOR_BGR2GRAY)
    Right_gray = cv2.cvtColor(Right_small, cv2.COLOR_BGR2GRAY)

    # Filtr bilateralny
    Left_gray = cv2.bilateralFilter(Left_gray, BILATERAL_DIAM, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    Right_gray = cv2.bilateralFilter(Right_gray, BILATERAL_DIAM, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

    # Dysparycja
    Left_disparity = stereo.compute(Left_gray, Right_gray).astype(np.float32) / 16.0
    Right_disparity = stereoR.compute(Right_gray, Left_gray).astype(np.float32) / 16.0

    # Filtr WLS
    WLS_disparity = wls_filter.filter(Left_disparity, Left_gray, None, Right_disparity)

    # Skalowanie ogniskowej do rozmiaru SGBM
    focal_length_px = focal_length_px_full * SCALE_FOR_SGBM

    # Mapa głębokości
    # Z = f * B / d (metry). Zwraca macierz float32, inf dla d<=0.
    disp = np.maximum(WLS_disparity, 0).astype(np.float32)
    Depth_Map = np.full_like(disp, np.inf, dtype=np.float32)
    valid = disp > 0.0
    Depth_Map[valid] = (focal_length_px * baseline_m) / disp[valid]

    # Przygotowanie obrazu do wizualizacji
    disp_vis = np.clip(WLS_disparity, min_disp, min_disp + num_disp)
    disp_vis = ((disp_vis - min_disp) / float(num_disp) * 255.0).astype(np.uint8)

    # ROI dla analizy mapy głębokości (ta sama skala co SGBM)
    depth_height, depth_width = Depth_Map.shape
    roi_width = depth_width // 3
    roi_height = depth_height // 2
    roi_x = (depth_width - roi_width + 150) // 2
    roi_y = (depth_height - roi_height) // 2

    # Utwórz maskę binarną dla obszarów „bliskich” (< 0,20 m)
    Binary_Mask = np.zeros_like(Depth_Map, dtype=np.uint8)
    Binary_Mask[(Depth_Map < 0.20) & (Depth_Map > 0)] = 255  # only valid finite depths

    # Ogranicz wyłącznie do ROI
    ROI_Mask = np.zeros_like(Binary_Mask)
    ROI_Mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = Binary_Mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Wyczyść małe plamki szumu
    ROI_Mask = cv2.morphologyEx(ROI_Mask, cv2.MORPH_OPEN, kernel, iterations=1)
    ROI_Mask = cv2.morphologyEx(ROI_Mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    current_stop_flag = False

    # Znajdź kontury w ROI
    Found_Contours, _ = cv2.findFound_Contours(ROI_Mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in Found_Contours:
        if cv2.contourArea(cnt) < CONTOUR_AREA_THRESHOLD:
            continue

        # Ramka ograniczająca
        x, y, w, h = cv2.boundingRect(cnt)

        # Oblicz medianę głębokości wewnątrz konturu
        mask = np.zeros_like(Depth_Map, dtype=np.uint8)
        cv2.drawFound_Contours(mask, [cnt], -1, 255, -1)
        depth_vals = Depth_Map[mask == 255]
        valid_depth = depth_vals[(depth_vals > 0) & np.isfinite(depth_vals)]
        if valid_depth.size == 0:
            continue

        distance = np.median(valid_depth)
        distance_history.append(distance)
        avg_distance = float(np.median(distance_history))

        if avg_distance < 0.20:
            current_stop_flag = True
        else:
            current_stop_flag = False

        color = (0, 0, 255) if current_stop_flag else (0, 255, 0)
        cv2.rectangle(disp_vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(disp_vis, f"{avg_distance:.2f} m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        break  # Tylko pierwszy kontur

    if current_stop_flag == True:
        lgpio.gpio_write(chip, PIN, 0)
    else:
        lgpio.gpio_write(chip, PIN, 1)

    # Wizualizacja ROI
    cv2.rectangle(disp_vis, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 255, 0), 2)

    # Pokaż mapę głębi
    window_name = 'Depth Map (m, pseudo-color)'
    cv2.imshow(window_name, disp_vis)

    # Callback myszy do sprawdzania głębi
    def show_depth_value(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            depth_value = Depth_Map[y, x]
            if np.isfinite(depth_value) and depth_value > 0:
                text = f"({x},{y}) depth: {depth_value:.2f} m"
                cv2.putText(disp_vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(window_name, disp_vis)

    cv2.setMouseCallback(window_name, show_depth_value)

    if cv2.waitKey(1) & 0xFF == 27:
        break
Cam.release()
cv2.destroyAllWindows()
