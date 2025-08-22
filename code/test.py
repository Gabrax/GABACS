import numpy as np
import cv2
import time
from sklearn.preprocessing import normalize  # (left from original, not required now)
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# =====================
# Parametry i ustawienia
# =====================
SCALE_FOR_SGBM = 0.5            # współczynnik zmiany rozdzielczości przed StereoSGBM
USE_BILATERAL = True            # filtr bilateralny przed SGBM
BILATERAL_DIAM = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

USE_WLS = True                  # WLS filtering po SGBM
WLS_LAMBDA = 80000
WLS_SIGMA = 1.8

# Histereza decyzji odległości
MIN_DISTANCE_TRIGGER = 0.65     # m
MAX_DISTANCE_RELEASE = 0.70     # m
DISPARITY_RANGE = (1.0, 150.0)  # do filtrowania outlierów
CONTOUR_AREA_THRESHOLD = 500
DISP_AVG_HISTORY = 5

# Maska tła (background subtraction)
USE_BG_SUBTRACTOR = True
BG_HISTORY = 300
BG_VAR_THRESHOLD = 16

# Wizualizacja / mapa głębokości
SHOW_DEPTH_MAP = True
MAX_DEPTH_VIS = 4.0   # m – zakres do wizualizacji (kolorowanie)

# =====================
# Narzędzia pomocnicze
# =====================
class ScalarKalman:
    """Prosty filtr Kalmana dla skalarnego pomiaru odległości."""
    def __init__(self, process_var=1e-3, meas_var=1e-2, init=1.0):
        self.x = init
        self.P = 1.0
        self.Q = process_var
        self.R = meas_var

    def update(self, z):
        # Predict (tożsamościowy model przejścia)
        self.P += self.Q
        # Update
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x


def depth_from_disparity(disparity, focal_px, baseline_m):
    """Z = f * B / d (metry). Zwraca macierz float32, inf dla d<=0."""
    disp = disparity.astype(np.float32)
    Z = np.full_like(disp, np.inf, dtype=np.float32)
    valid = disp > 0.0
    Z[valid] = (focal_px * baseline_m) / disp[valid]
    return Z


def colorize_depth(depth_m, max_depth=4.0):
    """Prosta wizualizacja: bliżej jaśniej. Zwraca obraz uint8 do pokazania."""
    depth = depth_m.copy()
    depth[~np.isfinite(depth)] = max_depth
    depth = np.clip(depth, 0, max_depth)
    inv = (max_depth - depth) / max_depth  # bliżej -> większa wartość
    vis = (inv * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    return vis


def mode_disparity(values, bin_width=0.25):
    """Zwrot modalnej dysparycji (najczęstszej), przez histogram z zadanym kosztem binów."""
    if values.size == 0:
        return None
    vmin, vmax = np.min(values), np.max(values)
    if vmin == vmax:
        return float(vmin)
    bins = int(max(1, np.ceil((vmax - vmin) / bin_width)))
    hist, edges = np.histogram(values, bins=bins, range=(vmin, vmax))
    idx = np.argmax(hist)
    # środek kosza
    return float(0.5 * (edges[idx] + edges[idx + 1]))


# =====================
# Kalibracja i rektyfikacja
# =====================
print('Starting stereo calibration ... ')

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

retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    ChessImaR.shape[::-1], criteria=criteria_stereo,
    flags=cv2.CALIB_FIX_INTRINSIC
)

print('Calibration complete')

# Rektyfikacja
rectify_scale = 0  # 0 = crop, 1 = no crop
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(
    MLS, dLS, MRS, dRS, ChessImaR.shape[::-1], R, T, rectify_scale, (0, 0)
)

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

wls_filter = None
if USE_WLS:
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

# Background subtractor
bg_subtractor = None
if USE_BG_SUBTRACTOR:
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESHOLD, detectShadows=False
    )

# Bufor i filtr Kalmana
distance_history = deque(maxlen=DISP_AVG_HISTORY)
kalman = ScalarKalman(process_var=1e-3, meas_var=5e-3, init=1.0)

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
    Left_nice = executor.submit(
        cv2.remap, left_frame, Left_Stereo_Map[0], Left_Stereo_Map[1],
        interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT
    ).result()
    Right_nice = executor.submit(
        cv2.remap, right_frame, Right_Stereo_Map[0], Right_Stereo_Map[1],
        interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT
    ).result()

    # Skala dla SGBM
    if SCALE_FOR_SGBM != 1.0:
        small_left = cv2.resize(Left_nice, None, fx=SCALE_FOR_SGBM, fy=SCALE_FOR_SGBM, interpolation=cv2.INTER_AREA)
        small_right = cv2.resize(Right_nice, None, fx=SCALE_FOR_SGBM, fy=SCALE_FOR_SGBM, interpolation=cv2.INTER_AREA)
    else:
        small_left, small_right = Left_nice, Right_nice

    # Szarość
    gray_left = cv2.cvtColor(small_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(small_right, cv2.COLOR_BGR2GRAY)

    # (2) Filtr bilateralny przed SGBM
    if USE_BILATERAL:
        gray_left = cv2.bilateralFilter(gray_left, BILATERAL_DIAM, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
        gray_right = cv2.bilateralFilter(gray_right, BILATERAL_DIAM, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

    # Dysparycja (na obrazach zeskalowanych)
    dispL = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    dispR = stereoR.compute(gray_right, gray_left).astype(np.float32) / 16.0

    # WLS filtr
    if USE_WLS and wls_filter is not None:
        filtered_disp = wls_filter.filter(dispL, gray_left, None, dispR)
    else:
        filtered_disp = dispL

    # Wizualizacja dysparycji
    disp_vis = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # (6) Maska tła z lewego obrazu (na tej samej skali)
    fgmask = None
    if USE_BG_SUBTRACTOR:
        fgmask = bg_subtractor.apply(gray_left)
        # porządki: erozja/dylatacja by ściąć szum
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Dodatkowe domknięcie na wizualizacji (do prostszej segmentacji bliskich obiektów)
    disp_closed = cv2.morphologyEx(disp_vis, cv2.MORPH_CLOSE, kernel)

    # ROI do analizy (na rozmiarze SGBM)
    disp_height, disp_width = disp_vis.shape
    roi_width = disp_width // 3
    roi_height = disp_height // 2
    roi_x = (disp_width - roi_width + 150) // 2
    roi_y = (disp_height - roi_height) // 2

    # Wizualizacja ROI
    cv2.rectangle(disp_closed, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 255), 2)

    # Prosty próg na "blisko" (na obrazie 8-bit po normalizacji)
    _, close_mask = cv2.threshold(disp_closed, 160, 255, cv2.THRESH_BINARY)

    # Ograniczenie do ROI
    roi_mask = np.zeros_like(close_mask)
    roi_mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = close_mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Połączenie z maską tła (jeśli włączona)
    if fgmask is not None:
        roi_mask = cv2.bitwise_and(roi_mask, fgmask)

    # Kontury w ROI
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_stop_flag = False

    # (3) Skalowanie ogniskowej do rozmiaru SGBM
    focal_length_px = focal_length_px_full * SCALE_FOR_SGBM

    # (4) Mapa głębokości Z
    depth_map_m = depth_from_disparity(np.maximum(filtered_disp, 0), focal_length_px, baseline_m)

    # (1) Projekcja kształtu (maski konturu) na mapę dysparycji + modalna wartość

    for cnt in contours:

        if cv2.contourArea(cnt) < CONTOUR_AREA_THRESHOLD:
            continue

        # Maska pojedynczego konturu (na rozmiarze SGBM)
        mask = np.zeros_like(disp_vis, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, color=255, thickness=cv2.FILLED)

        # Zabezpieczenie: ograniczamy do ROI, żeby nie wyjść poza
        mask_roi = np.zeros_like(mask)
        mask_roi[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = 255
        mask = cv2.bitwise_and(mask, mask_roi)

        # Pobranie dysparycji wewnątrz konturu z ograniczeniem zakresu
        disp_vals = filtered_disp[mask == 255]
        valid_disp = disp_vals[(disp_vals > DISPARITY_RANGE[0]) & (disp_vals < DISPARITY_RANGE[1])]
        if valid_disp.size == 0:
            continue

        # Modalna dysparycja przez histogram (stabilniejsza niż mediana przy jednolitych powierzchniach)
        modal_disp = mode_disparity(valid_disp, bin_width=0.25)
        if modal_disp is None or modal_disp <= 0:
            continue

        # Odległość z modalnej dysparycji
        distance = (focal_length_px * baseline_m) / modal_disp

        # Bufor + Kalman (5)
        distance_history.append(distance)
        # używamy mediany do pomiaru wejściowego Kalmana, by zbić outliery, a Kalman da płynność
        z_meas = float(np.median(distance_history))
        kalman_distance = kalman.update(z_meas)

        # Decyzja z histerezą
        if kalman_distance < MIN_DISTANCE_TRIGGER:
            current_stop_flag = True

        # Wizualizacja konturu + tekst
        x, y, w, h = cv2.boundingRect(cnt)
        box_color = (0, 0, 255) if kalman_distance < 0.5 else (0, 255, 0)
        cv2.rectangle(disp_closed, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(disp_closed, f"{kalman_distance:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        break  # jeden najbliższy kontur na klatkę

    # (opcjonalnie) sterowanie GPIO – pozostawione jak w oryginale (zakomentowane)
    # if current_stop_flag:
    #     lgpio.gpio_write(chip, PIN, 0)  # Stop
    # elif len(distance_history) == DISP_AVG_HISTORY and kalman_distance > MAX_DISTANCE_RELEASE:
    #     lgpio.gpio_write(chip, PIN, 1)  # Move forward

    # FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time + 1e-6)
    prev_time = current_time

    # Napisy na ekranie
    cv2.putText(disp_closed, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(disp_closed, f"Size (SGBM): {disp_width}x{disp_height}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Pokazywanie okien
    cv2.imshow('Filtered Disparity (debug)', disp_closed)

    if SHOW_DEPTH_MAP:
        depth_vis = colorize_depth(depth_map_m, MAX_DEPTH_VIS)
        # dopasuj rozmiar do okna dysparycji (jeśli skala < 1)
        if depth_vis.shape[:2] != disp_closed.shape[:2]:
            depth_vis = cv2.resize(depth_vis, (disp_closed.shape[1], disp_closed.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Depth Map (m, pseudo-color)', depth_vis)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

Cam.release()
cv2.destroyAllWindows()
