import numpy as np
import cv2
import time
# from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize

# Filtering
kernel= np.ones((3,3),np.uint8)

def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9
        Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        Distance= np.around(Distance*0.01,decimals=2)
        print('Distance: '+ str(Distance)+' m')

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

print('Calibration complete')

retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,imgpointsL,imgpointsR,mtxL,distL,mtxR,distR,ChessImaR.shape[::-1],criteria = criteria_stereo,flags = cv2.CALIB_FIX_INTRINSIC)

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
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

Cam = cv2.VideoCapture(0)
Cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  
Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
Cam.set(cv2.CAP_PROP_FPS, 60)

prev_time = time.time()

while True:
    current_time = time.time()
    ret, frame= Cam.read()
    if not ret:
        print("❌ Failed to capture frame. Exiting...")
        break

    height, width, _ = frame.shape
    mid = width // 2  # Middle point for splitting

    left_frame = frame[:, :mid]   # Left camera view
    right_frame = frame[:, mid:]  # Right camera view


    Left_nice= cv2.remap(left_frame,Left_Stereo_Map[0],Left_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)  
    Right_nice= cv2.remap(right_frame,Right_Stereo_Map[0],Right_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)

    gray_left = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)

    disp= stereo.compute(gray_left,gray_right)#.astype(np.float32)/ 16
    dispL= disp
    dispR= stereoR.compute(gray_right,gray_left)
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)

    #cv2.imshow('Disparity', disp)
    # closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 
    #cv2.imshow('Closing',closing)
    # dispc= (closing-closing.min())*255
    # dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    # disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
    #cv2.imshow('Color Depth',disp_Color)
    disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect
    # roi_mask = np.zeros_like(gray_left, dtype=np.uint8)
    # cv2.rectangle(roi_mask, (100, 100), (300, 300), 255, -1)  # Only process this region
    filteredImg= wls_filter.filter(dispL,gray_left,None,dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    #cv2.imshow('Disparity Map', filteredImg)

    filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN)

    # --- Close Object Detection ---
    _, close_mask = cv2.threshold(filteredImg, 160, 255, cv2.THRESH_BINARY)
    close_mask = cv2.morphologyEx(close_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(close_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)

            # Use RAW disparity map
            roi_disp = disp[y:y + h, x:x + w].astype(np.float32)

            # Center 3x3 window
            cx, cy = x + w // 2, y + h // 2
            sample_disp = disp[cy - 1:cy + 2, cx - 1:cx + 2].astype(np.float32)
            average_disp = np.mean(sample_disp[sample_disp > 0])

            if average_disp > 0:
                # Convert disparity to distance
                distance = -593.97 * average_disp**3 + 1506.8 * average_disp**2 - 1373.1 * average_disp + 522.06
                distance = np.around(distance * 0.01, decimals=2)

                if distance < 1.0:
                    # Choose color based on distance
                    if distance < 0.5:
                        box_color = (0, 0, 255)  # Red for very close
                    else:
                        box_color = (0, 255, 0)  # Green for safe close

                    cv2.rectangle(filt_Color, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(filt_Color, f"{distance:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    # --------------------------------
    
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(filt_Color, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('Filtered Color Depth', filt_Color)
    cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)

    if cv2.waitKey(1) & 0xFF == 27:
        break

Cam.release()
cv2.destroyAllWindows()
