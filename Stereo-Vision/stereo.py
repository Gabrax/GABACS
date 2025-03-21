import numpy as np
import cv2
# from openpyxl import Workbook  # Used for writing data into an Excel file
from sklearn.preprocessing import normalize

# Filtering
kernel = np.ones((3, 3), np.uint8)

def coords_mouse_disp(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        average = 0
        for u in range(-1, 2):
            for v in range(-1, 2):
                average += disp[y + u, x + v]
        average = average / 9
        Distance = -593.97 * average**(3) + 1506.8 * average**(2) - 1373.1 * average + 522.06
        Distance = np.around(Distance * 0.01, decimals=2)
        print('Distance: ' + str(Distance) + ' m')

# This section has to be uncommented if you want to take measurements and store them in the excel
##        ws.append([counterdist, average])
##        print('Measure at ' + str(counterdist) + ' cm, the disparity is ' + str(average))
##        if (counterdist <= 85):
##            counterdist += 3
##        elif(counterdist <= 120):
##            counterdist += 5
##        else:
##            counterdist += 10
##        print('Next distance to measure: ' + str(counterdist) + ' cm')

# Mouseclick callback
# wb = Workbook()
# ws = wb.active  

#*************************************************
#***** Parameters for Distortion Calibration *****
#*************************************************

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []   # 3d points in real world space
imgpointsR = []   # 2d points in image plane
imgpointsL = []

# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')
# Call all saved images
for i in range(0, 4):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) when starting from the image number 0
    t = str(i)
    ChessImaR = cv2.imread('right_chessboard-' + t + '.png', 0)    # Right side
    ChessImaL = cv2.imread('left_chessboard-' + t + '.png', 0)    # Left side
    retR, cornersR = cv2.findChessboardCorners(ChessImaR, (9, 6), None)  # Define the number of chessboard corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL, (9, 6), None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(ChessImaL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# Determine the new values for different parameters
# Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1], None, None)
hR, wR = ChessImaR.shape[:2]
OmtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR,
                                                   (wR, hR), 1, (wR, hR))

# Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1], None, None)
hL, wL = ChessImaL.shape[:2]
OmtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

print('Cameras Ready to use')

#********************************************
#***** Calibrate the Cameras for Stereo *****
#********************************************

# StereoCalibrate function
# flags = 0
# flags |= cv2.CALIB_FIX_INTRINSIC
# flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
# flags |= cv2.CALIB_USE_INTRINSIC_GUESS
# flags |= cv2.CALIB_FIX_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_ASPECT_RATIO
# flags |= cv2.CALIB_ZERO_TANGENT_DIST
# flags |= cv2.CALIB_RATIONAL_MODEL
# flags |= cv2.CALIB_SAME_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_K3
# flags |= cv2.CALIB_FIX_K4
# flags |= cv2.CALIB_FIX_K5
retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          ChessImaR.shape[::-1],
                                                          criteria=criteria_stereo,
                                                          flags=cv2.CALIB_FIX_INTRINSIC)

# StereoRectify function
rectify_scale = 0  # if 0 image cropped, if 1 image not cropped
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale, (0, 0))  # last parameter is alpha, if 0= cropped, if 1= not cropped
# initUndistortRectifyMap function
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables the program to work faster
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)
#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=5,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2)

# Used for the filtered image
stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

#*************************************
#***** Starting the StereoVision *****
#*************************************

# Call the single stereo camera (example for ZED-like cameras)
Cam = cv2.VideoCapture(0)  # Single camera index, assuming it's a stereo camera
Cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
Cam.set(cv2.CAP_PROP_FPS, 60)

while True:
    # Start Reading Camera images
    ret, frame = Cam.read()  # Read from the stereo camera

    if not ret:
        print("Failed to capture image")
        break

    # Assuming that the frame is already split into two (left and right lenses)
    height, width = frame.shape[:2]
    half_width = width // 2

    # Split the frame into left and right lenses
    left_frame = frame[:, :half_width]  # Left lens
    right_frame = frame[:, half_width:]  # Right lens

    # Stack the frames horizontally to display side by side
    combined_frame = np.hstack([left_frame, right_frame])

    # Show the combined image (split into two parts)
    cv2.imshow('Stereo View', combined_frame)

    # Convert from color(BGR) to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the 2 images for the Depth_image
    disp = stereo.compute(gray, gray)  # Depth map
    dispL = disp
    dispR = stereoR.compute(gray, gray)
    dispL = np.int16(dispL)
    dispR = np.int16(dispR)

    # Using the WLS filter
    filteredImg = wls_filter.filter(dispL, gray, None, dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    # Show the result for the Depth_image
    cv2.imshow('Filtered Color Depth', filteredImg)

    # Mouse click callback
    cv2.setMouseCallback("Filtered Color Depth", coords_mouse_disp, filteredImg)

    # End the Program if spacebar is pressed
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Release the Camera and close all windows
Cam.release()
cv2.destroyAllWindows()
