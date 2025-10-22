import cv2
import numpy as np

# === Camera settings ===
CAM_INDEX = 0
CAP_WIDTH = 1100
CAP_HEIGHT = 270
CAP_FPS = 60

Cam = cv2.VideoCapture(CAM_INDEX)
Cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
Cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
Cam.set(cv2.CAP_PROP_FPS, CAP_FPS)

# === StereoSGBM parameters (tuned for good results) ===
min_disp = 0
num_disp = 16 * 8  # must be divisible by 16
block_size = 5
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=1,
    speckleWindowSize=100,
    speckleRange=32
)

while True:
    ret, frame = Cam.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Split frame into left and right halves
    height, width, _ = frame.shape
    mid = width // 2
    left_frame = frame[:, :mid].copy()
    right_frame = frame[:, mid:].copy()

    # Convert to grayscale
    grayL = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Compute disparity map
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Normalize for display
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # Optional: Apply colormap for visualization
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    # Show windows
    cv2.imshow("Left", left_frame)
    cv2.imshow("Right", right_frame)
    cv2.imshow("Depth Map", disp_color)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

Cam.release()
cv2.destroyAllWindows()
