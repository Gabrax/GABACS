import cv2
import numpy as np
import torch

# Load MiDaS model for monocular depth estimation
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Camera Calibration parameters (replace these with your actual calibration data)
mtx = np.array([[470.05444369,  0, 308.97585387],
 [  0, 475.76629567, 293.98694653],
 [  0, 0, 1 ]])      # Example: Camera Matrix (0, 0, 1)

dist = np.array([ 0.02960611, -0.69054411, 0.01329125, -0.0061207, 0.92698465])  # Example: Distortion Coefficients (k1, k2, p1, p2)

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort the frame using the camera calibration parameters
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_matrix)

    # Resize frame for faster processing
    frame_resized = cv2.resize(undistorted_frame, (256, 256))

    # Convert to tensor
    input_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float() / 255
    input_tensor = input_tensor.to(device)

    # Run depth estimation
    with torch.no_grad():
        depth_map = model(input_tensor)
    depth_map = depth_map.squeeze().cpu().numpy()

    # Normalize and convert to 8-bit for visualization
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Crop only the bottom center region (adjust size as needed)
    h, w = depth_map.shape
    roi = depth_map[int(h * 0.6):, int(w * 0.3):int(w * 0.7)]  # Keep central 40% width, bottom 40% height
    median_depth = np.median(roi)
    print(f"Estimated Distance: {median_depth:.2f} meters")

    # Display results
    cv2.imshow("Depth Map", depth_map)
    cv2.imshow("Cropped ROI", roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
