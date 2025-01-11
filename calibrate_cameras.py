import cv2
import numpy as np
import glob
import os

# Set the size of the checkerboard (number of inner corners per row and column)
CHECKERBOARD = (9, 6)  # Adjust based on your calibration pattern
square_size = 2.54  # Size of a square in your defined unit (e.g., 1.0 for 1 cm)

# Termination criteria for corner sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points based on the checkerboard size
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Path to the folder containing calibration images
image_folder = "calibration_images"

# Load all images in the folder
images = glob.glob(os.path.join(image_folder, "*.jpg"))  # Adjust for your image format

if not images:
    print("No images found in the specified folder.")
    exit()

for img_path in images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)

        # Refine corner locations for better accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Visualize detected corners (optional)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Checkerboard Detection', img)
        cv2.waitKey(500)
    else:
        print(f"Checkerboard not detected in image: {img_path}")

cv2.destroyAllWindows()

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("Camera calibration successful!")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
else:
    print("Camera calibration failed.")
