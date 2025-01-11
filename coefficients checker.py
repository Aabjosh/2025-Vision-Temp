import cv2
import numpy as np
import glob

def calibrate_camera(images_path, chessboard_size=(9, 6), square_size=1.0):
    """
    Computes the intrinsic matrix and distortion coefficients of the camera.

    Args:
        images_path (str): Path to the calibration images (e.g., 'calibration_images/*.jpg').
        chessboard_size (tuple): Number of inner corners in the chessboard (cols, rows).
        square_size (float): Size of each square in the chessboard (real-world units, e.g., cm).

    Returns:
        tuple: (camera_matrix, distortion_coeffs, rvecs, tvecs)
    """

    # Prepare object points (3D points in the real world for the chessboard pattern)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Read all images for calibration
    images = glob.glob(images_path)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners for visual confirmation
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Perform camera calibration
    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("Camera calibration successful!")
        print("Intrinsic Matrix (Camera Matrix):")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(distortion_coeffs)
    else:
        print("Camera calibration failed!")

    return camera_matrix, distortion_coeffs, rvecs, tvecs

if __name__ == "__main__":
    # Path to calibration images (adjust path to your image folder)
    images_path = 'calibration_images/*.jpg'  # Change to your image path
    chessboard_size = (9, 6)  # Adjust based on your chessboard pattern
    square_size = 2.5  # Set the size of one square in cm (or any other unit)

    camera_matrix, distortion_coeffs, rvecs, tvecs = calibrate_camera(images_path, chessboard_size, square_size)
