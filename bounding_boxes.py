import cv2
import numpy as np
import time

def undistort_frame(frame, camera_matrix, dist_coeffs):
    """
    Undistort a frame using camera calibration parameters.
    """
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)
    return undistorted

def filter_hsv_inverted(frame, pv_hue_min, pv_hue_max, saturation_min, saturation_max, value_min, value_max):
    """
    Filters an image based on inverted hue values from PhotonVision and HSV ranges.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    def invert_and_scale_hue(pv_hue):
        inverted_hue = 360 - pv_hue if pv_hue != 0 else 0
        return int(inverted_hue / 2)

    cv_hue_min = invert_and_scale_hue(pv_hue_min)
    cv_hue_max = invert_and_scale_hue(pv_hue_max)

    if cv_hue_min > cv_hue_max:
        cv_hue_min, cv_hue_max = cv_hue_max, cv_hue_min

    lower_bound = np.array([cv_hue_min, saturation_min, value_min])
    upper_bound = np.array([cv_hue_max, saturation_max, value_max])

    return cv2.inRange(hsv, lower_bound, upper_bound)

def fit_cylinder_contour(contour):
    """
    Fits a more precise contour to a cylinder shape using ellipse fitting.
    Returns both the fitted contour and endpoints for visualization.
    """
    if len(contour) < 5:  # Need at least 5 points to fit an ellipse
        return None, None
    
    # Fit an ellipse to get orientation
    ellipse = cv2.fitEllipse(contour)
    (x, y), (width, height), angle = ellipse
    
    # Calculate major axis endpoints
    radians = np.deg2rad(angle)
    length = max(width, height) / 2
    dx = length * np.cos(radians)
    dy = length * np.sin(radians)
    
    endpoint1 = (int(x + dx), int(y + dy))
    endpoint2 = (int(x - dx), int(y - dy))
    
    # Create a more precise contour using convex hull
    hull = cv2.convexHull(contour)
    
    return hull, (endpoint1, endpoint2)

# Camera parameters
mtx = np.array([[568.20791041, 0.0, 341.37830129],
                [0.0, 569.81774569, 238.05524919],
                [0.0, 0.0, 1.0]])
dist = np.array([[0.08705563, 0.10725078, -0.01064468, 0.01151696, -0.33419273]])

# Main video capture loop
cap = cv2.VideoCapture(1)  # Use 0 for default camera

# Initial HSV values
pv_hue_min = 0
pv_hue_max = 1
saturation_min = 0
saturation_max = 255
value_min = 147
value_max = 255

# Create trackbars
def nothing(x):
    pass

cv2.namedWindow("Filtered Video")
cv2.createTrackbar("PV Hue Min", "Filtered Video", pv_hue_min, 360, nothing)
cv2.createTrackbar("PV Hue Max", "Filtered Video", pv_hue_max, 360, nothing)
cv2.createTrackbar("Sat Min", "Filtered Video", saturation_min, 255, nothing)
cv2.createTrackbar("Sat Max", "Filtered Video", saturation_max, 255, nothing)
cv2.createTrackbar("Val Min", "Filtered Video", value_min, 255, nothing)
cv2.createTrackbar("Val Max", "Filtered Video", value_max, 255, nothing)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Undistort the frame
    frame = undistort_frame(frame, mtx, dist)

    # Get current trackbar positions
    pv_hue_min = cv2.getTrackbarPos("PV Hue Min", "Filtered Video")
    pv_hue_max = cv2.getTrackbarPos("PV Hue Max", "Filtered Video")
    saturation_min = cv2.getTrackbarPos("Sat Min", "Filtered Video")
    saturation_max = cv2.getTrackbarPos("Sat Max", "Filtered Video")
    value_min = cv2.getTrackbarPos("Val Min", "Filtered Video")
    value_max = cv2.getTrackbarPos("Val Max", "Filtered Video")

    # Filter by HSV values
    mask = filter_hsv_inverted(frame, pv_hue_min, pv_hue_max, saturation_min, saturation_max, value_min, value_max)

    # Apply the mask
    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit the cylinder contour
        fitted_contour, endpoints = fit_cylinder_contour(largest_contour)
        
        if fitted_contour is not None:
            # Draw the fitted contour
            cv2.drawContours(filtered_frame, [fitted_contour], -1, (0, 255, 0), 2)
            
            # Draw the major axis line
            if endpoints:
                cv2.line(filtered_frame, endpoints[0], endpoints[1], (255, 0, 0), 2)

    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(filtered_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the results
    cv2.imshow("Filtered Video", filtered_frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()