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

def calculate_distance_and_angle(contour, focal_length, actual_diameter, camera_center):
    """
    Calculate the distance and angle to the cylinder using perspective projection.
    
    Args:
        contour: Detected cylinder contour
        focal_length: Camera focal length in pixels
        actual_diameter: Real diameter of the cylinder in meters
        camera_center: Tuple of (cx, cy) representing the camera's optical center
    
    Returns:
        tuple: (distance in meters, angle in degrees)
    """
    if len(contour) < 5:
        return None, None
        
    # Fit an ellipse to get the apparent width
    (x, y), (width, height), angle = cv2.fitEllipse(contour)
    
    # Convert pixel width to meters using perspective projection formula
    # distance = (actual_width_m * focal_length_pixels) / apparent_width_pixels
    apparent_diameter = min(width, height)  # Use the smaller dimension as diameter
    distance = (actual_diameter * focal_length) / apparent_diameter
    
    # Calculate horizontal angle using arctangent
    # Angle from center of image to center of cylinder
    dx = x - camera_center[0]
    angle = np.arctan2(dx, focal_length)
    angle_degrees = np.degrees(angle)
    
    return distance, angle_degrees

def fit_cylinder_contour(contour):
    """
    Fits a more precise contour to a cylinder shape using ellipse fitting.
    Returns both the fitted contour and endpoints for visualization.
    """
    if len(contour) < 5:
        return None, None
    
    ellipse = cv2.fitEllipse(contour)
    (x, y), (width, height), angle = ellipse
    
    radians = np.deg2rad(angle)
    length = max(width, height) / 2
    dx = length * np.cos(radians)
    dy = length * np.sin(radians)
    
    endpoint1 = (int(x + dx), int(y + dy))
    endpoint2 = (int(x - dx), int(y - dy))
    
    hull = cv2.convexHull(contour)
    
    return hull, (endpoint1, endpoint2)

# Camera parameters
mtx = np.array([[568.20791041, 0.0, 341.37830129],
                [0.0, 569.81774569, 238.05524919],
                [0.0, 0.0, 1.0]])
dist = np.array([[0.08705563, 0.10725078, -0.01064468, 0.01151696, -0.33419273]])

# Constants
CYLINDER_DIAMETER = 0.1016  # meters
FOCAL_LENGTH = (mtx[0, 0] + mtx[1, 1]) / 2  # Average of fx and fy
CAMERA_CENTER = (mtx[0, 2], mtx[1, 2])  # Camera optical center (cx, cy)

# Main video capture loop
cap = cv2.VideoCapture(1)

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
        
    frame = undistort_frame(frame, mtx, dist)

    # Get current trackbar positions
    pv_hue_min = cv2.getTrackbarPos("PV Hue Min", "Filtered Video")
    pv_hue_max = cv2.getTrackbarPos("PV Hue Max", "Filtered Video")
    saturation_min = cv2.getTrackbarPos("Sat Min", "Filtered Video")
    saturation_max = cv2.getTrackbarPos("Sat Max", "Filtered Video")
    value_min = cv2.getTrackbarPos("Val Min", "Filtered Video")
    value_max = cv2.getTrackbarPos("Val Max", "Filtered Video")

    mask = filter_hsv_inverted(frame, pv_hue_min, pv_hue_max, saturation_min, saturation_max, value_min, value_max)
    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        fitted_contour, endpoints = fit_cylinder_contour(largest_contour)
        
        if fitted_contour is not None:
            # Calculate distance and angle
            distance, angle = calculate_distance_and_angle(
                largest_contour, 
                FOCAL_LENGTH,
                CYLINDER_DIAMETER,
                CAMERA_CENTER
            )
            
            # Draw the fitted contour
            cv2.drawContours(filtered_frame, [fitted_contour], -1, (0, 255, 0), 2)
            
            # Draw the major axis line
            if endpoints:
                cv2.line(filtered_frame, endpoints[0], endpoints[1], (255, 0, 0), 2)
            
            # Display distance and angle
            if distance is not None and angle is not None:
                cv2.putText(filtered_frame, 
                           f"Distance: {distance:.2f}m", 
                           (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           (0, 255, 0), 
                           2, 
                           cv2.LINE_AA)
                cv2.putText(filtered_frame, 
                           f"Angle: {angle:.2f}deg", 
                           (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           (0, 255, 0), 
                           2, 
                           cv2.LINE_AA)

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