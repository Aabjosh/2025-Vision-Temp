import cv2
import numpy as np
import time

def filter_hsv_inverted(frame, pv_hue_min, pv_hue_max, saturation_min, saturation_max, value_min, value_max):
    """
    Filters an image based on inverted hue values from PhotonVision and HSV ranges.

    Args:
        frame: The input image (BGR format).
        pv_hue_min: Minimum hue value from PhotonVision (0-360).
        pv_hue_max: Maximum hue value from PhotonVision (0-360).
        saturation_min: Minimum saturation value (0-255).
        saturation_max: Maximum saturation value (0-255).
        value_min: Minimum value (brightness) value (0-255).
        value_max: Maximum value (brightness) value (0-255).

    Returns:
        A binary mask where white pixels represent pixels within the specified HSV range.
    """
    # Convert from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Invert and convert PhotonVision hue to OpenCV hue
    def invert_and_scale_hue(pv_hue):
        inverted_hue = 360 - pv_hue if pv_hue != 0 else 0
        return int(inverted_hue / 2)

    cv_hue_min = invert_and_scale_hue(pv_hue_min)
    cv_hue_max = invert_and_scale_hue(pv_hue_max)

    # Ensure hue_min is always less than hue_max for inRange
    if cv_hue_min > cv_hue_max:
        cv_hue_min, cv_hue_max = cv_hue_max, cv_hue_min

    # Define the lower and upper bounds of the HSV range
    lower_bound = np.array([cv_hue_min, saturation_min, value_min])
    upper_bound = np.array([cv_hue_max, saturation_max, value_max])

    # Create a mask using inRange
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    return mask

# Main video capture loop
cap = cv2.VideoCapture(1)  # Use 0 for the default camera

# PhotonVision HSV range values (example values)
pv_hue_min = 0  # Example: Replace with actual PhotonVision values
pv_hue_max = 1  # Example: Replace with actual PhotonVision values
saturation_min = 0
saturation_max = 255
value_min = 147
value_max = 255

# Create trackbars for interactive adjustment
def nothing(x):
    pass

cv2.namedWindow("Filtered Video")
cv2.createTrackbar("PV Hue Min", "Filtered Video", pv_hue_min, 360, nothing)
cv2.createTrackbar("PV Hue Max", "Filtered Video", pv_hue_max, 360, nothing)
cv2.createTrackbar("Sat Min", "Filtered Video", saturation_min, 255, nothing)
cv2.createTrackbar("Sat Max", "Filtered Video", saturation_max, 255, nothing)
cv2.createTrackbar("Val Min", "Filtered Video", value_min, 255, nothing)
cv2.createTrackbar("Val Max", "Filtered Video", value_max, 255, nothing)

# FPS calculation variables
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get current trackbar positions
    pv_hue_min = cv2.getTrackbarPos("PV Hue Min", "Filtered Video")
    pv_hue_max = cv2.getTrackbarPos("PV Hue Max", "Filtered Video")
    saturation_min = cv2.getTrackbarPos("Sat Min", "Filtered Video")
    saturation_max = cv2.getTrackbarPos("Sat Max", "Filtered Video")
    value_min = cv2.getTrackbarPos("Val Min", "Filtered Video")
    value_max = cv2.getTrackbarPos("Val Max", "Filtered Video")

    # Filter by HSV values (with inverted hue from PhotonVision)
    mask = filter_hsv_inverted(frame, pv_hue_min, pv_hue_max, saturation_min, saturation_max, value_min, value_max)

    # Apply the mask to the original frame
    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours and the largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea, default=[])

    if len(largest_contour) > 0:
        rect = cv2.minAreaRect(largest_contour)
        (x, y), (w, h), angle = rect
        points = cv2.boxPoints(rect)
        points = np.int32(points)
        cv2.polylines(filtered_frame, [points], True, (0, 255, 0), 2)

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS on the filtered frame
    cv2.putText(filtered_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the filtered frame and the mask
    cv2.imshow("Filtered Video", filtered_frame)
    cv2.imshow("Mask", mask)

    # Exit on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()