import cv2
import numpy as np
import time

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

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return mask

def draw_3d_box(frame, rect, height_factor=1.5):
    """
    Draws a 3D bounding box given a 2D rotated rectangle.
    
    Args:
        frame: The input image to draw on
        rect: The cv2.minAreaRect object
        height_factor: Factor to determine the height of the 3D box (adjusts perspective)
    """
    # Get the corner points of the 2D rectangle base
    box_points = cv2.boxPoints(rect)
    box_points = np.int32(box_points)  # Changed from int0 to int32
    
    # Calculate the center and dimensions
    (center_x, center_y), (width, height), angle = rect
    
    # Create top points by shifting the base points up
    # The shift amount is determined by the height_factor and the object's width
    perspective_factor = width * 0.1  # Adjust this to change the perspective effect
    top_points = box_points.copy()
    
    # Shift top points up and adjust for perspective
    for i in range(4):
        # Move points up
        top_points[i][1] -= int(height * height_factor)
        # Add perspective effect
        if box_points[i][0] > center_x:
            top_points[i][0] += int(perspective_factor)
        else:
            top_points[i][0] -= int(perspective_factor)

    # Draw the base in green
    cv2.drawContours(frame, [box_points], 0, (0, 255, 0), 2)
    
    # Draw the top in blue
    cv2.drawContours(frame, [top_points], 0, (255, 0, 0), 2)
    
    # Draw the vertical edges in red
    for i in range(4):
        pt1 = tuple(box_points[i])
        pt2 = tuple(top_points[i])
        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

# Main video capture loop
cap = cv2.VideoCapture(1)  # Use 0 for the default camera

# PhotonVision HSV range values
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
cv2.createTrackbar("Height Factor", "Filtered Video", 15, 30, nothing)  # Height factor * 0.1

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
    height_factor = cv2.getTrackbarPos("Height Factor", "Filtered Video") / 10.0

    # Filter by HSV values
    mask = filter_hsv_inverted(frame, pv_hue_min, pv_hue_max, saturation_min, saturation_max, value_min, value_max)

    # Apply the mask to the original frame
    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours and the largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        draw_3d_box(filtered_frame, rect, height_factor)

    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(filtered_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the filtered frame and the mask
    cv2.imshow("Filtered Video", filtered_frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()