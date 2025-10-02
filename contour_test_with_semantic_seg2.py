import cv2
import numpy as np

# Open camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # ----------------- Define trapezoid -----------------
    pts = np.array([
        [int(width * 0.7), int(height * 0.66)],  # top-right
        [int(width * 0.3), int(height * 0.66)],  # top-left
        [0, height],                             # bottom-left
        [width, height]                          # bottom-right
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Create trapezoid mask
    trapezoid_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(trapezoid_mask, [pts], 255)

    # ----------------- Grayscale and threshold -----------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # ----------------- Find contours -----------------
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Start with trapezoid area fully red
    segmented = frame.copy()
    cv2.fillPoly(segmented, [pts], (0, 0, 255))  # obstacles = red

    # ----------------- Find free space -----------------
    max_area = 0
    free_space = None
    for cnt in contours:
        # Check if contour intersects trapezoid
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
        if cv2.countNonZero(cv2.bitwise_and(contour_mask, trapezoid_mask)) > 0:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                free_space = cnt

    # Draw free space green
    if free_space is not None:
        green_overlay = np.zeros_like(frame)
        cv2.drawContours(green_overlay, [free_space], -1, (0, 255, 0), -1)
        green_mask = cv2.bitwise_and(green_overlay, green_overlay, mask=trapezoid_mask)
        segmented = cv2.add(segmented, green_mask)

    # ----------------- Compute path center for PID -----------------
    path_rows = [int(height*0.75), int(height*0.8), int(height*0.85), int(height*0.9)]
    centers = []

    # Create binary mask of free space for center calculation
    free_mask = np.zeros((height, width), dtype=np.uint8)
    if free_space is not None:
        cv2.drawContours(free_mask, [free_space], -1, 255, -1)
        free_mask = cv2.bitwise_and(free_mask, trapezoid_mask)

    for row_y in path_rows:
        row_pixels = np.where(free_mask[row_y, :] > 0)[0]
        if len(row_pixels) > 0:
            centers.append(np.mean(row_pixels))

    if centers:
        path_center = int(np.mean(centers))
        error = path_center - (width // 2)  # PID error
        # Draw path center
        cv2.circle(segmented, (path_center, int(height*0.9)), 7, (0, 255, 0), -1)
        cv2.putText(segmented, f"Error: {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Draw trapezoid boundary
    cv2.polylines(segmented, [pts], True, (255, 255, 255), 2)

    # ----------------- Display -----------------
    cv2.imshow("Free Space and Obstacles", segmented)
    cv2.imshow("Original Frame", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
