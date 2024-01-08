import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Video capture from file
cap = cv2.VideoCapture("Videos/cars.mp4")

# YOLO model initialization
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Class names for object detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Mask image for bitwise AND operation
mask = cv2.imread("mask.png")

# Tracking using SORT (Simple Online and Realtime Tracking) algorithm
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Region of interest limits for counting
limits = [400, 297, 673, 297]

# List to store the unique IDs of tracked objects
totalCount = []

while True:
    # Read a frame from the video
    success, img = cap.read()
    if not success:
        break

    # Apply bitwise AND operation with the mask
    imgRegion = cv2.bitwise_and(img, cv2.resize(mask, (img.shape[1], img.shape[0])))

    # Overlay graphics on the frame
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    # Perform object detection using YOLO
    results = model(imgRegion, stream=True)

    # Initialize empty array for detections
    detections = np.empty((0, 5))

    # Process YOLO results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Check if the detected object is a vehicle with sufficient confidence
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update object tracking using SORT
    resultsTracker = tracker.update(detections)

    # Draw region limits on the frame
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # Process tracked results
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Draw rectangle and ID for each tracked object
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        # Draw a circle at the center of the tracked object
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if the object crosses the counting line
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Display the count on the frame
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # Display the frame
    cv2.imshow("Image", img)

    key = cv2.waitKey(1) & 0xFF  # Masking to get the least significant byte
    if key == ord('q'):  # Check if the 'q' key is pressed
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
