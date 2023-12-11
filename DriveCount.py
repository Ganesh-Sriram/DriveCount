from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


cap = cv2.VideoCapture("../Videos/cars.mp4")

model = YOLO("../YOLO-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair dryer", "toothbrush"]

mask = cv2.imread("mask.png")

# Object Tracker
track_obj = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

# Tracker Limit Line
limit = [423, 297, 673, 297]

VehicleCount = []

while True:
    success, img = cap.read()
    specific_region = cv2.bitwise_and(img, mask)
    result = model(specific_region, stream=True)

    detections = np.empty((0, 5))

    for i in result:
        boxes = i.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)

            # Confidence Level
            conf = (math.ceil(box.conf[0]*100))/100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if (currentClass == "car" or currentClass == "bus" or currentClass == "truck" or currentClass == "motorbike"
                    and conf > 0.3):
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(30, y1)),
                #                    scale=0.7, thickness=1, offset=3)

                # Adding records to the 'detections' list
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    TrackerResult = track_obj.update(detections)

    cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (0, 0, 255), 3)

    for tr in TrackerResult:
        x1, y1, x2, y2, id = tr
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(tr)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(30, y1)),
                           scale=0.7, thickness=1, offset=3)

        # Point marking for all tracking objects
        cx, cy = x1+(x2-x1)//2, y1+(y2-y1)//2
        cv2.circle(img, (cx, cy), 5, (0, 130, 255), cv2.FILLED)

        # Vehicle Count
        if limit[0] < cx < limit[2] and limit[1]-20 < cy < limit[3]+20:
            if VehicleCount.count(id) == 0:
                VehicleCount.append(id)
                cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (0, 255, 0), 3)

    cvzone.putTextRect(img, f'Vehicles Passed: {len(VehicleCount)}', (50, 50), colorR=(255, 255, 255), colorT=(0, 0, 0))

    cv2.imshow("Image", img)
    cv2.waitKey(0)
